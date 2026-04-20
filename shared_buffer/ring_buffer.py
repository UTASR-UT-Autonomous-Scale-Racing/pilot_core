"""
SharedRingBuffer — fixed-slot, single-producer / multi-consumer ring buffer
backed by multiprocessing.shared_memory.

Design goals
------------
* **Zero-copy** for readers on the same machine (shared memory, no pickle).
* **Non-blocking** writes — old slots are silently overwritten.
* **Pop-most-recent** reads — consumers always get the freshest data.
* **Timestamp contract** — every slot carries ``produced_ts`` and ``read_ts``.

Thread / process safety
-----------------------
A single writer is assumed per buffer instance.  Slots follow a seqlock
protocol so readers never see a torn record:

1. the writer zeroes the slot's ``seq_no`` (readers skip the slot),
2. writes the payload,
3. publishes the header with the new ``seq_no`` last;

the reader copies header + payload, then re-reads ``seq_no`` and retries if
the writer lapped the slot mid-copy.  Readers *write* ``read_ts`` as a
single naturally aligned 8-byte ``float64`` store (atomic on x86/ARM64).
The write counter is an ``mp.Value('Q')`` owned by the creating handle and
protected by its built-in lock.
"""

from __future__ import annotations

import multiprocessing as mp
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

from .schema import HEADER_SIZE, SCHEMA_VERSION, DTypeTag, SlotHeader

# Seqlock read attempts before giving up for this call (writer lapping us
# repeatedly means the reader is far too slow for the slot count anyway).
_READ_RETRIES = 8


class SharedRingBuffer:
    """A shared-memory ring buffer with fixed-size slots.

    Parameters
    ----------
    name : str
        Unique name for the shared memory block (e.g. ``"frame_buf"``).
    slots : int
        Number of slots in the ring.
    slot_payload_size : int
        Maximum payload size (bytes) per slot.  Larger payloads are
        **truncated** so a write can never overflow into the next slot.
    dtype_tag : DTypeTag
        The type tag written into every slot header.
    create : bool
        If *True* (default for the orchestrator), create the shm block.
        Workers that only read should pass *False* and attach to an existing
        block by *name*.
    """

    def __init__(
        self,
        name: str,
        slots: int,
        slot_payload_size: int,
        dtype_tag: DTypeTag = DTypeTag.UNKNOWN,
        create: bool = True,
    ) -> None:
        self.name = name
        self.slots = slots
        self.slot_payload_size = slot_payload_size
        self.slot_size = HEADER_SIZE + slot_payload_size
        self.dtype_tag = dtype_tag
        self._total_size = self.slots * self.slot_size

        if create:
            # Remove a stale block left behind by a crashed run
            try:
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
            # Zero-initialise (seq_no == 0 means "unwritten")
            self._shm.buf[: self._total_size] = b"\x00" * self._total_size
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)

        # Write counter shared across processes; only the producer increments it.
        if create:
            self._seq = mp.Value("Q", 0, lock=True)  # uint64
        else:
            # Readers don't need the counter; they scan slot headers.
            self._seq = None

    # --- helpers ---

    def _slot_offset(self, index: int) -> int:
        """Byte offset of slot *index* in the shared memory block."""
        return index * self.slot_size

    # --- write (producer) ---

    def write(self, payload: bytes) -> SlotHeader:
        """Write *payload* into the next slot, returning the header written.

        The caller is responsible for serialising their domain object into
        ``bytes`` before calling this method (typed wrappers do this).
        """
        if self._seq is None:
            raise RuntimeError("Cannot write from a read-only buffer handle.")

        payload_len = len(payload)
        if payload_len > self.slot_payload_size:
            payload = payload[: self.slot_payload_size]
            payload_len = self.slot_payload_size

        with self._seq.get_lock():
            self._seq.value += 1
            seq_no = self._seq.value

        slot_idx = (seq_no - 1) % self.slots
        offset = self._slot_offset(slot_idx)

        header = SlotHeader(
            seq_no=seq_no,
            produced_ts=time.time(),
            read_ts=0.0,
            payload_size=payload_len,
            schema_version=SCHEMA_VERSION,
            dtype_tag=int(self.dtype_tag),
        )

        # Seqlock publish: invalidate → payload → header (seq_no last)
        struct.pack_into("<Q", self._shm.buf, offset, 0)
        self._shm.buf[offset + HEADER_SIZE : offset + HEADER_SIZE + payload_len] = payload
        self._shm.buf[offset : offset + HEADER_SIZE] = header.pack()

        return header

    # --- read (consumer) ---

    def read_latest(self) -> Optional[Tuple[SlotHeader, bytes]]:
        """Return ``(header, payload_bytes)`` for the slot with the highest
        ``seq_no``, or *None* if no complete slot has been written yet.

        After reading, ``read_ts`` in the slot header is updated to
        ``time.time()`` so producers/monitors can track consumer lag.
        """
        for _ in range(_READ_RETRIES):
            result = self._try_read_latest()
            if result is not None or not self._any_slot_written():
                return result
        return None

    def _any_slot_written(self) -> bool:
        return any(
            struct.unpack_from("<Q", self._shm.buf, self._slot_offset(i))[0]
            for i in range(self.slots)
        )

    def _try_read_latest(self) -> Optional[Tuple[SlotHeader, bytes]]:
        """One seqlock read attempt; *None* if the slot changed mid-copy."""
        best_seq = 0
        best_idx = -1

        for i in range(self.slots):
            offset = self._slot_offset(i)
            raw_header = bytes(self._shm.buf[offset : offset + HEADER_SIZE])
            # seq_no is the first 8 bytes (little-endian uint64)
            (seq_no,) = struct.unpack_from("<Q", raw_header, 0)
            if seq_no > best_seq:
                best_seq = seq_no
                best_idx = i

        if best_idx < 0:
            return None

        offset = self._slot_offset(best_idx)
        raw_header = bytes(self._shm.buf[offset : offset + HEADER_SIZE])
        header = SlotHeader.unpack(raw_header)

        if header.seq_no != best_seq:
            return None  # overwritten between scan and header read

        payload = bytes(
            self._shm.buf[
                offset + HEADER_SIZE : offset + HEADER_SIZE + header.payload_size
            ]
        )

        # Validate: the writer must not have touched the slot during the copy
        (seq_after,) = struct.unpack_from("<Q", self._shm.buf, offset)
        if seq_after != header.seq_no:
            return None

        # read_ts is the float64 at header offset 16 (see schema.py)
        now = time.time()
        struct.pack_into("<d", self._shm.buf, offset + 16, now)
        header.read_ts = now

        return header, payload

    # --- lifecycle ---

    def close(self) -> None:
        """Close (but don't unlink) the shared memory handle."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink (destroy) the shared memory block.  Only call from the
        process that created it (typically the orchestrator)."""
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
