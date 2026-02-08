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
A single writer is assumed per buffer instance.  Multiple readers are safe
because they only *read* the payload area and atomically *write* ``read_ts``
(a single 8-byte ``float64`` store, naturally aligned → atomic on x86/ARM64).
The ``write_index`` is an ``mp.Value('Q')`` protected by its built-in lock.
"""

from __future__ import annotations

import multiprocessing as mp
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

from .schema import HEADER_SIZE, HEADER_STRUCT, SCHEMA_VERSION, DTypeTag, SlotHeader


class SharedRingBuffer:
    """A shared-memory ring buffer with fixed-size slots.

    Parameters
    ----------
    name : str
        Unique name for the shared memory block (e.g. ``"frame_buf"``).
    slots : int
        Number of slots in the ring.
    slot_payload_size : int
        Maximum payload size (bytes) per slot.  Payloads smaller than this
        are zero-padded; payloads larger will be **truncated** (a bug guard).
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
            # Clean up stale block if it exists from a previous crash
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

        # Shared write counter — only the producer should increment this.
        # We use a multiprocessing Value so it's visible across processes.
        self._seq_counter_name = f"_seq_{name}"
        if create:
            self._seq = mp.Value("Q", 0, lock=True)  # uint64
        else:
            # Readers don't need the counter; they scan slot headers.
            self._seq = None

    #  helpers 

    def _slot_offset(self, index: int) -> int:
        """Byte offset of slot *index* in the shared memory block."""
        return index * self.slot_size

    #  write (producer) 

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

        # Write header + payload into shm
        header_bytes = header.pack()
        self._shm.buf[offset : offset + HEADER_SIZE] = header_bytes
        self._shm.buf[offset + HEADER_SIZE : offset + HEADER_SIZE + payload_len] = payload

        return header

    #  read (consumer) ─

    def read_latest(self) -> Optional[Tuple[SlotHeader, bytes]]:
        """Return ``(header, payload_bytes)`` for the slot with the highest
        ``seq_no``, or *None* if no slot has been written yet.

        After reading, ``read_ts`` in the slot header is updated to
        ``time.time()`` so producers/monitors can track consumer lag.
        """
        best_seq = 0
        best_idx = -1

        for i in range(self.slots):
            offset = self._slot_offset(i)
            raw_header = bytes(self._shm.buf[offset : offset + HEADER_SIZE])
            # Quick extract of just seq_no (first 8 bytes, little-endian uint64)
            (seq_no,) = struct.unpack_from("<Q", raw_header, 0)
            if seq_no > best_seq:
                best_seq = seq_no
                best_idx = i

        if best_idx < 0:
            return None

        offset = self._slot_offset(best_idx)
        raw_header = bytes(self._shm.buf[offset : offset + HEADER_SIZE])
        header = SlotHeader.unpack(raw_header)

        payload = bytes(
            self._shm.buf[
                offset + HEADER_SIZE : offset + HEADER_SIZE + header.payload_size
            ]
        )

        # Stamp read_ts (offset 16 in the header is the read_ts float64)
        now = time.time()
        struct.pack_into("<d", self._shm.buf, offset + 16, now)
        header.read_ts = now

        return header, payload

    #  lifecycle ─

    def close(self) -> None:
        """Close (but don't unlink) the shared memory handle."""
        self._shm.close()

    def unlink(self) -> None:
        """Unlink (destroy) the shared memory block.  Only call from the
        process that created it (typically the orchestrator)."""
        self._shm.close()
        self._shm.unlink()

    # convenience for ``with`` blocks
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
