"""
Fixed-slot ring buffer on top of multiprocessing.shared_memory. One producer,
any number of consumers.

Readers on the same box get the data without pickling. Writes never block,
the oldest slot just gets overwritten, and reads always return the newest
slot rather than draining a queue. Every slot carries produced_ts and read_ts
so we can measure lag end to end.

Only one writer per buffer is supported. Slots use a seqlock so a reader
can't see a half-written record: the writer zeroes seq_no first (readers
skip zeroed slots), then writes the payload, then writes the header with the
new seq_no last. The reader copies header + payload, re-reads seq_no, and
retries if the writer lapped it mid-copy.

Readers do write one thing: read_ts, as a single aligned 8-byte float64
store, which is atomic on x86 and ARM64. The write counter is an
mp.Value('Q') held by whoever created the buffer, guarded by its own lock.
"""

from __future__ import annotations

import multiprocessing as mp
import struct
import time
from multiprocessing import shared_memory
from typing import Optional, Tuple

from .schema import HEADER_SIZE, SCHEMA_VERSION, DTypeTag, SlotHeader

# seqlock retries per read_latest() call. If the writer laps us this many
# times in a row the reader is way too slow for the slot count anyway
_READ_RETRIES = 8


class SharedRingBuffer:
    """Shared-memory ring buffer with fixed-size slots.

    Parameters
    ----------
    name : str
        Name of the shm block, e.g. "frame_buf".
    slots : int
        Number of slots in the ring.
    slot_payload_size : int
        Max payload bytes per slot. Anything longer gets truncated so a
        write can't spill into the next slot.
    dtype_tag : DTypeTag
        Tag stamped into every slot header.
    create : bool
        True creates the block (the orchestrator does this). Read-only
        workers pass False and attach to the existing block by name.
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
            # a crashed run can leave the block behind, clear it first
            try:
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
            # seq_no == 0 means unwritten
            self._shm.buf[: self._total_size] = b"\x00" * self._total_size
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)

        # cross-process write counter, only the producer bumps it
        if create:
            self._seq = mp.Value("Q", 0, lock=True)  # uint64
        else:
            # readers scan slot headers instead
            self._seq = None

    def _slot_offset(self, index: int) -> int:
        return index * self.slot_size

    def write(self, payload: bytes) -> SlotHeader:
        """Write raw bytes into the next slot and return the header.

        Serialising is the caller's job; the typed wrappers handle that.
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

        # seqlock publish: invalidate -> payload -> header (seq_no goes last)
        struct.pack_into("<Q", self._shm.buf, offset, 0)
        self._shm.buf[offset + HEADER_SIZE : offset + HEADER_SIZE + payload_len] = payload
        self._shm.buf[offset : offset + HEADER_SIZE] = header.pack()

        return header

    def read_latest(self) -> Optional[Tuple[SlotHeader, bytes]]:
        """(header, payload) for the slot with the highest seq_no, or None
        if nothing complete has been written yet.

        Also stamps read_ts in the slot header with time.time(), which is
        how the monitors track consumer lag.
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
        """Single seqlock attempt. None if the slot changed under us."""
        best_seq = 0
        best_idx = -1

        for i in range(self.slots):
            offset = self._slot_offset(i)
            raw_header = bytes(self._shm.buf[offset : offset + HEADER_SIZE])
            # seq_no = first 8 bytes, LE uint64
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

        # writer touched the slot during the copy? then throw it away
        (seq_after,) = struct.unpack_from("<Q", self._shm.buf, offset)
        if seq_after != header.seq_no:
            return None

        # read_ts lives at header offset 16, see schema.py
        now = time.time()
        struct.pack_into("<d", self._shm.buf, offset + 16, now)
        header.read_ts = now

        return header, payload

    def close(self) -> None:
        """Close our handle. Doesn't unlink."""
        self._shm.close()

    def unlink(self) -> None:
        """Destroy the shm block. Only the creating process (normally the
        orchestrator) should call this."""
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
