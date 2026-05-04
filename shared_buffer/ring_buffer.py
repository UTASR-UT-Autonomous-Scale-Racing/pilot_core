"""
Fixed-slot ring buffer on top of multiprocessing.shared_memory. One producer,
any number of consumers.

Readers on the same box get the data without pickling. Writes never block,
the oldest slot just gets overwritten. read_latest() returns the newest slot;
read_since(seq) returns every slot still in the ring that is newer than seq
(for consumers such as the recorder that must not drop samples). Every slot
carries produced_ts and read_ts so we can measure lag end to end.

Slot layout (slot_size is rounded up to 64 bytes so every slot, and therefore
every 8-byte seq field, stays 8-byte aligned):

    [64 bytes]            SlotHeader (schema.py), seq_no in the first 8 bytes
    [slot_payload_size]   payload (padded to a multiple of 8)
    [8 bytes]             uint64 trailing copy of seq_no

Seqlock protocol. Only one writer per buffer is supported.

  writer:  1. header with seq_no = 0 (readers skip zeroed slots), trailer = 0
           2. payload
           3. trailing seq_no          (separate 8-byte struct.pack_into)
           4. header seq_no            (separate 8-byte struct.pack_into, last)
  reader:  1. copy the header, seq_before = its seq_no (skip if 0)
           2. copy the payload
           3. read the trailing seq_no, then re-read the header seq_no
           4. accept only if seq_before == trailing == seq_after

On weakly ordered CPUs (the Jetson's ARM cores) stores from one memcpy can
become visible out of order; that is why seq_no is never published in the
same memcpy as the rest of the header and why the trailer, written after the
payload, is compared too. Python cannot emit a DMB directly, so between the
steps both sides also take and release a process-local lock (_fence), whose
atomic instructions order the surrounding loads/stores in practice. This is
best effort, not a formal C11 guarantee; tests/test_ring_buffer_consistency.py
checks that readers never get a torn record.

Readers only write into shared memory when they ask for it: read_latest(
stamp=True) / read_since(..., stamp=True) store read_ts as a single aligned
8-byte float64 (atomic on x86 and ARM64). Plain polls never touch the block.

The write counter is an mp.Value('Q', lock=False) created by whoever created
the buffer and inherited by the producer process (also by a producer that
start.py restarts after a crash). No lock is needed: there is exactly one live
writer at a time, because start.py restarts a worker only after the previous
process has exited. A lock would be actively harmful here: a producer killed
while holding it would leave it locked and the restarted producer would block
forever in write().
"""

from __future__ import annotations

import multiprocessing as mp
import struct
import threading
import time
from multiprocessing import shared_memory
from typing import List, Optional, Tuple

from .schema import HEADER_SIZE, SCHEMA_VERSION, DTypeTag, SlotHeader

# seqlock retries per read_latest() call. If the writer laps us this many
# times in a row the reader is way too slow for the slot count anyway
_READ_RETRIES = 8

_SEQ = struct.Struct("<Q")
_TS = struct.Struct("<d")
_READ_TS_OFFSET = 16      # see schema.py
_TRAILER_SIZE = 8
_SLOT_ALIGN = 64

_fence_lock = threading.Lock()


def _fence() -> None:
    """Best-effort full memory barrier (see module docstring)."""
    _fence_lock.acquire()
    _fence_lock.release()


def _round_up(n: int, a: int) -> int:
    return (n + a - 1) // a * a


class SharedRingBuffer:
    """Shared-memory ring buffer with fixed-size slots.

    Parameters
    ----------
    name : str
        Name of the shm block, e.g. "frame_buf".
    slots : int
        Number of slots in the ring.
    slot_payload_size : int
        Max payload bytes per slot. Larger writes raise ValueError.
    dtype_tag : DTypeTag
        Tag stamped into every slot header.
    create : bool
        True creates the block (the orchestrator does this). Read-only
        workers pass False and attach to the existing block by name; they
        must pass the same slots / slot_payload_size.
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
        self._trailer_rel = HEADER_SIZE + _round_up(slot_payload_size, 8)
        self.slot_size = _round_up(self._trailer_rel + _TRAILER_SIZE, _SLOT_ALIGN)
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
            # a fresh POSIX shm block is zero-filled by the kernel, and
            # seq_no == 0 already means "unwritten", so no explicit clear
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total_size
            )
            # cross-process write counter; single live writer, see docstring
            self._seq = mp.Value("Q", 0, lock=False)  # uint64
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            if self._shm.size < self._total_size:
                size = self._shm.size
                self._shm.close()
                raise ValueError(
                    f"shm block {name!r} is {size} bytes, expected at least "
                    f"{self._total_size} for {slots} x {slot_payload_size}-byte slots"
                )
            # readers scan slot headers instead
            self._seq = None

    def _slot_offset(self, index: int) -> int:
        return index * self.slot_size

    def _slot_seq(self, index: int) -> int:
        return _SEQ.unpack_from(self._shm.buf, index * self.slot_size)[0]

    # writing

    def write(self, payload: bytes, ts: Optional[float] = None) -> SlotHeader:
        """Write raw bytes into the next slot and return the header.

        ``ts`` is the sample's time on the ``time.monotonic()`` clock (e.g. the
        camera capture time); defaults to now.  Serialising is the caller's
        job; the typed wrappers handle that.
        """
        if self._seq is None:
            raise RuntimeError("Cannot write from a read-only buffer handle.")

        payload_len = len(payload)
        if payload_len > self.slot_payload_size:
            raise ValueError(
                f"payload of {payload_len} bytes exceeds the slot size of {self.slot_payload_size}"
            )

        self._seq.value += 1
        seq_no = self._seq.value

        offset = self._slot_offset((seq_no - 1) % self.slots)
        buf = self._shm.buf

        header = SlotHeader(
            seq_no=0,
            produced_ts=time.monotonic() if ts is None else float(ts),
            read_ts=0.0,
            payload_size=payload_len,
            schema_version=SCHEMA_VERSION,
            dtype_tag=int(self.dtype_tag),
        )

        # 1. invalidate: header with seq_no = 0 first, then clear the trailer
        _SEQ.pack_into(buf, offset, 0)
        _fence()
        buf[offset : offset + HEADER_SIZE] = header.pack()
        _SEQ.pack_into(buf, offset + self._trailer_rel, 0)
        # 2. payload
        buf[offset + HEADER_SIZE : offset + HEADER_SIZE + payload_len] = payload
        _fence()
        # 3. trailing seq, 4. header seq (publish)
        _SEQ.pack_into(buf, offset + self._trailer_rel, seq_no)
        _fence()
        _SEQ.pack_into(buf, offset, seq_no)

        header.seq_no = seq_no
        return header

    # reading

    def latest_seq(self) -> int:
        """Highest published seq_no (0 if nothing written), without copying data.

        Lets pollers skip the full payload copy when nothing new has arrived.
        """
        return max(self._slot_seq(i) for i in range(self.slots))

    def _read_slot(
        self, index: int, expected_seq: int = 0, stamp: bool = False
    ) -> Optional[Tuple[SlotHeader, bytes]]:
        """One seqlock attempt on one slot. None if the slot is empty, torn,
        or (with expected_seq) no longer holds that record."""
        offset = self._slot_offset(index)
        buf = self._shm.buf

        header = SlotHeader.unpack(bytes(buf[offset : offset + HEADER_SIZE]))
        seq_before = header.seq_no
        if seq_before == 0 or (expected_seq and seq_before != expected_seq):
            return None
        if header.payload_size > self.slot_payload_size:
            return None  # header from a half-published write
        _fence()
        payload = bytes(
            buf[offset + HEADER_SIZE : offset + HEADER_SIZE + header.payload_size]
        )
        _fence()
        trailer = _SEQ.unpack_from(buf, offset + self._trailer_rel)[0]
        seq_after = _SEQ.unpack_from(buf, offset)[0]
        if not (seq_before == trailer == seq_after):
            return None  # writer touched the slot during the copy

        now = time.monotonic()
        header.read_ts = now
        if stamp:
            _TS.pack_into(buf, offset + _READ_TS_OFFSET, now)
        return header, payload

    def read_latest(self, stamp: bool = False) -> Optional[Tuple[SlotHeader, bytes]]:
        """(header, payload) for the slot with the highest seq_no, or None
        if nothing complete has been written yet.

        header.read_ts is set to the local read time. Only with stamp=True is
        it also stored in the shared slot header (that's how monitors and the
        recorder track consumer lag); plain polls don't write to shared memory.
        """
        for _ in range(_READ_RETRIES):
            seqs = [self._slot_seq(i) for i in range(self.slots)]
            best = max(range(self.slots), key=seqs.__getitem__)
            if seqs[best] == 0:
                return None
            result = self._read_slot(best, seqs[best], stamp)
            if result is not None:
                return result
        return None

    def read_since(self, seq: int, stamp: bool = False) -> List[Tuple[SlotHeader, bytes]]:
        """Every record still in the ring with seq_no > seq, oldest first.

        Each record is read under the seqlock; a slot that the writer
        overwrites while we read it is skipped (its newer record shows up on
        the next call). Records older than the ring depth are gone for good:
        compare the first returned seq_no with seq + 1 to count drops.
        """
        seqs = [(self._slot_seq(i), i) for i in range(self.slots)]
        out: List[Tuple[SlotHeader, bytes]] = []
        for s, i in sorted(seqs):
            if s <= seq:
                continue
            result = self._read_slot(i, s, stamp)
            if result is not None:
                out.append(result)
        return out

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
