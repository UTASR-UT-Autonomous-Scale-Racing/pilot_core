"""
Common base for the typed ring-buffer wrappers: subclasses encode in write() and
provide a decode(payload) staticmethod; _MULTI means decode returns a tuple that
is spread into read_latest()'s result.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .ring_buffer import SharedRingBuffer
from .schema import SlotHeader


class TypedRingBuffer:
    _MULTI = False
    _rb: SharedRingBuffer

    @staticmethod
    def decode(payload: bytes) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def ring(self) -> SharedRingBuffer:
        """The underlying SharedRingBuffer (raw bytes access)."""
        return self._rb

    def _pack(self, header: SlotHeader, payload: bytes) -> tuple:
        values = self.decode(payload)
        return (header, *values) if self._MULTI else (header, values)

    def latest_seq(self) -> int:
        """Highest published seq_no, without copying any data."""
        return self._rb.latest_seq()

    def read_latest(self, stamp: bool = False) -> Optional[Tuple]:
        """Newest (header, *decoded values), or None if nothing's been written.

        stamp=True also stores read_ts in shared memory (see SharedRingBuffer).
        """
        result = self._rb.read_latest(stamp=stamp)
        if result is None:
            return None
        return self._pack(*result)

    def read_since(self, seq: int, stamp: bool = False) -> List[Tuple]:
        """Every record newer than seq still in the ring, oldest first, each
        as (header, *decoded values)."""
        return [self._pack(h, p) for h, p in self._rb.read_since(seq, stamp=stamp)]

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
