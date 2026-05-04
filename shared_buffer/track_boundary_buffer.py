"""
Track boundary polylines (outer + inner wall).

Payload layout:
    [4 bytes]       uint32          outer_count (N_outer)
    [N_outer*8]     float32[N][2]   outer boundary points (BEV plane)
    [4 bytes]       uint32          inner_count (N_inner)
    [N_inner*8]     float32[N][2]   inner boundary points (BEV plane)

BEV here means the two ground-plane world axes picked by bev_axes. Max 10k
vertices per side, anything past that is dropped.
read_latest() -> (header, outer, inner); decode(payload) -> (outer, inner).
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_COUNT_FMT = "<I"
_COUNT_SIZE = 4
_MAX_BOUNDARY_PTS = 10_000
_MAX_PAYLOAD = 2 * _COUNT_SIZE + 2 * _MAX_BOUNDARY_PTS * 2 * 4  # ~160 KB


def _read_line(payload: bytes, offset: int) -> Tuple[np.ndarray, int]:
    (n,) = struct.unpack_from(_COUNT_FMT, payload, offset)
    offset += _COUNT_SIZE
    if n == 0:
        return np.empty((0, 2), dtype=np.float32), offset
    line = np.frombuffer(payload, dtype=np.float32, count=n * 2, offset=offset).reshape((n, 2)).copy()
    return line, offset + n * 2 * 4


class TrackBoundaryBuffer(TypedRingBuffer):

    _MULTI = True

    def __init__(
        self,
        slots: int = 2,
        name: str = "boundary_buf",
        create: bool = True,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_MAX_PAYLOAD,
            dtype_tag=DTypeTag.BOUNDARY,
            create=create,
        )

    def write(
        self, outer_line: np.ndarray, inner_line: np.ndarray, ts: Optional[float] = None
    ) -> SlotHeader:
        """outer_line (N, 2) and inner_line (M, 2), both in BEV coords."""
        outer = outer_line.astype(np.float32)
        inner = inner_line.astype(np.float32)

        n_outer = min(len(outer), _MAX_BOUNDARY_PTS)
        n_inner = min(len(inner), _MAX_BOUNDARY_PTS)

        payload = (
            struct.pack(_COUNT_FMT, n_outer)
            + outer[:n_outer].tobytes()
            + struct.pack(_COUNT_FMT, n_inner)
            + inner[:n_inner].tobytes()
        )
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """payload -> (outer, inner), each (N, 2) float32"""
        outer, offset = _read_line(payload, 0)
        inner, _ = _read_line(payload, offset)
        return outer, inner
