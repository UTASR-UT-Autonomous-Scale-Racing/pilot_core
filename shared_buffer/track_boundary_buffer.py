"""
Track boundary polylines (outer + inner wall).

Payload layout:
    [4 bytes]       uint32          outer_count (N_outer)
    [N_outer*8]     float32[N][2]   outer boundary points (BEV plane)
    [4 bytes]       uint32          inner_count (N_inner)
    [N_inner*8]     float32[N][2]   inner boundary points (BEV plane)

BEV here means the two ground-plane world axes picked by bev_axes. Max 10k
vertices per side, anything past that is dropped.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_COUNT_FMT = "<I"
_COUNT_SIZE = 4
_MAX_BOUNDARY_PTS = 10_000
_MAX_PAYLOAD = 2 * _COUNT_SIZE + 2 * _MAX_BOUNDARY_PTS * 2 * 4  # ~160 KB


class TrackBoundaryBuffer:

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
            dtype_tag=DTypeTag.UNKNOWN,
            create=create,
        )

    def write(
        self, outer_line: np.ndarray, inner_line: np.ndarray
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
        return self._rb.write(payload)

    def read_latest(
        self,
    ) -> Optional[Tuple[SlotHeader, np.ndarray, np.ndarray]]:
        """(header, outer, inner) with each line (N, 2) float32; None if empty"""
        result = self._rb.read_latest()
        if result is None:
            return None

        header, payload = result
        offset = 0

        (n_outer,) = struct.unpack_from(_COUNT_FMT, payload, offset)
        offset += _COUNT_SIZE
        if n_outer > 0:
            outer = np.frombuffer(
                payload, dtype=np.float32, count=n_outer * 2, offset=offset
            ).reshape((n_outer, 2)).copy()
        else:
            outer = np.empty((0, 2), dtype=np.float32)
        offset += n_outer * 2 * 4

        (n_inner,) = struct.unpack_from(_COUNT_FMT, payload, offset)
        offset += _COUNT_SIZE
        if n_inner > 0:
            inner = np.frombuffer(
                payload, dtype=np.float32, count=n_inner * 2, offset=offset
            ).reshape((n_inner, 2)).copy()
        else:
            inner = np.empty((0, 2), dtype=np.float32)

        return header, outer, inner

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
