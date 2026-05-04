"""
3D map points from SLAM.

    [4 bytes]    uint32         num_points (N)
    [N*12 bytes] float32[N][3]  x, y, z

Default cap is 50k points, which works out to ~600 KB per slot.
read_latest() -> (header, (N, 3) float32); decode(payload) -> (N, 3) float32.
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_COUNT_FMT = "<I"
_COUNT_SIZE = 4
_MAX_POINTS = 50_000
_MAX_PAYLOAD = _COUNT_SIZE + _MAX_POINTS * 3 * 4  # ~600 KB


class PointCloudBuffer(TypedRingBuffer):
    """Point cloud buffer. Writes beyond max_points get clipped."""

    def __init__(
        self,
        slots: int = 4,
        name: str = "pc_buf",
        create: bool = True,
        max_points: int = _MAX_POINTS,
    ) -> None:
        self._max_points = max_points
        payload_size = _COUNT_SIZE + max_points * 3 * 4
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=payload_size,
            dtype_tag=DTypeTag.POINTCLOUD,
            create=create,
        )

    def write(self, points: np.ndarray, ts: Optional[float] = None) -> SlotHeader:
        pts = points.astype(np.float32)
        n = min(len(pts), self._max_points)
        pts = pts[:n]
        count_bytes = struct.pack(_COUNT_FMT, n)
        payload = count_bytes + pts.tobytes()
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> np.ndarray:
        """payload -> (N, 3) float32 points"""
        (n,) = struct.unpack_from(_COUNT_FMT, payload, 0)
        if n == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.frombuffer(payload, dtype=np.float32, count=n * 3, offset=_COUNT_SIZE).reshape((n, 3)).copy()
