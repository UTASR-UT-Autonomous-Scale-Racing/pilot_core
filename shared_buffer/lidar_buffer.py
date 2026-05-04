"""
2D lidar scan: 1080 float32 ranges in metres, angle_min first, angle_max last
(4320 bytes). read_latest() -> (header, ranges); decode(payload) -> (1080,) float32.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_N_RAYS = 1080
_PAYLOAD_SIZE = _N_RAYS * 4  # 4320 bytes


class LidarBuffer(TypedRingBuffer):

    def __init__(
        self,
        slots: int = 8,
        name: str = "lidar_buf",
        create: bool = True,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_PAYLOAD_SIZE,
            dtype_tag=DTypeTag.LIDAR,
            create=create,
        )

    def write(self, ranges: np.ndarray, ts: Optional[float] = None) -> SlotHeader:
        """ranges: (1080,), metres"""
        payload = ranges.astype(np.float32).tobytes()
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> np.ndarray:
        """payload -> (1080,) float32 ranges"""
        return np.frombuffer(payload, dtype=np.float32, count=_N_RAYS).copy()
