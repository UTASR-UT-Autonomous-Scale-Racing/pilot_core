"""
2D lidar scan: 1080 float32 ranges in metres, angle_min first, angle_max last
(4320 bytes).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_N_RAYS = 1080
_PAYLOAD_SIZE = _N_RAYS * 4  # 4320 bytes


class LidarBuffer:

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

    def write(self, ranges: np.ndarray) -> SlotHeader:
        """ranges: (1080,), metres"""
        payload = ranges.astype(np.float32).tobytes()
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        result = self._rb.read_latest()
        if result is None:
            return None
        header, payload = result
        ranges = np.frombuffer(payload, dtype=np.float32, count=_N_RAYS).copy()
        return header, ranges

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
