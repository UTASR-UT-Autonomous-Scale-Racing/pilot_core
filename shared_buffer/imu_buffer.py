"""
IMU vectors, 9 x float32 (36 bytes):

    qw, qx, qy, qz, ax, ay, az, throttle_raw, steering_raw

Orientation quaternion, accel in m/s^2, then the raw throttle/steering
feedback from the Arduino. NB: the OAK-D IMU path has no such feedback, so it
puts gyro x / y (rad/s) in the last two slots instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_IMU_FLOATS = 9
_PAYLOAD_SIZE = _IMU_FLOATS * 4  # 36 bytes


class IMUBuffer:

    def __init__(
        self,
        slots: int = 32,
        name: str = "imu_buf",
        create: bool = True,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_PAYLOAD_SIZE,
            dtype_tag=DTypeTag.IMU,
            create=create,
        )

    def write(self, imu_vector: np.ndarray, ts: Optional[float] = None) -> SlotHeader:
        """imu_vector is (9,), laid out as in the module docstring."""
        payload = imu_vector.astype(np.float32).tobytes()
        return self._rb.write(payload, ts)

    def latest_seq(self) -> int:
        """Highest published seq_no, without copying any data."""
        return self._rb.latest_seq()

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        """(header, (9,) float32 vector) or None"""
        result = self._rb.read_latest()
        if result is None:
            return None
        header, payload = result
        vector = np.frombuffer(payload, dtype=np.float32, count=_IMU_FLOATS).copy()
        return header, vector

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
