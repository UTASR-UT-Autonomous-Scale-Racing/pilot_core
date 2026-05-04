"""
IMU vectors, 9 x float32 (36 bytes):

    qw, qx, qy, qz, ax, ay, az, throttle_raw, steering_raw

Orientation quaternion, accel in m/s^2, then the raw throttle/steering
feedback from the Arduino (pulse widths in us). NB: the OAK-D IMU path has no
such feedback, so it puts gyro x / y (rad/s) in the last two slots instead.
read_latest() -> (header, (9,) float32); decode(payload) -> (9,) float32.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_IMU_FLOATS = 9
_PAYLOAD_SIZE = _IMU_FLOATS * 4  # 36 bytes


class IMUBuffer(TypedRingBuffer):

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
        vec = np.asarray(imu_vector, dtype=np.float32)
        if vec.shape != (_IMU_FLOATS,):
            raise ValueError(f"IMU vector must have shape ({_IMU_FLOATS},), got {vec.shape}")
        return self._rb.write(vec.tobytes(), ts)

    @staticmethod
    def decode(payload: bytes) -> np.ndarray:
        """payload -> (9,) float32 vector"""
        return np.frombuffer(payload, dtype=np.float32, count=_IMU_FLOATS).copy()
