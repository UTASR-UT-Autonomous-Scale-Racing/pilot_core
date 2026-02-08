"""
IMUBuffer — typed wrapper for IMU sensor vectors.

Payload layout:
    [9 × float32]  w, x, y, z, ax, ay, az, throttle_raw, steering_raw
    Total = 36 bytes
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_IMU_FLOATS = 9
_PAYLOAD_SIZE = _IMU_FLOATS * 4  # 36 bytes


class IMUBuffer:
    """High-level buffer for 9-DOF IMU vectors.

    Parameters
    ----------
    slots : int
        Number of ring-buffer slots (default 32).
    name : str
        Shared-memory block name.
    create : bool
        Pass *True* from orchestrator, *False* from workers.
    """

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

    def write(self, imu_vector: np.ndarray) -> SlotHeader:
        """Write a 9-float IMU vector [w, x, y, z, ax, ay, az, throttle, steering].

        Parameters
        ----------
        imu_vector : np.ndarray
            Shape ``(9,)``, dtype ``float32``.
        """
        payload = imu_vector.astype(np.float32).tobytes()
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        """Read the most recent IMU vector.

        Returns
        -------
        (header, vector) or *None*.  ``vector`` is ``(9,) float32``.
        """
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
