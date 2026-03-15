"""
IMUBuffer — typed wrapper for IMU sensor vectors.

Payload layout:
    [16 × float32]
        w, x, y, z,
        ax, ay, az,
        gx, gy, gz,
        mx, my, mz,
        throttle_raw, steering_raw,
        timestamp

    Total = 64 bytes
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_IMU_FLOATS = 16
_PAYLOAD_SIZE = _IMU_FLOATS * 4  # 64 bytes


class IMUBuffer:
    """High-level buffer for IMU vectors.

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
        """Write an IMU vector.

        Canonical format is 16 float32 values:
        [w, x, y, z, ax, ay, az, gx, gy, gz, mx, my, mz, throttle, steering, timestamp].

        Parameters
        ----------
        imu_vector : np.ndarray
            Shape ``(N,)``, dtype ``float32``.
        """
        payload = imu_vector.astype(np.float32).tobytes()
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        """Read the most recent IMU vector.

        Returns
        -------
        (header, vector) or *None*.  ``vector`` is ``(N,) float32``.
        """
        result = self._rb.read_latest()
        if result is None:
            return None
        header, payload = result
        n_floats = len(payload) // 4
        vector = np.frombuffer(payload, dtype=np.float32, count=n_floats).copy()
        return header, vector

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
