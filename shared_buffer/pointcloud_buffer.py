"""
PointCloudBuffer — typed wrapper for 3D map-point data from SLAM.

Payload layout:
    [4 bytes]   uint32          num_points  (N)
    [N*12 bytes] float32[N][3]  point positions (x, y, z)

Supports up to 50 000 points (~600 KB per slot).
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_COUNT_FMT = "<I"
_COUNT_SIZE = 4
_MAX_POINTS = 50_000
_MAX_PAYLOAD = _COUNT_SIZE + _MAX_POINTS * 3 * 4  # ~600 KB


class PointCloudBuffer:
    """High-level buffer for 3D point cloud data.

    Parameters
    ----------
    slots : int
        Number of ring-buffer slots (default 4).
    name : str
        Shared-memory block name.
    create : bool
        Pass *True* from orchestrator, *False* from workers.
    max_points : int
        Maximum number of 3D points per write (default 50 000).
    """

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
            dtype_tag=DTypeTag.UNKNOWN,  # reuse UNKNOWN or add a new tag
            create=create,
        )

    def write(self, points: np.ndarray) -> SlotHeader:
        """Write a point cloud.

        Parameters
        ----------
        points : np.ndarray
            Shape ``(N, 3)``, dtype float32.  Clamped to ``max_points``.
        """
        pts = points.astype(np.float32)
        n = min(len(pts), self._max_points)
        pts = pts[:n]
        count_bytes = struct.pack(_COUNT_FMT, n)
        payload = count_bytes + pts.tobytes()
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        """Read the most recent point cloud.

        Returns
        -------
        (header, points) or *None*.
        ``points`` is ``(N, 3) float32``.
        """
        result = self._rb.read_latest()
        if result is None:
            return None

        header, payload = result
        (n,) = struct.unpack_from(_COUNT_FMT, payload, 0)
        if n > 0:
            pts = np.frombuffer(
                payload, dtype=np.float32, count=n * 3, offset=_COUNT_SIZE
            ).reshape((n, 3)).copy()
        else:
            pts = np.empty((0, 3), dtype=np.float32)

        return header, pts

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
