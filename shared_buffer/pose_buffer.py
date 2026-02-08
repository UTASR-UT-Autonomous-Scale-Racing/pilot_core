"""
PoseBuffer — typed wrapper for SLAM pose + map-point data.

Payload layout:
    [64 bytes]  float32[4][4]   pose matrix (row-major)
    [4 bytes]   uint32          num_map_points  (N)
    [N*12 bytes] float32[N][3]  map point positions (x, y, z)
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_POSE_BYTES = 4 * 4 * 4  # 4×4 float32 matrix = 64 bytes
_COUNT_FMT = "<I"
_COUNT_SIZE = 4
_MAX_MAP_POINTS = 2048
_MAX_PAYLOAD = _POSE_BYTES + _COUNT_SIZE + _MAX_MAP_POINTS * 3 * 4  # ~25 KB


class PoseBuffer:
    """High-level buffer for SLAM pose + map points.

    Parameters
    ----------
    slots : int
        Number of ring-buffer slots (default 8).
    name : str
        Shared-memory block name.
    create : bool
        Pass *True* from orchestrator, *False* from workers.
    """

    def __init__(
        self,
        slots: int = 8,
        name: str = "pose_buf",
        create: bool = True,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_MAX_PAYLOAD,
            dtype_tag=DTypeTag.POSE,
            create=create,
        )

    def write(
        self, pose: np.ndarray, map_points: Optional[np.ndarray] = None
    ) -> SlotHeader:
        """Write a 4×4 pose matrix and optional map points.

        Parameters
        ----------
        pose : np.ndarray
            Shape ``(4, 4)``, dtype float32.
        map_points : np.ndarray, optional
            Shape ``(N, 3)``, dtype float32.  May be *None* or empty.
        """
        pose_bytes = pose.astype(np.float32).tobytes()

        if map_points is not None and len(map_points) > 0:
            mp = map_points.astype(np.float32)
            n = min(len(mp), _MAX_MAP_POINTS)
            mp = mp[:n]
            count_bytes = struct.pack(_COUNT_FMT, n)
            payload = pose_bytes + count_bytes + mp.tobytes()
        else:
            payload = pose_bytes + struct.pack(_COUNT_FMT, 0)

        return self._rb.write(payload)

    def read_latest(
        self,
    ) -> Optional[Tuple[SlotHeader, np.ndarray, np.ndarray]]:
        """Read the most recent pose + map points.

        Returns
        -------
        (header, pose, map_points) or *None*.
        ``pose`` is ``(4,4) float32``; ``map_points`` is ``(N,3) float32``.
        """
        result = self._rb.read_latest()
        if result is None:
            return None

        header, payload = result

        pose = np.frombuffer(payload, dtype=np.float32, count=16, offset=0).reshape(
            (4, 4)
        ).copy()

        (n,) = struct.unpack_from(_COUNT_FMT, payload, _POSE_BYTES)
        if n > 0:
            mp_offset = _POSE_BYTES + _COUNT_SIZE
            map_points = np.frombuffer(
                payload, dtype=np.float32, count=n * 3, offset=mp_offset
            ).reshape((n, 3)).copy()
        else:
            map_points = np.empty((0, 3), dtype=np.float32)

        return header, pose, map_points

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
