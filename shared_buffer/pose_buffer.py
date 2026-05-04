"""
PoseBuffer: SLAM pose + map points (+ speed).

Payload layout:
    [64 bytes]  float32[4][4]   pose matrix (row-major, camera-to-world)
    [4 bytes]   uint32          num_map_points  (N)
    [N*12 bytes] float32[N][3]  map point positions (x, y, z)
    [4 bytes]   float32         speed (m/s)

read_latest() -> (header, pose, map_points, speed);
decode(payload) -> (pose, map_points, speed).
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_POSE_BYTES = 4 * 4 * 4  # 4x4 float32
_COUNT_FMT = "<I"
_COUNT_SIZE = 4
_MAX_MAP_POINTS = 2048
_SPEED_SIZE = 4
_MAX_PAYLOAD = _POSE_BYTES + _COUNT_SIZE + _MAX_MAP_POINTS * 3 * 4 + _SPEED_SIZE  # ~25 KB


class PoseBuffer(TypedRingBuffer):

    _MULTI = True

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
        self, pose: np.ndarray, map_points: Optional[np.ndarray] = None, speed: float = 0.0,
        ts: Optional[float] = None,
    ) -> SlotHeader:
        """pose is the 4x4 camera-to-world matrix. map_points (N, 3) can be
        None/empty and gets cut to 2048. speed is m/s, 0.0 if unknown.
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

        payload += struct.pack("<f", float(speed))
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> Tuple[np.ndarray, np.ndarray, float]:
        """payload -> (pose 4x4, map_points Nx3, speed m/s)"""
        pose = np.frombuffer(payload, dtype=np.float32, count=16, offset=0).reshape((4, 4)).copy()

        (n,) = struct.unpack_from(_COUNT_FMT, payload, _POSE_BYTES)
        if n > 0:
            mp_offset = _POSE_BYTES + _COUNT_SIZE
            map_points = np.frombuffer(
                payload, dtype=np.float32, count=n * 3, offset=mp_offset
            ).reshape((n, 3)).copy()
        else:
            map_points = np.empty((0, 3), dtype=np.float32)

        speed_offset = _POSE_BYTES + _COUNT_SIZE + n * 3 * 4
        if len(payload) >= speed_offset + _SPEED_SIZE:
            (speed,) = struct.unpack_from("<f", payload, speed_offset)
        else:
            speed = 0.0
        return pose, map_points, speed
