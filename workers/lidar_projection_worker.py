"""
Simulator only: 2D LIDAR scans + ground-truth pose (4x4 Twv, world-from-vehicle,
Z-up world, x-forward/y-left vehicle) -> voxel-deduplicated world-frame (N, 3)
float32 cloud, rewritten in full to the PointCloudBuffer.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import LidarBuffer, PoseBuffer

try:
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None  # type: ignore[misc]

# Hokuyo UST-10LX, as simulated by AutoDRIVE
_N_RAYS = 1080
_ANGLE_MIN = -2.35619449  # rad, -135°
_ANGLE_INCREMENT = 0.00436332  # rad/ray (0.25 deg)

# mount offset, vehicle frame, metres
_LIDAR_OFFSET = np.array([0.2733, 0.0, 0.096], dtype=np.float32)


class LidarProjectionWorker(BaseWorker):
    """2D LIDAR scans + GT pose -> world-frame 3D point cloud.

    Config kwargs:
      lidar_buffer          LidarBuffer, 1080-ray range scans
      pose_buffer           PoseBuffer, GT pose (Twv)
      output_buffer         PointCloudBuffer
      range_min, range_max  rays outside are dropped, default 0.1 / 9.5 m
      voxel_size            dedup cell size, default 0.10 m
      move_threshold        min ground-plane movement between projections, default 0.05 m
    """

    def __init__(self, name: str = "lidar_proj", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._lidar_buf: LidarBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: Any | None = None
        self._range_min: float = 0.1
        self._range_max: float = 9.5
        self._voxel_size: float = 0.10
        self._move_threshold: float = 0.05
        # accumulated cloud: one point per occupied voxel, plus its packed key
        self._voxel_pts = np.empty((0, 3), dtype=np.float32)
        self._voxel_keys = np.empty(0, dtype=np.int64)
        self._pc_capacity: int = 50_000
        self._capacity_warned = False
        self._last_lidar_seq: int = 0
        self._last_proj_pos: np.ndarray | None = None
        self._frame_count: int = 0

        # ray angles are fixed, precompute
        angles = _ANGLE_MIN + np.arange(_N_RAYS) * _ANGLE_INCREMENT
        self._ray_cos = np.cos(angles).astype(np.float32)
        self._ray_sin = np.sin(angles).astype(np.float32)

    def setup(self) -> None:
        self._lidar_buf = self.config["lidar_buffer"]
        self._pose_buf = self.config["pose_buffer"]
        self._pc_buf = self.config["output_buffer"]
        self._range_min = float(self.config.get("range_min", 0.1))
        self._range_max = float(self.config.get("range_max", 9.5))
        self._voxel_size = float(self.config.get("voxel_size", 0.10))
        self._move_threshold = float(self.config.get("move_threshold", 0.05))
        self._pc_capacity = self._pc_buf.max_points
        self.logger.info(
            f"LidarProjectionWorker ready "
            f"(range=[{self._range_min},{self._range_max}]m "
            f"voxel={self._voxel_size}m move_threshold={self._move_threshold}m)"
        )

    def step(self) -> None:
        if self._lidar_buf.latest_seq() <= self._last_lidar_seq:
            time.sleep(0.005)
            return
        lidar_result = self._lidar_buf.read_latest()
        if lidar_result is None:
            time.sleep(0.01)
            return

        lidar_header, ranges = lidar_result
        if lidar_header.seq_no <= self._last_lidar_seq:
            time.sleep(0.005)
            return
        self._last_lidar_seq = lidar_header.seq_no

        pose_result = self._pose_buf.read_latest()
        if pose_result is None:
            return
        pose = pose_result[1]  # 4x4 float32 Twv (world-from-vehicle)

        current_pos = pose[:2, 3]
        if self._last_proj_pos is not None:
            if float(np.linalg.norm(current_pos - self._last_proj_pos)) < self._move_threshold:
                return
        self._last_proj_pos = current_pos.copy()

        pts_world = self._project_scan(ranges, pose)
        if len(pts_world) == 0:
            return

        # voxel dedup; first point in a cell wins
        new_count = self._add_to_voxel_map(pts_world)
        if len(self._voxel_pts) == 0:
            return

        self._pc_buf.write(self._voxel_pts, ts=lidar_header.produced_ts)
        self._frame_count += 1

        if not self._capacity_warned and len(self._voxel_pts) >= 0.9 * self._pc_capacity:
            self._capacity_warned = True
            self.logger.warning(
                f"Point cloud has {len(self._voxel_pts)} voxels, near the "
                f"PointCloudBuffer capacity of {self._pc_capacity}; points past it "
                f"are dropped. Increase voxel_size or the buffer's max_points."
            )

        if self._frame_count % 50 == 0:
            tx, ty = float(pose[0, 3]), float(pose[1, 3])
            self.logger.info(
                f"pos=({tx:.2f},{ty:.2f})  valid_rays={len(pts_world)}  "
                f"new_voxels={new_count}  total_voxels={len(self._voxel_pts)}  "
                f"frame={self._frame_count}"
            )

    def _add_to_voxel_map(self, pts: np.ndarray) -> int:
        """Add the points whose voxel is still empty; returns how many were new.

        Voxel indices (np.floor(p / voxel_size)) are packed into one int64 with
        21 bits per axis, which covers +/-1e6 voxels (+/-100 km at 0.1 m).
        """
        idx = np.floor(pts / self._voxel_size).astype(np.int64) + (1 << 20)
        keys = (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]
        keys, first = np.unique(keys, return_index=True)   # dedupe within the scan
        new = ~np.isin(keys, self._voxel_keys, assume_unique=True)
        if not np.any(new):
            return 0
        self._voxel_keys = np.concatenate([self._voxel_keys, keys[new]])
        self._voxel_pts = np.concatenate([self._voxel_pts, pts[first[new]].astype(np.float32)])
        return int(new.sum())

    def _project_scan(self, ranges: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """(1080,) ranges in metres + Twv pose -> (N, 3) world XYZ, valid rays only."""
        valid = (ranges >= self._range_min) & (ranges <= self._range_max)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        r = ranges[valid]
        cos_a = self._ray_cos[valid]
        sin_a = self._ray_sin[valid]

        x_lidar = r * cos_a
        y_lidar = r * sin_a
        z_lidar = np.zeros(len(r), dtype=np.float32)

        # mount offset only, the LIDAR isn't rotated relative to the car
        x_veh = x_lidar + _LIDAR_OFFSET[0]
        y_veh = y_lidar + _LIDAR_OFFSET[1]
        z_veh = z_lidar + _LIDAR_OFFSET[2]

        pts_veh = np.stack([x_veh, y_veh, z_veh], axis=1)  # (N, 3)

        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = pts_veh @ R.T + t

        return pts_world.astype(np.float32)

    def teardown(self) -> None:
        self.logger.info(
            f"LidarProjectionWorker shut down after {self._frame_count} frames "
            f"({len(self._voxel_pts)} total voxels)"
        )
