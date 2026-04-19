"""
LidarProjectionWorker — builds a world-frame point cloud from 2D LIDAR scans
combined with the simulator's ground-truth pose (PoseBuffer).  Simulator
only: the world frame is Z-up (ROS / AutoDRIVE), vehicle frame x-forward,
y-left.

The AutoDRIVE simulator publishes a Hokuyo UST-10LX-equivalent scan:
  - 1080 rays, 270° sweep (angle_min=-2.35619449 rad, increment=0.00436332 rad)
  - range_min=0.06 m, range_max=10.0 m
  - LIDAR mount in vehicle frame: [x=0.2733, y=0.0, z=0.096], no rotation

Pipeline
--------
1. Read latest LidarBuffer scan (1080 float32 ranges).
2. Read latest PoseBuffer pose (4×4 Twv, world-from-vehicle).
3. Convert each valid ray to a 2D point in vehicle frame using polar→Cartesian.
4. Apply LIDAR-to-vehicle offset, then transform to world frame via pose.
5. Voxel-grid deduplication (``voxel_size`` cells) into an accumulated map.
6. Write the whole accumulated (N, 3) float32 cloud to PointCloudBuffer.

Scans are only projected after the vehicle has moved ``move_threshold``
metres in the ground plane since the last projection.
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

# Hokuyo UST-10LX specs (AutoDRIVE simulator)
_N_RAYS = 1080
_ANGLE_MIN = -2.35619449  # rad  (-135°)
_ANGLE_INCREMENT = 0.00436332  # rad per ray  (0.25°)

# LIDAR mount offset in vehicle frame (metres)
_LIDAR_OFFSET = np.array([0.2733, 0.0, 0.096], dtype=np.float32)


class LidarProjectionWorker(BaseWorker):
    """Converts 2D LIDAR scans + GT pose into a world-frame 3D point cloud.

    Config kwargs
    -------------
    lidar_buffer : LidarBuffer
        Source of 1080-ray range scans.
    pose_buffer : PoseBuffer
        Ground-truth vehicle pose (4×4 Twv, world-from-vehicle).
    output_buffer : PointCloudBuffer
        Destination for accumulated (N, 3) float32 world-frame points.
    range_min : float
        Drop rays shorter than this (default: 0.1 m).
    range_max : float
        Drop rays longer than this (default: 9.5 m).
    voxel_size : float
        Deduplication cell size in metres (default: 0.10).
    move_threshold : float
        Minimum vehicle movement (metres) between projections (default: 0.05).
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
        self._voxel_map: dict = {}
        self._last_lidar_seq: int = 0
        self._last_proj_pos: np.ndarray | None = None
        self._frame_count: int = 0

        # Precompute ray angles (constant for all scans)
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
        self.logger.info(
            f"LidarProjectionWorker ready "
            f"(range=[{self._range_min},{self._range_max}]m "
            f"voxel={self._voxel_size}m move_threshold={self._move_threshold}m)"
        )

    def step(self) -> None:
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
        pose = pose_result[1]  # 4×4 float32 Twv (world-from-vehicle)

        # Move gate: only project after moving move_threshold in the ground (x, y) plane
        current_pos = pose[:2, 3]
        if self._last_proj_pos is not None:
            if float(np.linalg.norm(current_pos - self._last_proj_pos)) < self._move_threshold:
                return
        self._last_proj_pos = current_pos.copy()

        pts_world = self._project_scan(ranges, pose)
        if len(pts_world) == 0:
            return

        # Voxel-grid deduplication
        keys = np.floor(pts_world / self._voxel_size).astype(np.int64)
        new_count = 0
        for k, pt in zip(map(tuple, keys), pts_world):
            if k not in self._voxel_map:
                self._voxel_map[k] = pt
                new_count += 1

        if not self._voxel_map:
            return

        all_pts = np.array(list(self._voxel_map.values()), dtype=np.float32)
        self._pc_buf.write(all_pts)
        self._frame_count += 1

        if self._frame_count % 50 == 0:
            tx, ty = float(pose[0, 3]), float(pose[1, 3])
            self.logger.info(
                f"pos=({tx:.2f},{ty:.2f})  valid_rays={len(pts_world)}  "
                f"new_voxels={new_count}  total_voxels={len(self._voxel_map)}  "
                f"frame={self._frame_count}"
            )

    def _project_scan(self, ranges: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Convert a 1080-ray scan to world-frame 3D points.

        Parameters
        ----------
        ranges : float32 (1080,) — per-ray distances in metres
        pose   : float32 (4, 4) — Twv world-from-vehicle transform

        Returns
        -------
        float32 (N, 3) world-frame XYZ points for valid rays only.
        """
        valid = (ranges >= self._range_min) & (ranges <= self._range_max)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        r = ranges[valid]
        cos_a = self._ray_cos[valid]
        sin_a = self._ray_sin[valid]

        # 2D polar → Cartesian in LIDAR frame (x=forward, y=left, z=up)
        x_lidar = r * cos_a
        y_lidar = r * sin_a
        z_lidar = np.zeros(len(r), dtype=np.float32)

        # Apply LIDAR mount offset → vehicle frame
        x_veh = x_lidar + _LIDAR_OFFSET[0]
        y_veh = y_lidar + _LIDAR_OFFSET[1]
        z_veh = z_lidar + _LIDAR_OFFSET[2]

        pts_veh = np.stack([x_veh, y_veh, z_veh], axis=1)  # (N, 3)

        # Transform to world frame via pose
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = pts_veh @ R.T + t

        return pts_world.astype(np.float32)

    def teardown(self) -> None:
        self.logger.info(
            f"LidarProjectionWorker shut down after {self._frame_count} frames "
            f"({len(self._voxel_map)} total voxels)"
        )
