"""
DepthProjectionWorker — builds a world-frame point cloud of track walls by
combining the segmentation mask (which already encodes wall depth) with the
simulator ground-truth pose.

The VisionWorker writes a float32 (H, W) mask into SegmentationBuffer where:
  - wall pixels  = depth value in metres
  - other pixels = 0.0

This worker reads that mask, projects only the non-zero (wall) pixels into 3D
world-frame coordinates using the camera intrinsics and the current pose
(Twc, world-from-camera), then writes the resulting (N, 3) float32 array to
PointCloudBuffer for use by MappingWorker.

Back-projection formula (pinhole camera model):
  Given pixel (u, v) with depth d (metres):
    X_cam = (u - cx) * d / fx
    Y_cam = (v - cy) * d / fy
    Z_cam = d
    P_world = R @ [X_cam, Y_cam, Z_cam]^T + t
  where R, t come from the Twc pose matrix.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import PoseBuffer, SegmentationBuffer

try:
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None  # type: ignore[misc]


class DepthProjectionWorker(BaseWorker):
    """Projects segmented wall pixels to world-frame 3D points.

    Config kwargs
    -------------
    seg_buffer : SegmentationBuffer
        Wall mask: float32 (H, W), wall pixels = depth in metres, else 0.0.
    pose_buffer : PoseBuffer
        Twc pose (world-from-camera) 4×4 float32.
    output_buffer : PointCloudBuffer
        Destination for projected (N, 3) float32 world-frame points.
    fx, fy, cx, cy : float
        Camera intrinsics in pixels.
    depth_min : float
        Discard wall pixels with depth below this (default 0.1 m).
    depth_max : float
        Discard wall pixels with depth above this (default 14.0 m).
    """

    def __init__(self, name: str = "depth_proj", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._seg_buf: SegmentationBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: Any | None = None
        self._fx = self._fy = self._cx = self._cy = 0.0
        self._depth_min = 0.1
        self._depth_max = 14.0
        self._last_seg_seq = 0
        self._frame_count = 0
        self._voxel_map: dict = {}        # voxel_key → representative point (XYZ)
        self._voxel_size = 0.10           # 10 cm voxels (> 5.9cm depth quantization step)
        self._move_threshold = 0.10       # only project when car moves > 10 cm
        self._last_proj_pos: np.ndarray | None = None   # XY position of last projection

    def setup(self) -> None:
        self._seg_buf = self.config["seg_buffer"]
        self._pose_buf = self.config["pose_buffer"]
        self._pc_buf = self.config["output_buffer"]
        self._fx = float(self.config["fx"])
        self._fy = float(self.config["fy"])
        self._cx = float(self.config["cx"])
        self._cy = float(self.config["cy"])
        self._depth_min = float(self.config.get("depth_min", 0.1))
        self._depth_max = float(self.config.get("depth_max", 14.0))
        self.logger.info(
            f"DepthProjectionWorker ready "
            f"(fx={self._fx} fy={self._fy} cx={self._cx} cy={self._cy} "
            f"depth=[{self._depth_min},{self._depth_max}]m)"
        )

    def step(self) -> None:
        seg_result = self._seg_buf.read_latest()
        if seg_result is None:
            time.sleep(0.01)
            return

        seg_header, mask = seg_result
        if seg_header.seq_no <= self._last_seg_seq:
            time.sleep(0.005)
            return
        self._last_seg_seq = seg_header.seq_no
        self._frame_count += 1

        pose_result = self._pose_buf.read_latest()
        if pose_result is None:
            return

        pose = pose_result[1]  # 4×4 float32 Twc (world-from-camera)

        # Move-gated projection: only project when the car has moved far enough.
        # This avoids the depth-quantization artifact where the same wall surface
        # back-projects to slightly different world positions at each distance,
        # creating a thick shell of duplicate voxels as the car approaches.
        current_pos = pose[:2, 3]  # X, Y horizontal position
        if self._last_proj_pos is not None:
            moved = float(np.linalg.norm(current_pos - self._last_proj_pos))
            if moved < self._move_threshold:
                return
        self._last_proj_pos = current_pos.copy()

        pts_world = self._project_wall_pixels(mask, pose)
        if len(pts_world) == 0:
            return

        # Voxel-grid deduplication: one point kept per 10cm cell.
        keys = (pts_world / self._voxel_size).astype(np.int32)
        new_count = 0
        for k, pt in zip(map(tuple, keys), pts_world):
            if k not in self._voxel_map:
                self._voxel_map[k] = pt
                new_count += 1

        if not self._voxel_map:
            return

        all_pts = np.array(list(self._voxel_map.values()), dtype=np.float32)
        self._pc_buf.write(all_pts)

        if self._frame_count % 30 == 0:
            tx, ty, tz = float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])
            dist = math.sqrt(tx*tx + ty*ty + tz*tz)
            wall_px = int(np.count_nonzero(mask))
            self.logger.info(
                f"pose_t=({tx:.4f},{ty:.4f},{tz:.4f}) dist={dist:.4f}m  "
                f"wall_px={wall_px}  new_voxels={new_count}  "
                f"total_voxels={len(self._voxel_map)}  frame={self._frame_count}"
            )

    def _project_wall_pixels(self, mask: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Back-project wall pixels (mask > 0) to world-frame 3D points.

        Parameters
        ----------
        mask : float32 (H, W)  — wall pixels carry depth value (metres), rest = 0.0
        pose : float32 (4, 4)  — Twc world-from-camera transform

        Returns
        -------
        float32 (N, 3) world-frame XYZ points, one per wall pixel in depth range.
        """
        # Find wall pixels within the valid depth range
        valid = (mask > self._depth_min) & (mask < self._depth_max)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        v_idx, u_idx = np.where(valid)
        d = mask[v_idx, u_idx]

        # Back-project to camera frame using pinhole model
        x_c = (u_idx - self._cx) * d / self._fx
        y_c = (v_idx - self._cy) * d / self._fy
        z_c = d
        pts_cam = np.stack([x_c, y_c, z_c], axis=1).astype(np.float32)  # (N, 3)

        # Rotate + translate to world frame
        R = pose[:3, :3]
        t = pose[:3, 3]
        pts_world = pts_cam @ R.T + t

        return pts_world.astype(np.float32)

    def teardown(self) -> None:
        self.logger.info(
            f"DepthProjectionWorker shut down after {self._frame_count} frames"
        )
