"""
MappingWorker — accumulates BEV wall points and vehicle trajectory from
LIDAR projection + ZMQ pose, detects loop closure, and extracts track boundaries.

Phase A (mapping):
    - Read PoseBuffer for vehicle poses (position + heading)
    - Read PointCloudBuffer for 3D wall points from LidarProjectionWorker
    - Accumulate until loop closure event (manual 'L' + ENTER)

Phase B (boundary extraction):
    - For every accumulated wall point, find the nearest point on the
      driven trajectory and check which side (left / right) the wall
      point lies on using the cross product with the forward direction.
    - Left-side points  → one boundary line (published as 'inner_line')
    - Right-side points → other boundary line (published as 'outer_line')
    - This approach works for any track topology — no donut assumption.
    - Points are ordered by arc-length along the trajectory for clean
      LINE_STRIP visualization in RViz.
    - Writes boundaries to TrackBoundaryBuffer.
    - Signals mapping_done_event for the orchestrator.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from .bev import pose_to_bev
from shared_buffer import PoseBuffer, SegmentationBuffer
from shared_buffer.pointcloud_buffer import PointCloudBuffer
from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer


def _to_bev(points_3d: np.ndarray, axes: tuple = (0, 2)) -> np.ndarray:
    """Project 3D world points to 2D BEV by selecting two horizontal axes."""
    return points_3d[:, list(axes)]


def split_walls_by_trajectory(
    bev_wall_pts: np.ndarray,
    trajectory_xyh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split wall points into left / right relative to the driven trajectory.

    For every wall point the nearest trajectory pose is found via a KD-tree.
    The cross product of (forward direction at that pose) and (vector to wall
    point) determines the side: positive → left, negative → right.
    Points are sorted by their trajectory arc-length so they form a smooth
    polyline suitable for LINE_STRIP visualization.

    Parameters
    ----------
    bev_wall_pts : np.ndarray
        ``(N, 2)`` BEV wall points.
    trajectory_xyh : np.ndarray
        ``(T, 3)`` columns = [x, y, heading_rad] of the driven path.

    Returns
    -------
    (left_pts, right_pts) — each ``(M, 2) float32``, ordered by arc-length.
    """
    from scipy.spatial import KDTree

    traj_pos = trajectory_xyh[:, :2]   # (T, 2)
    traj_hdg = trajectory_xyh[:, 2]    # (T,)

    # Arc-lengths along trajectory (used for ordering boundary points)
    arc = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(traj_pos, axis=0), axis=1)
    )])  # (T,)

    # Nearest trajectory point for every wall point
    tree = KDTree(traj_pos)
    _, nearest_idx = tree.query(bev_wall_pts)  # (N,)

    # Forward vectors at nearest trajectory points
    hdgs = traj_hdg[nearest_idx]           # (N,)
    fwd_x = np.cos(hdgs)                   # (N,)
    fwd_y = np.sin(hdgs)                   # (N,)

    # Vector from trajectory point → wall point
    v = bev_wall_pts - traj_pos[nearest_idx]  # (N, 2)

    # Cross product z-component: fwd × v
    #   positive → wall is to the LEFT  (ROS: y > 0)
    #   negative → wall is to the RIGHT (ROS: y < 0)
    cross = fwd_x * v[:, 1] - fwd_y * v[:, 0]  # (N,)
    left_mask = cross > 0

    left_pts = bev_wall_pts[left_mask]
    right_pts = bev_wall_pts[~left_mask]

    # Order each side by arc-length of their nearest trajectory point
    def _order_by_arc(pts: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        order = np.argsort(arc[idx])
        return pts[order].astype(np.float32)

    left_pts = _order_by_arc(left_pts, nearest_idx[left_mask])
    right_pts = _order_by_arc(right_pts, nearest_idx[~left_mask])

    return left_pts, right_pts


class MappingWorker(BaseWorker):
    """Worker that accumulates BEV data and extracts track boundaries.

    Config kwargs
    -------------
    pose_buffer : PoseBuffer
        Vehicle poses (position + orientation).
    segment_buffer : SegmentationBuffer
        Wall masks from vision (read but not used in LIDAR mode).
    pointcloud_buffer : PointCloudBuffer
        3D wall points from LidarProjectionWorker.
    boundary_buffer : TrackBoundaryBuffer
        Output: extracted left / right track boundaries.
    loop_event : mp.Event
        Set when the operator signals loop closure (manual 'L' + ENTER).
    mapping_done_event : mp.Event
        Set by this worker when boundaries are extracted and ready.
    boundary_min_points : int
        Minimum wall points before attempting extraction (default 50).
    bev_axes : list[int]
        Two axis indices for the horizontal BEV plane.
        Simulator (Z-up): [0, 1].  SLAM camera (Y-down): [0, 2].
    forward_axis : int
        Body axis pointing along the direction of travel (see
        :func:`workers.bev.pose_to_bev`).  Simulator: 0.  SLAM camera: 2.
    """

    def __init__(self, name: str = "mapping", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._pose_buf: PoseBuffer | None = None
        self._seg_buf: SegmentationBuffer | None = None
        self._pc_buf: PointCloudBuffer | None = None
        self._boundary_buf: TrackBoundaryBuffer | None = None
        self._loop_event: Optional[mp.Event] = None
        self._mapping_done_event: Optional[mp.Event] = None

        self._min_points: int = 50
        self._bev_axes: tuple = (0, 2)
        self._save_path: Optional[str] = None

        # Accumulated data
        # _bev_trajectory stores (x, y, heading_rad) per pose
        self._bev_wall_pts: list[np.ndarray] = []
        self._bev_trajectory: list[np.ndarray] = []
        self._last_pose_seq = 0
        self._last_pc_seq = 0
        self._last_seg_seq = 0
        self._boundaries_extracted = False

    def setup(self) -> None:
        self._pose_buf = self.config["pose_buffer"]
        self._seg_buf = self.config["segment_buffer"]
        self._pc_buf = self.config["pointcloud_buffer"]
        self._boundary_buf = self.config["boundary_buffer"]
        self._loop_event = self.config.get("loop_event")
        self._mapping_done_event = self.config.get("mapping_done_event")
        self._min_points = self.config.get("boundary_min_points", 50)
        self._bev_axes = tuple(self.config.get("bev_axes", [0, 2]))
        self._forward_axis = int(self.config.get("forward_axis", 2))
        self._save_path = self.config.get("boundary_save_path")

        self.logger.info(
            f"MappingWorker ready (min_pts={self._min_points}, "
            f"bev_axes={self._bev_axes}, forward_axis={self._forward_axis})"
        )

    def step(self) -> None:
        if self._boundaries_extracted:
            time.sleep(0.5)
            return

        # --- Read latest pose → accumulate trajectory (x, y, heading) ---
        pose_result = self._pose_buf.read_latest()
        if pose_result is not None:
            pose_header, pose, _, _ = pose_result
            if pose_header.seq_no > self._last_pose_seq:
                (x, y), heading = pose_to_bev(pose, self._bev_axes, self._forward_axis)
                self._bev_trajectory.append(
                    np.array([x, y, heading], dtype=np.float32)
                )
                self._last_pose_seq = pose_header.seq_no

        # --- Read latest point cloud (full deduplicated wall cloud) ---
        pc_result = self._pc_buf.read_latest()
        if pc_result is not None:
            pc_header, pts_3d = pc_result
            if pc_header.seq_no > self._last_pc_seq and len(pts_3d) > 0:
                self._bev_wall_pts = [_to_bev(pts_3d, self._bev_axes)]
                self._last_pc_seq = pc_header.seq_no
                self.logger.debug(
                    f"Wall map updated: {len(pts_3d)} voxels → "
                    f"{len(self._bev_wall_pts[0])} BEV pts"
                )

        # --- Check loop closure → extract boundaries ---
        if (
            self._loop_event is not None
            and self._loop_event.is_set()
            and not self._boundaries_extracted
        ):
            self._extract_boundaries()

        time.sleep(0.01)

    def _extract_boundaries(self) -> None:
        """Split accumulated wall points into left/right using trajectory."""
        if not self._bev_wall_pts:
            self.logger.warning("No wall points accumulated — will retry in 2s")
            time.sleep(2.0)
            return

        all_wall = np.vstack(self._bev_wall_pts)

        if len(all_wall) < self._min_points:
            self.logger.warning(
                f"Only {len(all_wall)} wall points (need {self._min_points}) "
                f"— will retry in 2s"
            )
            time.sleep(2.0)
            return

        if len(self._bev_trajectory) < 2:
            self.logger.warning("Not enough trajectory points — will retry in 2s")
            time.sleep(2.0)
            return

        trajectory = np.array(self._bev_trajectory, dtype=np.float32)  # (T, 3)

        self.logger.info(
            f"Extracting boundaries: {len(all_wall)} wall pts, "
            f"{len(trajectory)} trajectory poses …"
        )

        try:
            left_line, right_line = split_walls_by_trajectory(all_wall, trajectory)
        except Exception:
            self.logger.exception("Boundary split failed")
            return

        if len(left_line) == 0 or len(right_line) == 0:
            self.logger.warning(
                f"One side empty (left={len(left_line)}, right={len(right_line)}) "
                f"— will retry in 2s"
            )
            time.sleep(2.0)
            return

        # Write: left → inner_line, right → outer_line
        # (swap if vehicle drives clockwise and convention is reversed)
        self._boundary_buf.write(right_line, left_line)
        self._boundaries_extracted = True

        if self._save_path is not None:
            import os
            os.makedirs(self._save_path, exist_ok=True)
            np.save(os.path.join(self._save_path, "outer_line.npy"), right_line)
            np.save(os.path.join(self._save_path, "inner_line.npy"), left_line)
            self.logger.info(f"Boundaries saved to {self._save_path}")

        self.logger.info("=" * 60)
        self.logger.info(f"Track mapped. {len(all_wall)} wall points.")
        self.logger.info(f"Left boundary:  {len(left_line)} pts")
        self.logger.info(f"Right boundary: {len(right_line)} pts")
        self.logger.info("=" * 60)

        if self._mapping_done_event is not None:
            self._mapping_done_event.set()

    def teardown(self) -> None:
        total_wall = sum(len(b) for b in self._bev_wall_pts)
        self.logger.info(
            f"MappingWorker shutting down — "
            f"{total_wall} wall pts, {len(self._bev_trajectory)} traj pts, "
            f"boundaries_extracted={self._boundaries_extracted}"
        )
