"""
MappingWorker — accumulates BEV wall points and vehicle trajectory from
SLAM + vision outputs, detects loop closure, and extracts track boundaries.

Phase A (mapping):
    - Read PoseBuffer for vehicle poses
    - Read SegmentationBuffer for wall masks
    - Read PointCloudBuffer for 3D map points
    - Project wall-labelled 3D points to BEV (X, Z)
    - Accumulate until loop closure event

Phase B (boundary extraction):
    - Compute alpha-shape or distance-based boundary split
    - Write outer/inner lines to TrackBoundaryBuffer
    - Signal mapping_done_event for the orchestrator
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import PoseBuffer, SegmentationBuffer
from shared_buffer.pointcloud_buffer import PointCloudBuffer
from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer


def _to_bev(points_3d: np.ndarray) -> np.ndarray:
    """Project 3D world points to 2D bird's-eye view by dropping Y (height).

    Parameters
    ----------
    points_3d : np.ndarray
        Shape ``(N, 3)`` with columns (X, Y, Z).

    Returns
    -------
    np.ndarray — shape ``(N, 2)`` with columns (X, Z).
    """
    return points_3d[:, [0, 2]]


def _order_boundary(pts: np.ndarray) -> np.ndarray:
    """Order 2D points into a continuous polyline via angular sort."""
    if len(pts) < 3:
        return pts
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(angles)
    return pts[order]


def extract_track_boundaries(
    bev_wall_pts: np.ndarray,
    trajectory_bev: np.ndarray,
    alpha: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract outer and inner track boundary lines from BEV wall points.

    Parameters
    ----------
    bev_wall_pts : np.ndarray
        ``(M, 2)`` all wall points in BEV.
    trajectory_bev : np.ndarray
        ``(T, 2)`` vehicle trajectory in BEV.
    alpha : float
        Alpha-shape parameter (higher = tighter fit).

    Returns
    -------
    (outer_line, inner_line) — each ``(N, 2)`` ordered boundary coords.
    """
    centroid = trajectory_bev.mean(axis=0)

    # Try alpha-shape approach if libraries are available
    try:
        import alphashape
        from shapely.geometry import Point

        hull = alphashape.alphashape(bev_wall_pts, alpha)

        if hull.geom_type == "MultiPolygon":
            for poly in hull.geoms:
                if poly.contains(Point(centroid)):
                    hull = poly
                    break
            else:
                hull = max(hull.geoms, key=lambda p: p.area)

        outer_line = np.array(hull.exterior.coords, dtype=np.float32)

        if hull.interiors:
            inner_line = np.array(
                hull.interiors[0].coords, dtype=np.float32
            )
        else:
            # No hole detected → fall through to distance-based split
            raise ValueError("No interior ring")

        return outer_line, inner_line

    except Exception:
        # Fallback: split wall points by median distance from centroid
        dists = np.linalg.norm(bev_wall_pts - centroid, axis=1)
        median_d = np.median(dists)
        inner_pts = bev_wall_pts[dists < median_d]
        outer_pts = bev_wall_pts[dists >= median_d]
        inner_line = _order_boundary(inner_pts)
        outer_line = _order_boundary(outer_pts)
        return outer_line.astype(np.float32), inner_line.astype(np.float32)


class MappingWorker(BaseWorker):
    """Worker that accumulates BEV data and extracts track boundaries.

    Config kwargs
    -------------
    pose_buffer : PoseBuffer
        Vehicle poses from SLAM.
    segment_buffer : SegmentationBuffer
        Wall masks from vision.
    pointcloud_buffer : PointCloudBuffer
        3D map points from SLAM.
    boundary_buffer : TrackBoundaryBuffer
        Output: extracted track boundaries.
    loop_event : mp.Event
        Set by SLAMWorker when loop closure detected.
    mapping_done_event : mp.Event
        Set by this worker when boundaries are extracted and ready.
    alpha_shape : float
        Alpha-shape parameter (default 25.0).
    boundary_min_points : int
        Minimum wall points before attempting extraction (default 50).
    """

    def __init__(self, name: str = "mapping", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._pose_buf: PoseBuffer | None = None
        self._seg_buf: SegmentationBuffer | None = None
        self._pc_buf: PointCloudBuffer | None = None
        self._boundary_buf: TrackBoundaryBuffer | None = None
        self._loop_event: Optional[mp.Event] = None
        self._mapping_done_event: Optional[mp.Event] = None

        self._alpha: float = 25.0
        self._min_points: int = 50

        # Accumulated data
        self._bev_wall_pts: list[np.ndarray] = []
        self._bev_trajectory: list[np.ndarray] = []
        self._last_pose_seq = 0
        self._last_pc_seq = 0
        self._last_seg_seq = 0
        self._boundaries_extracted = False

        # Latest wall mask for labelling
        self._latest_mask: Optional[np.ndarray] = None

    def setup(self) -> None:
        self._pose_buf = self.config["pose_buffer"]
        self._seg_buf = self.config["segment_buffer"]
        self._pc_buf = self.config["pointcloud_buffer"]
        self._boundary_buf = self.config["boundary_buffer"]
        self._loop_event = self.config.get("loop_event")
        self._mapping_done_event = self.config.get("mapping_done_event")
        self._alpha = self.config.get("alpha_shape", 25.0)
        self._min_points = self.config.get("boundary_min_points", 50)

        self.logger.info(
            f"MappingWorker ready (alpha={self._alpha}, "
            f"min_pts={self._min_points})"
        )

    def step(self) -> None:
        if self._boundaries_extracted:
            # Phase B complete — just sleep
            time.sleep(0.5)
            return

        #  Read latest wall mask ─
        seg_result = self._seg_buf.read_latest()
        if seg_result is not None:
            seg_header, mask = seg_result
            if seg_header.seq_no > self._last_seg_seq:
                self._latest_mask = mask
                self._last_seg_seq = seg_header.seq_no

        #  Read latest pose → accumulate trajectory 
        pose_result = self._pose_buf.read_latest()
        if pose_result is not None:
            pose_header, pose, _ = pose_result
            if pose_header.seq_no > self._last_pose_seq:
                # Extract translation (X, Z) from 4×4 pose
                tx, tz = pose[0, 3], pose[2, 3]
                self._bev_trajectory.append(np.array([tx, tz], dtype=np.float32))
                self._last_pose_seq = pose_header.seq_no

        #  Read latest point cloud → label & accumulate wall pts ─
        pc_result = self._pc_buf.read_latest()
        if pc_result is not None:
            pc_header, pts_3d = pc_result
            if pc_header.seq_no > self._last_pc_seq and len(pts_3d) > 0:
                # For now, accept all points as wall points
                # TODO: use self._latest_mask + camera intrinsics to filter
                bev_pts = _to_bev(pts_3d)
                self._bev_wall_pts.append(bev_pts)
                self._last_pc_seq = pc_header.seq_no
                self.logger.debug(
                    f"Accumulated {len(bev_pts)} BEV wall pts "
                    f"(total batches: {len(self._bev_wall_pts)})"
                )

        #  Check loop closure → extract boundaries ─
        if (
            self._loop_event is not None
            and self._loop_event.is_set()
            and not self._boundaries_extracted
        ):
            self._extract_boundaries()

        time.sleep(0.01)

    def _extract_boundaries(self) -> None:
        """Merge accumulated wall points and extract track boundaries."""
        if not self._bev_wall_pts:
            self.logger.warning("No wall points accumulated — cannot extract")
            return

        all_wall = np.vstack(self._bev_wall_pts)
        trajectory = (
            np.array(self._bev_trajectory, dtype=np.float32)
            if self._bev_trajectory
            else np.zeros((1, 2), dtype=np.float32)
        )

        if len(all_wall) < self._min_points:
            self.logger.warning(
                f"Only {len(all_wall)} wall points (need {self._min_points}) "
                f"— skipping extraction"
            )
            return

        self.logger.info(
            f"Extracting boundaries from {len(all_wall)} wall pts, "
            f"{len(trajectory)} trajectory pts …"
        )

        try:
            outer_line, inner_line = extract_track_boundaries(
                all_wall, trajectory, alpha=self._alpha
            )
        except Exception:
            self.logger.exception("Failed to extract boundaries")
            return

        # Write to boundary buffer
        self._boundary_buf.write(outer_line, inner_line)
        self._boundaries_extracted = True

        # Print summary
        origin_bev = trajectory[0] if len(trajectory) > 0 else np.zeros(2)
        self.logger.info("=" * 60)
        self.logger.info(
            f"Track mapped. {len(all_wall)} wall points."
        )
        self.logger.info(f"Outer boundary: {len(outer_line)} vertices")
        self.logger.info(f"Inner boundary: {len(inner_line)} vertices")
        self.logger.info(
            f"Origin: ({origin_bev[0]:.2f}, {origin_bev[1]:.2f})"
        )
        self.logger.info("=" * 60)

        # Signal that mapping is done
        if self._mapping_done_event is not None:
            self._mapping_done_event.set()

    def teardown(self) -> None:
        total_wall = sum(len(b) for b in self._bev_wall_pts)
        self.logger.info(
            f"MappingWorker shutting down — "
            f"{total_wall} wall pts, {len(self._bev_trajectory)} traj pts, "
            f"boundaries_extracted={self._boundaries_extracted}"
        )
