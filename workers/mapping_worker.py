"""
MappingWorker: records the driven BEV trajectory (PoseBuffer, or SLAM keyframes if
given) and the newest wall point cloud, and on ``loop_event`` extracts outer/inner
track boundaries (extract_track_boundaries) into TrackBoundaryBuffer, then sets
mapping_done_event. map_reset_event (SLAM world frame changed) drops everything recorded.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from .bev import pose_to_bev
from shared_buffer import PoseBuffer
from shared_buffer.pointcloud_buffer import PointCloudBuffer
from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer


def _to_bev(points_3d: np.ndarray, axes: tuple = (0, 2)) -> np.ndarray:
    """3D -> BEV, just picks out the two horizontal axes."""
    return points_3d[:, list(axes)]


def split_walls_by_trajectory(
    bev_wall_pts: np.ndarray,
    trajectory_xyh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(N, 2) BEV wall points + (T, 3) [x, y, heading] path -> (left, right), each
    (M, 2) float32 ordered by arc length; side from the sign of heading x (pose -> point).
    No filtering; extract_track_boundaries is the full pipeline."""
    from scipy.spatial import KDTree

    traj_pos = trajectory_xyh[:, :2]   # (T, 2)
    traj_hdg = trajectory_xyh[:, 2]    # (T,)

    arc = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(traj_pos, axis=0), axis=1)
    )])  # (T,)

    tree = KDTree(traj_pos)
    _, nearest_idx = tree.query(bev_wall_pts)  # (N,)

    hdgs = traj_hdg[nearest_idx]  # (N,)
    fwd_x = np.cos(hdgs)
    fwd_y = np.sin(hdgs)

    v = bev_wall_pts - traj_pos[nearest_idx]  # (N, 2)

    # > 0: wall on the left
    cross = fwd_x * v[:, 1] - fwd_y * v[:, 0]  # (N,)
    left_mask = cross > 0

    left_pts = bev_wall_pts[left_mask]
    right_pts = bev_wall_pts[~left_mask]

    def _order_by_arc(pts: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        order = np.argsort(arc[idx])
        return pts[order].astype(np.float32)

    left_pts = _order_by_arc(left_pts, nearest_idx[left_mask])
    right_pts = _order_by_arc(right_pts, nearest_idx[~left_mask])

    return left_pts, right_pts


def clean_trajectory(
    traj: np.ndarray,
    min_step: float = 0.02,
    lap_close_radius: float = 0.5,
    min_lap_length: float = 3.0,
) -> tuple[np.ndarray, bool]:
    """Drop stationary samples and trim to one lap.

    traj is (T, D) with the BEV position in columns 0-1. A sample is kept
    only if it is at least min_step from the previously kept one. The lap
    ends at the closest approach to the start within lap_close_radius, after
    at least min_lap_length of driving. Returns (trajectory, closed).
    """
    if len(traj) == 0:
        return traj, False
    keep = [0]
    for i in range(1, len(traj)):
        if np.linalg.norm(traj[i, :2] - traj[keep[-1], :2]) >= min_step:
            keep.append(i)
    traj = traj[keep]
    if len(traj) < 3:
        return traj, False

    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))])
    d0 = np.linalg.norm(traj[:, :2] - traj[0, :2], axis=1)
    near = np.flatnonzero((arc >= min_lap_length) & (d0 < lap_close_radius))
    if len(near) == 0:
        return traj, False
    end = near[0]
    while end + 1 < len(traj) and d0[end + 1] < d0[end]:
        end += 1
    return traj[: end + 1], True


def floor_heights(
    points: np.ndarray, floor_guess: float, tol: float = 0.1, iterations: int = 200, seed: int = 0,
) -> Optional[np.ndarray]:
    """Height of each (N, 3) [bev0, bev1, up] point above the dominant near-horizontal
    plane within tol of floor_guess (RANSAC + least squares), which absorbs residual
    SLAM tilt. None if no floor is found."""
    cand = points[np.abs(points[:, 2] - floor_guess) < tol].astype(np.float64)
    if len(cand) < 50:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iterations):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm * np.sign(n[2] or 1.0)
        if n[2] < 0.95:                          # within ~18 deg of horizontal
            continue
        inliers = np.abs((cand - p[0]) @ n) < 0.02
        if inliers.sum() > best_n:
            best, best_n = inliers, int(inliers.sum())
    if best is None or best_n < 0.3 * len(cand):
        return None
    c = cand[best].mean(axis=0)
    n = np.linalg.svd(cand[best] - c, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    return (points - c) @ n


def signed_area(xy: np.ndarray) -> float:
    """Shoelace area of the closed polygon; > 0 for counter-clockwise."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _resample(xy: np.ndarray, ds: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ds metres along the polyline, and their arc length."""
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    s = np.arange(0.0, arc[-1] + 1e-9, ds)
    return np.column_stack([np.interp(s, arc, xy[:, 0]), np.interp(s, arc, xy[:, 1])]), s


def _smooth(pts: np.ndarray, window: int, closed: bool) -> np.ndarray:
    """Moving average along an ordered polyline (wraps around if closed)."""
    if window <= 1 or len(pts) < window:
        return pts
    k = np.ones(window) / window
    h = window // 2
    if closed:
        padded = np.concatenate([pts[-h:], pts, pts[:h]])
    else:
        padded = np.concatenate([np.repeat(pts[:1], h, 0), pts, np.repeat(pts[-1:], h, 0)])
    return np.column_stack([np.convolve(padded[:, i], k, mode="valid") for i in range(2)])[: len(pts)]


def _wall_offset(offsets: np.ndarray, layer: float = 0.10, dense: float = 0.7) -> tuple[float, int]:
    """Wall distance from one bin's lateral offsets: the nearest layer-wide cell with
    at least `dense` x the fullest cell's count (ignores strays in front and points
    behind the wall). Returns (offset m, points in that cell)."""
    cells = np.floor(offsets / layer).astype(np.int64)
    ids, counts = np.unique(cells, return_counts=True)
    k = np.flatnonzero(counts >= dense * counts.max())[0]
    return float(np.median(offsets[np.abs(cells - ids[k]) <= 1])), int(counts[k])


def extract_track_boundaries(
    wall_bev: np.ndarray,
    wall_height: Optional[np.ndarray],
    traj_bev: np.ndarray,
    closed: bool,
    max_half_width: float = 1.5,
    height_band: tuple[float, float] = (0.05, 1.0),
    bin_size: float = 0.10,
    smooth_window: int = 5,
    min_bin_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Wall points -> (outer, inner, ccw) polylines ordered along the lap.

    wall_height (N,) is height above the floor (None: no filter); traj_bev is the
    cleaned trajectory (clean_trajectory). Per arc-length bin and side the wall is
    at _wall_offset; thin bins are interpolated, then smoothed. CCW: right = outer.
    """
    from scipy.spatial import KDTree

    ccw = signed_area(traj_bev) > 0

    keep = np.ones(len(wall_bev), dtype=bool)
    if wall_height is not None:
        keep &= (wall_height >= height_band[0]) & (wall_height <= height_band[1])
    pts = wall_bev[keep]

    ds = min(bin_size / 2.0, 0.05)
    path, s = _resample(traj_bev.astype(np.float64), ds)
    if len(path) < 2 or len(pts) == 0:
        empty = np.empty((0, 2), np.float32)
        return empty, empty, ccw
    tangent = np.gradient(path, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)

    dist, idx = KDTree(path).query(pts)
    gate = dist < max_half_width
    pts, dist, idx = pts[gate], dist[gate], idx[gate]

    v = pts - path[idx]
    cross = tangent[idx, 0] * v[:, 1] - tangent[idx, 1] * v[:, 0]
    bins = np.floor(s[idx] / bin_size).astype(np.int64)

    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])   # left of travel

    n_bins = int(np.floor(s[-1] / bin_size)) + 1
    centre_idx = np.minimum(np.round((np.arange(n_bins) + 0.5) * bin_size / ds).astype(int), len(path) - 1)

    # sparse maps: scale the per-bin minimum with the typical count (both walls
    # together, so one sparsely seen wall can't lower its own bar)
    counts = np.unique(bins * 2 + (cross > 0), return_counts=True)[1]
    need = int(np.clip(0.25 * np.median(counts), 1, min_bin_points)) if len(counts) else 1

    def _side(mask: np.ndarray, sign: float) -> np.ndarray:
        if not np.any(mask):
            return np.empty((0, 2), np.float32)
        have, offset = [], []
        for b in np.unique(bins[mask]):
            sel = mask & (bins == b)
            if sel.sum() < need:
                continue
            off, layer_count = _wall_offset(np.abs(cross[sel]))
            if layer_count >= need:            # a real layer, not a few strays
                have.append(b)
                offset.append(off)
        if len(have) < 2:
            return np.empty((0, 2), np.float32)
        have, offset = np.array(have), np.array(offset)
        all_bins = np.arange(n_bins)
        if closed:
            off = np.interp(all_bins, have, offset, period=n_bins)
        else:
            all_bins = np.arange(have[0], have[-1] + 1)
            off = np.interp(all_bins, have, offset)
        if smooth_window > 1 and len(off) >= smooth_window:
            h = smooth_window // 2
            padded = np.r_[off[-h:], off, off[:h]] if closed else np.r_[[off[0]] * h, off, [off[-1]] * h]
            off = np.convolve(padded, np.ones(smooth_window) / smooth_window, mode="valid")[: len(all_bins)]
        j = centre_idx[all_bins]
        return (path[j] + sign * off[:, None] * normal[j]).astype(np.float32)

    left, right = _side(cross > 0, 1.0), _side(cross <= 0, -1.0)
    if ccw:
        return right, left, ccw
    return left, right, ccw


class MappingWorker(BaseWorker):
    """Accumulates BEV data and extracts the track boundaries.

    Config kwargs:
      pose_buffer             PoseBuffer
      pointcloud_buffer       PointCloudBuffer, 3D wall points
      keyframe_buffer         SLAM keyframe positions, used as trajectory if >= 10 (optional)
      boundary_buffer         TrackBoundaryBuffer
      loop_event              starts extraction (operator [l] or SLAM loop)
      mapping_done_event      set once boundaries are written
      map_reset_event         drop everything and start over; cleared here (optional)
      boundary_min_points     min wall points to extract, default 50
      boundary_save_path      directory for outer_line.npy / inner_line.npy (optional)
      bev_axes                horizontal axes: sim [0, 1], SLAM level frame [0, 2] (default)
      forward_axis            body travel axis: sim 0, SLAM 2 (default)
      up_sign                 +1 if the vertical axis points up; default -1 if it is axis 1
      camera_height           pose origin height above the floor, default 0.15 m
      wall_height_band        kept wall heights above the floor, default [0.05, 1.0] m
      max_track_half_width    lateral gate from the trajectory, default 1.5 m
      boundary_bin_size       arc-length bin, default 0.10 m
      boundary_smooth_window  moving-average window in bins, default 5
      lap_close_radius, min_lap_length   lap trimming (clean_trajectory), default 0.5 / 3.0 m
      auto_loop_close         set loop_event itself once the pose trajectory is back within
                              lap_close_radius of its start after auto_loop_min_length m
                              (ground-truth pose in the sim; no SLAM loop detection there)
      auto_loop_min_length    m, default 8.0
      auto_loop_overrun       m driven past the closest approach to the start before the
                              lap is closed, so the wall map covers the seam, default 1.0
      jump_threshold          m; a pose step longer than this is a teleport (the sim respawns
                              the car after a wall hit): the trajectory is cut back to the
                              point nearest the new pose. Default None (off): SLAM poses can
                              legitimately be sparse
    """

    def __init__(self, name: str = "mapping", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: PointCloudBuffer | None = None
        self._kf_buf: PointCloudBuffer | None = None
        self._boundary_buf: TrackBoundaryBuffer | None = None
        self._loop_event: Optional[mp.Event] = None
        self._mapping_done_event: Optional[mp.Event] = None
        self._map_reset_event: Optional[mp.Event] = None

        self._min_points: int = 50
        self._bev_axes: tuple = (0, 2)
        self._save_path: Optional[str] = None

        self._bev_wall_pts: list[np.ndarray] = []
        self._wall_vertical: Optional[np.ndarray] = None
        self._bev_trajectory: list[np.ndarray] = []  # (x, y, heading_rad, vertical) rows
        self._kf_traj: Optional[np.ndarray] = None   # (K, 3) 3D keyframe positions
        self._last_pose_seq = 0
        self._last_pc_seq = 0
        self._last_kf_seq = 0
        self._boundaries_extracted = False

    def setup(self) -> None:
        c = self.config
        self._pose_buf = c["pose_buffer"]
        self._pc_buf = c["pointcloud_buffer"]
        self._kf_buf = c.get("keyframe_buffer")
        self._boundary_buf = c["boundary_buffer"]
        self._loop_event = c.get("loop_event")
        self._mapping_done_event = c.get("mapping_done_event")
        self._map_reset_event = c.get("map_reset_event")
        self._min_points = c.get("boundary_min_points", 50)
        self._bev_axes = tuple(c.get("bev_axes", [0, 2]))
        self._forward_axis = int(c.get("forward_axis", 2))
        self._save_path = c.get("boundary_save_path")
        self._vert_axis = 3 - sum(self._bev_axes)
        self._up_sign = int(c.get("up_sign", -1 if self._vert_axis == 1 else 1))
        self._camera_height = float(c.get("camera_height", 0.15))
        self._height_band = tuple(c.get("wall_height_band", [0.05, 1.0]))
        self._half_width = float(c.get("max_track_half_width", 1.5))
        self._bin_size = float(c.get("boundary_bin_size", 0.10))
        self._smooth_window = int(c.get("boundary_smooth_window", 5))
        self._lap_close_radius = float(c.get("lap_close_radius", 0.5))
        self._min_lap_length = float(c.get("min_lap_length", 3.0))
        self._auto_loop = bool(c.get("auto_loop_close", False))
        self._auto_loop_min = float(c.get("auto_loop_min_length", 8.0))
        self._auto_loop_overrun = float(c.get("auto_loop_overrun", 1.0))
        jump = c.get("jump_threshold")
        self._jump = float(jump) if jump is not None else math.inf
        self._path_length = 0.0
        self._path_s: list[float] = []      # arc length at each trajectory point
        self._max_dist_from_start = 0.0
        self._closest_start: Optional[tuple[float, float]] = None   # (distance, arc length)

        self.logger.info(
            f"MappingWorker ready (min_pts={self._min_points}, "
            f"bev_axes={self._bev_axes}, forward_axis={self._forward_axis}, "
            f"half_width={self._half_width}m, height_band={self._height_band}m)"
        )

    def _reset(self) -> None:
        """SLAM changed maps: everything recorded is in a stale frame."""
        self.logger.warning(
            f"Map reset signalled — dropping {len(self._bev_trajectory)} poses and "
            "the wall map, recording starts again (drive the lap again)"
        )
        self._bev_trajectory = []
        self._path_length = 0.0
        self._path_s = []
        self._max_dist_from_start = 0.0
        self._closest_start = None
        self._bev_wall_pts = []
        self._wall_vertical = None
        self._kf_traj = None
        # anything already in the buffers may still be from the old map
        self._last_pose_seq = self._pose_buf.latest_seq()
        self._last_pc_seq = self._pc_buf.latest_seq()
        if self._kf_buf is not None:
            self._last_kf_seq = self._kf_buf.latest_seq()
        self._map_reset_event.clear()

    def step(self) -> None:
        if self._boundaries_extracted:
            time.sleep(0.5)
            return

        if self._map_reset_event is not None and self._map_reset_event.is_set():
            self._reset()

        if self._pose_buf.latest_seq() > self._last_pose_seq:
            pose_result = self._pose_buf.read_latest()
            if pose_result is not None and pose_result[0].seq_no > self._last_pose_seq:
                pose_header, pose, _, _ = pose_result
                (x, y), heading = pose_to_bev(pose, self._bev_axes, self._forward_axis)
                if self._bev_trajectory:
                    step = float(np.hypot(x - self._bev_trajectory[-1][0], y - self._bev_trajectory[-1][1]))
                    if step > self._jump:
                        self._cut_back_to(x, y, step)
                    else:
                        self._path_length += step
                self._bev_trajectory.append(
                    np.array([x, y, heading, pose[self._vert_axis, 3]], dtype=np.float32)
                )
                self._path_s.append(self._path_length)
                self._last_pose_seq = pose_header.seq_no
                if self._auto_loop:
                    self._check_auto_loop(x, y)

        # the cloud is the full deduplicated wall map, so replace rather than append
        if self._pc_buf.latest_seq() > self._last_pc_seq:
            pc_result = self._pc_buf.read_latest()
            if pc_result is not None:
                pc_header, pts_3d = pc_result
                if pc_header.seq_no > self._last_pc_seq and len(pts_3d) > 0:
                    self._bev_wall_pts = [_to_bev(pts_3d, self._bev_axes)]
                    self._wall_vertical = pts_3d[:, self._vert_axis].copy()
                    self._last_pc_seq = pc_header.seq_no
                    self.logger.debug(f"Wall map updated: {len(pts_3d)} pts")

        if self._kf_buf is not None and self._kf_buf.latest_seq() > self._last_kf_seq:
            kf_result = self._kf_buf.read_latest()
            if kf_result is not None and kf_result[0].seq_no > self._last_kf_seq:
                self._kf_traj = kf_result[1]
                self._last_kf_seq = kf_result[0].seq_no

        if (
            self._loop_event is not None
            and self._loop_event.is_set()
            and not self._boundaries_extracted
        ):
            self._extract_boundaries()

        time.sleep(0.01)

    def _cut_back_to(self, x: float, y: float, step: float) -> None:
        """Teleport (sim respawn after a wall hit): drop the stretch the car is put back
        over, so the lap doesn't double back on itself."""
        traj = np.array(self._bev_trajectory, dtype=np.float32)[:, :2]
        lo = max(0, len(traj) - 300)                    # the last few metres only
        k = lo + int(np.argmin(np.hypot(traj[lo:, 0] - x, traj[lo:, 1] - y)))
        dropped = len(traj) - k
        del self._bev_trajectory[k:]
        del self._path_s[k:]
        self._path_length = self._path_s[-1] if self._path_s else 0.0
        self.logger.warning(f"Pose jumped {step:.2f} m (sim respawn?) — trajectory cut back by {dropped} poses")

    def _check_auto_loop(self, x: float, y: float) -> None:
        """Back past the start after a long enough drive: end the mapping lap."""
        if self._loop_event is None or self._loop_event.is_set():
            return
        x0, y0 = self._bev_trajectory[0][:2]
        d = float(np.hypot(x - x0, y - y0))
        self._max_dist_from_start = max(self._max_dist_from_start, d)
        if self._path_length < self._auto_loop_min or self._max_dist_from_start <= 2 * self._lap_close_radius:
            return
        if d < self._lap_close_radius and (self._closest_start is None or d < self._closest_start[0]):
            self._closest_start = (d, self._path_length)
        # overrun the start a little so the wall map covers the seam; clean_trajectory
        # trims the lap at the closest approach
        if self._closest_start is not None and self._path_length - self._closest_start[1] >= self._auto_loop_overrun:
            self.logger.info(f"Auto loop closure: back at the start after {self._closest_start[1]:.1f} m")
            self._loop_event.set()

    def _trajectory(self) -> tuple[np.ndarray, str]:
        """(T, 3) [bev0, bev1, vertical] trajectory: keyframes if available."""
        if self._kf_traj is not None and len(self._kf_traj) >= 10:
            k = self._kf_traj
            return np.column_stack([_to_bev(k, self._bev_axes), k[:, self._vert_axis]]), "keyframes"
        t = np.array(self._bev_trajectory, dtype=np.float32).reshape(-1, 4)
        return t[:, [0, 1, 3]], "poses"

    def _retry(self, msg: str) -> None:
        self.logger.warning(f"{msg} — will retry in 2s")
        time.sleep(2.0)

    def _extract_boundaries(self) -> None:
        if not self._bev_wall_pts:
            return self._retry("No wall points accumulated")

        all_wall = np.vstack(self._bev_wall_pts)
        if len(all_wall) < self._min_points:
            return self._retry(f"Only {len(all_wall)} wall points (need {self._min_points})")

        raw_traj, source = self._trajectory()
        traj, closed = clean_trajectory(
            raw_traj, lap_close_radius=self._lap_close_radius,
            min_lap_length=self._min_lap_length,
        )
        if len(traj) < 2:
            return self._retry(f"Not enough moving trajectory points ({len(traj)} from {source})")

        wall_height = None
        if self._wall_vertical is not None and len(self._wall_vertical) == len(all_wall):
            up = self._up_sign * self._wall_vertical
            cam_up = self._up_sign * float(np.median(traj[:, 2]))
            wall_height = floor_heights(np.column_stack([all_wall, up]), cam_up - self._camera_height)
            if wall_height is None:
                self.logger.warning("No floor plane in the map — heights from camera_height instead")
                wall_height = up - cam_up + self._camera_height

        self.logger.info(
            f"Extracting boundaries: {len(all_wall)} wall pts, {len(traj)} trajectory "
            f"pts from {source} ({'closed lap' if closed else 'open path'}) …"
        )
        try:
            outer, inner, ccw = extract_track_boundaries(
                all_wall, wall_height, traj[:, :2], closed,
                max_half_width=self._half_width, height_band=self._height_band,
                bin_size=self._bin_size, smooth_window=self._smooth_window,
            )
        except Exception:
            self.logger.exception("Boundary extraction failed")
            return

        if len(outer) == 0 or len(inner) == 0:
            return self._retry(f"One side empty (outer={len(outer)}, inner={len(inner)})")

        self._boundary_buf.write(outer, inner)
        self._boundaries_extracted = True

        if self._save_path is not None:
            import os
            os.makedirs(self._save_path, exist_ok=True)
            np.save(os.path.join(self._save_path, "outer_line.npy"), outer)
            np.save(os.path.join(self._save_path, "inner_line.npy"), inner)
            self.logger.info(f"Boundaries saved to {self._save_path}")

        self.logger.info("=" * 60)
        self.logger.info(
            f"Track mapped ({'counter-clockwise' if ccw else 'clockwise'} lap). "
            f"{len(all_wall)} wall points."
        )
        self.logger.info(f"Outer boundary: {len(outer)} pts")
        self.logger.info(f"Inner boundary: {len(inner)} pts")
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
