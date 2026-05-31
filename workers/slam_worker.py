"""
SLAMWorker: FrameBuffer -> ORB_SLAM3 RGBD -> PoseBuffer (Twc, camera-to-world), map points
and keyframe positions. Poses carry the frame's capture ts (time.monotonic()) and are only
published while tracking is OK, in a floor-aligned "level" world frame (Y down along the
floor normal, Z = forward on the floor, so BEV axes [0, 2] hold). A map switch sets
map_reset_event; in localization mode poses are withheld outside the reference map.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, PoseBuffer


# ORB_SLAM3::Tracking::eTrackingState
_TRACKING_OK = 2
_STATE_NAMES = {
    0: "NO_IMAGES_YET", 1: "NOT_INITIALIZED", 2: "OK", 3: "RECENTLY_LOST", 4: "LOST",
}


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform (R^T, -R^T t)."""
    R, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def level_rotation(n_down: np.ndarray) -> np.ndarray:
    """3x3 rotation from the camera frame to a floor-aligned frame.

    n_down is the floor normal in camera coordinates pointing down (away
    from the camera, towards the floor). Rows of the result are the level
    axes in camera coordinates: X = Y x Z, Y = n_down, Z = camera forward
    projected onto the floor. p_level = R @ p_cam.
    """
    y = np.asarray(n_down, dtype=np.float64)
    y = y / np.linalg.norm(y)
    z = np.array([0.0, 0.0, 1.0]) - y[2] * y
    z /= np.linalg.norm(z)
    x = np.cross(y, z)
    return np.stack([x, y, z]).astype(np.float32)


def read_settings_intrinsics(path: str) -> Optional[SimpleNamespace]:
    """fx, fy, cx, cy (and width, height) from an ORB-SLAM3 settings YAML."""
    import cv2

    from control.types import resolve_path

    path = resolve_path(path)
    if not os.path.isfile(path):
        return None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        vals = {}
        for key in ("fx", "fy", "cx", "cy"):
            node = fs.getNode(f"Camera1.{key}")
            if node.empty():
                node = fs.getNode(f"Camera.{key}")
            if node.empty():
                return None
            vals[key] = float(node.real())
        for key in ("width", "height"):
            node = fs.getNode(f"Camera.{key}")
            vals[key] = int(node.real()) if not node.empty() else None
        return SimpleNamespace(**vals)
    finally:
        fs.release()


class SLAMWorker(BaseWorker):
    """Visual SLAM worker (ORB_SLAM3 through pybind11).

    Config kwargs:
      input_buffer          FrameBuffer
      output_buffer         PoseBuffer (Twc, level frame, capture timestamps)
      pointcloud_buffer     PointCloudBuffer for map points (optional)
      keyframe_buffer       PointCloudBuffer for keyframe positions, loop-corrected (optional)
      vocab_path            ORBvoc.txt
      settings_path         ORB_SLAM3 settings YAML (also the floor-fit intrinsics)
      loop_event            set on loop closure once global BA has settled
      localize_event        when set, switch to localization-only mode
      map_reset_event       set when the world frame changes; cleared by MappingWorker
      warmup_frames         frames skipped while stereo depth settles, default 15
      level_world           publish in the floor-aligned frame, default True
      level_frames          good floor fits to average, default 5
      level_timeout_frames  give up levelling after this many frames, default 150
      floor_height_range    accepted camera height above the floor, default (0.03, 1.0) m
      synthetic             allow the identity-pose stub without orbslam3_python
    """

    _PC_PUBLISH_INTERVAL = 30  # also publish point cloud every N frames
    _LOOP_SETTLE_S = 1.0       # map stable this long after the last change before signalling a loop

    def __init__(self, name: str = "slam", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: Optional[Any] = None
        self._kf_buf: Optional[Any] = None
        self._slam = None
        self._last_seq = 0
        self._frame_count = 0
        self._track_ms: list = []    # track_rgbd time since the last status line
        self._loop_event: Optional[mp.Event] = None
        self._localize_event: Optional[mp.Event] = None
        self._map_reset_event: Optional[mp.Event] = None
        self._localization_activated = False
        self._tracking_ok = True
        self._warmup_frames = 0

        # floor alignment
        self._loop_pending = False
        self._last_map_change = 0.0
        self._level_world = bool(kwargs.get("level_world", True))
        self._level_frames = int(kwargs.get("level_frames", 5))
        self._level_timeout = int(kwargs.get("level_timeout_frames", 150))
        self._floor_height_range = tuple(kwargs.get("floor_height_range", (0.03, 1.0)))
        self._K: Optional[SimpleNamespace] = None
        self._plane_fitter = None
        self._plane_fits: list[np.ndarray] = []
        self._level_tries = 0
        self._R_level: Optional[np.ndarray] = None  # 4x4, world -> level

        # map identity
        self._map_key: Optional[tuple] = None   # current (map_id, init_kf_id)
        self._ref_map_key: Optional[tuple] = None  # map the poses are published in

    def setup(self) -> None:
        self._frame_buf = self.config["input_buffer"]
        self._pose_buf = self.config["output_buffer"]
        self._pc_buf = self.config.get("pointcloud_buffer")
        self._kf_buf = self.config.get("keyframe_buffer")
        self._loop_event = self.config.get("loop_event")
        self._localize_event = self.config.get("localize_event")
        self._map_reset_event = self.config.get("map_reset_event")
        self._warmup_frames = int(self.config.get("warmup_frames", 15))
        synthetic = (
            bool(self.config.get("synthetic", False))
            or os.environ.get("PILOT_SYNTHETIC", "") == "1"
        )

        vocab_path = self.config.get(
            "vocab_path", "ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        settings_path = self.config.get(
            "settings_path", "orbslam_oakd_rgbd.yaml"
        )

        try:
            import orbslam3_python as _slam_mod
        except ImportError as exc:
            if not synthetic:
                raise RuntimeError(
                    f"orbslam3_python could not be imported ({exc}). Build the "
                    "ORB-SLAM3 bindings (docs/slam.md). If the module exists, a "
                    "shared library is not found: rebuild orbslam3_wrapper (it "
                    "sets an RPATH) or add ORB_SLAM3/lib and its Thirdparty lib "
                    "dirs to LD_LIBRARY_PATH. The identity-pose stub is only "
                    "allowed in synthetic mode (synthetic: true / PILOT_SYNTHETIC=1)."
                ) from exc
            self.logger.warning(
                "orbslam3_python not available — synthetic mode, running SLAM "
                "stub (identity poses, no point cloud)"
            )
            self._slam = None
            return

        from control.types import resolve_path

        vocab_path, settings_path = resolve_path(vocab_path), resolve_path(settings_path)
        self._slam = _slam_mod.System(vocab_path, settings_path, _slam_mod.RGBD)
        self.logger.info(
            f"ORB_SLAM3 initialised (vocab={vocab_path}, settings={settings_path})"
        )
        if self._level_world:
            self._K = read_settings_intrinsics(settings_path)
            if self._K is None:
                self.logger.warning(
                    f"No intrinsics in {settings_path} — cannot fit the floor, "
                    "publishing the unlevelled ORB-SLAM3 frame"
                )

    # floor alignment

    def _update_level(self, depth: np.ndarray) -> None:
        """Fit the floor in this frame's depth until R_level is known."""
        if not self._level_world or self._K is None:
            self._R_level = np.eye(4, dtype=np.float32)
            return
        if self._plane_fitter is None:
            from vision.segmenters import GroundPlaneConfig, GroundPlaneSegmenter
            self._plane_fitter = GroundPlaneSegmenter(
                self._K, GroundPlaneConfig(camera_height_range=self._floor_height_range))

        self._level_tries += 1
        pts, valid = self._plane_fitter._points(depth)
        lower = valid.copy()
        lower[: lower.shape[0] // 2] = False  # floor is in the lower half
        # n points up, n.p + d = 0, d = camera height (planes above the camera are rejected)
        plane = self._plane_fitter.fit_plane(pts[lower])
        if plane is not None:
            self._plane_fits.append(plane)

        if len(self._plane_fits) >= self._level_frames:
            planes = np.array(self._plane_fits)
            n_up = planes[:, :3].mean(axis=0)
            n_up /= np.linalg.norm(n_up)
            cam_height = float(np.median(planes[:, 3]))
            R = np.eye(4, dtype=np.float32)
            R[:3, :3] = level_rotation(-n_up)
            self._R_level = R
            pitch = math.degrees(math.atan2(-n_up[2], -n_up[1]))
            roll = math.degrees(math.atan2(n_up[0], -n_up[1]))
            self.logger.info(
                f"Floor fitted from {len(planes)} frames: camera pitch {pitch:.1f} deg, "
                f"roll {roll:.1f} deg, height {cam_height:.3f} m — publishing in the "
                f"level world frame"
            )
        elif self._level_tries >= self._level_timeout:
            self._R_level = np.eye(4, dtype=np.float32)
            self.logger.warning(
                f"Floor not found in {self._level_tries} frames "
                f"({len(self._plane_fits)} good fits) — publishing the unlevelled "
                "ORB-SLAM3 frame (BEV will be tilted)"
            )

    def _level_points(self, pts: np.ndarray) -> np.ndarray:
        return (pts @ self._R_level[:3, :3].T).astype(np.float32)

    # map identity

    def _check_map(self) -> bool:
        """Watch for new / re-initialised maps. True if poses may be published."""
        if not hasattr(self._slam, "map_info"):
            return True
        map_id, init_kf, n_kf = self._slam.map_info()
        key = (int(map_id), int(init_kf))
        if n_kf == 0:
            return False  # map was just cleared, nothing to localise against
        if key != self._map_key:
            if self._map_key is not None:
                self.logger.warning(
                    f"ORB-SLAM3 switched map {self._map_key} -> {key} "
                    f"({n_kf} keyframes): the world frame changed"
                )
                if self._map_reset_event is not None:
                    self._map_reset_event.set()
            self._map_key = key
            if self._ref_map_key is None or not self._localization_activated:
                self._ref_map_key = key
        if key != self._ref_map_key:
            return False  # localization mode in a map that isn't the mapped track
        return True

    def step(self) -> None:
        if (
            not self._localization_activated
            and self._localize_event is not None
            and self._localize_event.is_set()
        ):
            if self._slam is not None:
                self._slam.activate_localization_mode()
                self.logger.info("Switched to localization-only mode")
            self._localization_activated = True

        if self._frame_buf.latest_seq() <= self._last_seq:
            time.sleep(0.005)
            return
        result = self._frame_buf.read_latest()
        if result is None:
            time.sleep(0.01)
            return

        header, rgb, depth = result
        if header.seq_no <= self._last_seq:
            time.sleep(0.005)
            return
        self._last_seq = header.seq_no
        ts = header.produced_ts  # capture time (time.monotonic clock)

        self._frame_count += 1

        if self._slam is None:
            # stub (synthetic mode only), no orbslam3_python
            self._pose_buf.write(np.eye(4, dtype=np.float32), ts=ts)
            return

        # warm-up frames: sparse early depth makes ORB-SLAM3 init a thin map and
        # keep resetting; they still feed the floor fit
        if self._R_level is None:
            self._update_level(depth)
        if self._frame_count <= self._warmup_frames:
            return

        # sanity check on the first tracked frame, depth should be in metres
        if self._frame_count == self._warmup_frames + 1:
            nonzero = depth[depth > 0]
            self.logger.debug(
                f"[SLAM depth check] shape={depth.shape} dtype={depth.dtype} "
                f"min={float(depth.min()):.4f} max={float(depth.max()):.4f} "
                f"mean_nonzero={float(nonzero.mean()) if len(nonzero) else 0:.4f}m "
                f"nonzero_pct={100*len(nonzero)/(depth.size):.1f}%"
            )

        t0 = time.perf_counter()
        Tcw = self._slam.track_rgbd(rgb, depth, ts)
        self._track_ms.append(1e3 * (time.perf_counter() - t0))
        state = self._slam.get_tracking_state()
        tracking_ok = state == _TRACKING_OK
        if tracking_ok != self._tracking_ok:
            self._tracking_ok = tracking_ok
            if tracking_ok:
                self.logger.info("Tracking OK")
            else:
                self.logger.warning("Tracking lost — withholding poses")

        map_ok = self._check_map() if tracking_ok else False
        level_ok = self._R_level is not None
        Twc = invert_se3(np.asarray(Tcw, dtype=np.float32))
        if level_ok:
            Twc = self._R_level @ Twc
        if tracking_ok and map_ok and level_ok:
            self._pose_buf.write(Twc, ts=ts)

        if self._frame_count % 30 == 0:
            tx, ty, tz = (float(v) for v in Twc[:3, 3])
            self.logger.info(
                f"SLAM state={_STATE_NAMES.get(state, state)} "
                f"pos=({tx:.3f},{ty:.3f},{tz:.3f}) "
                f"dist={math.sqrt(tx * tx + ty * ty + tz * tz):.3f}m "
                f"map={self._map_key} frames={self._frame_count} "
                f"track={np.mean(self._track_ms):.1f}ms"
                + ("" if level_ok else " (fitting floor, poses held)")
            )
            self._track_ms.clear()

        # periodic publish too, otherwise MappingWorker has no points until a loop
        map_changed = self._slam.map_changed()
        if map_changed:
            self._last_map_change = time.monotonic()
            self._loop_pending = True
        publish_pc = map_changed or self._frame_count % self._PC_PUBLISH_INTERVAL == 0

        # map points are only consistent after global BA: signal the loop once
        # GBA is done and the map has been stable for _LOOP_SETTLE_S
        loop_ready = (
            self._loop_pending
            and not (hasattr(self._slam, "is_running_gba") and self._slam.is_running_gba())
            and time.monotonic() - self._last_map_change >= self._LOOP_SETTLE_S
        )
        if (publish_pc or loop_ready) and level_ok and (self._pc_buf is not None or self._kf_buf is not None):
            self._publish_map(ts, map_changed or loop_ready)
        if loop_ready and level_ok:
            self._loop_pending = False
            if self._loop_event is not None:
                self.logger.info("Loop closed and map optimised — signalling loop closure")
                self._loop_event.set()

    def _publish_map(self, ts: float, map_changed: bool) -> None:
        if self._pc_buf is not None:
            all_pts = np.asarray(self._slam.get_all_map_points(), dtype=np.float32)
            if len(all_pts) > 0:
                self._pc_buf.write(self._level_points(all_pts), ts=ts)
                if map_changed:
                    self.logger.info(f"Map changed — published {len(all_pts)} pts")
                else:
                    self.logger.debug(f"Periodic PC publish: {len(all_pts)} pts")
        if self._kf_buf is not None:
            kf = np.asarray(self._slam.get_all_keyframe_poses(), dtype=np.float32)
            if len(kf) > 0:
                self._kf_buf.write(self._level_points(kf[:, :3, 3]), ts=ts)

    def teardown(self) -> None:
        if self._slam is not None:
            self._slam.shutdown()
            self.logger.info("ORB_SLAM3 shut down")
        else:
            self.logger.info("SLAMWorker (stub) shut down")
