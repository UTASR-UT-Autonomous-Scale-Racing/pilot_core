"""
SLAMWorker: FrameBuffer -> ORB_SLAM3 RGBD tracking (pybind11 wrapper) ->
PoseBuffer, plus the map point cloud.

PoseBuffer gets camera-to-world (Twc), so pose[:3, 3] is the camera position
in the map frame. ORB-SLAM3 hands back world-to-camera (Tcw) and we invert it
here. If tracking isn't OK we don't publish at all, since the pose ORB-SLAM3
returns then is identity or stale.

Two phases. While mapping it's TrackRGBD every frame, point cloud published
when the map changes, loop closure signalled. Once driving we switch to
localization mode and only use TrackRGBD for the live pose.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, PoseBuffer

# optional, only needed if a point-cloud buffer is passed in
try:
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None  # type: ignore[misc]


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


class SLAMWorker(BaseWorker):
    """Visual SLAM worker (ORB_SLAM3 through pybind11).

    Config kwargs
    -------------
    input_buffer : FrameBuffer
        RGB + depth in.
    output_buffer : PoseBuffer
        Twc out.
    pointcloud_buffer : PointCloudBuffer, optional
        3D map points.
    vocab_path : str
        ORBvoc.txt
    settings_path : str
        ORB_SLAM3 camera settings YAML.
    loop_event : multiprocessing.Event, optional
        Set when ORB-SLAM3 reports a map change (loop closure or global BA).
    localize_event : multiprocessing.Event, optional
        When set we switch ORB-SLAM3 into localization-only mode.
    warmup_frames : int
        Frames to drop at startup while stereo depth settles. Default 15.

    If orbslam3_python isn't installed this runs as a stub: identity poses,
    no point cloud.
    """

    def __init__(self, name: str = "slam", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: Optional[Any] = None
        self._slam = None
        self._last_seq = 0
        self._frame_count = 0
        self._loop_event: Optional[mp.Event] = None
        self._localize_event: Optional[mp.Event] = None
        self._localization_activated = False
        self._tracking_ok = True
        self._PC_PUBLISH_INTERVAL = 30  # also publish point cloud every N frames
        self._warmup_frames = 0

    def setup(self) -> None:
        self._frame_buf = self.config["input_buffer"]
        self._pose_buf = self.config["output_buffer"]
        self._pc_buf = self.config.get("pointcloud_buffer")
        self._loop_event = self.config.get("loop_event")
        self._localize_event = self.config.get("localize_event")
        self._warmup_frames = int(self.config.get("warmup_frames", 15))

        vocab_path = self.config.get(
            "vocab_path", "ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        settings_path = self.config.get(
            "settings_path", "orbslam_oakd_rgbd.yaml"
        )

        try:
            import orbslam3_python as _slam_mod

            self._slam = _slam_mod.System(
                vocab_path, settings_path, _slam_mod.RGBD
            )
            self.logger.info(
                f"ORB_SLAM3 initialised (vocab={vocab_path}, "
                f"settings={settings_path})"
            )
        except ImportError:
            self.logger.warning(
                "orbslam3_python not available — running SLAM stub "
                "(identity poses, no point cloud)"
            )
            self._slam = None

    def step(self) -> None:
        # mapping done -> localization only
        if (
            not self._localization_activated
            and self._localize_event is not None
            and self._localize_event.is_set()
        ):
            if self._slam is not None:
                self._slam.activate_localization_mode()
                self.logger.info("Switched to localization-only mode")
            self._localization_activated = True

        result = self._frame_buf.read_latest()
        if result is None:
            time.sleep(0.01)
            return

        header, rgb, depth = result
        if header.seq_no <= self._last_seq:
            time.sleep(0.005)
            return
        self._last_seq = header.seq_no

        self._frame_count += 1

        # Drop the warm-up frames. Right after the OAK-D starts the stereo
        # depth is sparse, and ORB-SLAM3 would initialise a thin map, lose it,
        # and keep resetting.
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

        if self._slam is not None:
            Tcw = self._slam.track_rgbd(rgb, depth, header.produced_ts)
            state = self._slam.get_tracking_state()
            tracking_ok = state == _TRACKING_OK
            if tracking_ok != self._tracking_ok:
                self._tracking_ok = tracking_ok
                if tracking_ok:
                    self.logger.info("Tracking OK")
                else:
                    self.logger.warning("Tracking lost — withholding poses")
            Twc = invert_se3(np.asarray(Tcw, dtype=np.float32))
            if tracking_ok:
                self._pose_buf.write(Twc)

            if self._frame_count % 30 == 0:
                tx, ty, tz = (float(v) for v in Twc[:3, 3])
                all_pts = np.asarray(self._slam.get_all_map_points(), dtype=np.float32)
                if len(all_pts) > 0:
                    lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
                    pt_range = (
                        f"X=[{lo[0]:.3f},{hi[0]:.3f}] Y=[{lo[1]:.3f},{hi[1]:.3f}] "
                        f"Z=[{lo[2]:.3f},{hi[2]:.3f}]"
                    )
                else:
                    pt_range = "no pts"
                self.logger.info(
                    f"SLAM state={_STATE_NAMES.get(state, state)} "
                    f"pos=({tx:.3f},{ty:.3f},{tz:.3f}) "
                    f"dist={math.sqrt(tx * tx + ty * ty + tz * tz):.3f}m "
                    f"map_pts={len(all_pts)} frames={self._frame_count}"
                )
                self.logger.debug(f"  map_pt_range: {pt_range}")

            # Publish on map change and also every N frames, otherwise
            # MappingWorker has no wall points until a loop is detected.
            map_changed = self._slam.map_changed()
            publish_pc = map_changed or (
                self._pc_buf is not None
                and self._frame_count % self._PC_PUBLISH_INTERVAL == 0
            )
            if publish_pc and self._pc_buf is not None:
                all_pts = np.asarray(self._slam.get_all_map_points(), dtype=np.float32)
                if len(all_pts) > 0:
                    self._pc_buf.write(all_pts)
                    if map_changed:
                        self.logger.info(f"Map changed — published {len(all_pts)} pts")
                    else:
                        self.logger.debug(f"Periodic PC publish: {len(all_pts)} pts")

            # map_changed() is true once per loop closure / global BA
            if map_changed and self._loop_event is not None:
                self._loop_event.set()
        else:
            # stub, no orbslam3_python
            identity = np.eye(4, dtype=np.float32)
            self._pose_buf.write(identity)

    def teardown(self) -> None:
        if self._slam is not None:
            self._slam.shutdown()
            self.logger.info("ORB_SLAM3 shut down")
        else:
            self.logger.info("SLAMWorker (stub) shut down")
