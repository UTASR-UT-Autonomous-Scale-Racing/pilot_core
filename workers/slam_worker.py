"""
SLAMWorker — reads frames from FrameBuffer, runs ORB_SLAM3 RGBD tracking
via the pybind11 wrapper, and writes pose + point-cloud data into shared
buffers.

Pose contract: PoseBuffer carries the **camera-to-world** transform (Twc),
so ``pose[:3, 3]`` is the camera position in the map frame.  ORB-SLAM3
returns world-to-camera (Tcw); it is inverted here.  Frames where tracking
is not OK are not published (ORB-SLAM3 returns an identity/stale pose).

Operates in two phases:
    Phase A (mapping):  TrackRGBD per frame, accumulate trajectory, publish
                        point cloud when map changes, signal loop closure.
    Phase B (driving):  ActivateLocalizationMode, TrackRGBD for live pose only.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, PoseBuffer

# Optional imports — the point-cloud buffer may not exist if not enabled
try:
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None  # type: ignore[misc]


# ORB_SLAM3::Tracking::eTrackingState
_TRACKING_OK = 2


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a 4×4 rigid transform."""
    R, t = T[:3, :3], T[:3, 3]
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


class SLAMWorker(BaseWorker):
    """Worker process for visual SLAM (ORB_SLAM3 via pybind11).

    Config kwargs
    -------------
    input_buffer : FrameBuffer
        Shared FrameBuffer to read RGB+Depth from.
    output_buffer : PoseBuffer
        Shared PoseBuffer to write pose matrices into.
    pointcloud_buffer : PointCloudBuffer | None
        Shared PointCloudBuffer for the 3D map (optional).
    vocab_path : str
        Path to ORB vocabulary file (``ORBvoc.txt``).
    settings_path : str
        Path to camera settings YAML for ORB_SLAM3.
    loop_event : multiprocessing.Event | None
        Event to set when a loop closure is detected.
    localize_event : multiprocessing.Event | None
        Event that, once set, switches to localization-only mode.
    """

    def __init__(self, name: str = "slam", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: Optional[Any] = None
        self._slam = None
        self._last_seq = 0
        self._loop_event: Optional[mp.Event] = None
        self._localize_event: Optional[mp.Event] = None
        self._localization_activated = False
        self._tracking_ok = True

    def setup(self) -> None:
        self._frame_buf = self.config["input_buffer"]
        self._pose_buf = self.config["output_buffer"]
        self._pc_buf = self.config.get("pointcloud_buffer")
        self._loop_event = self.config.get("loop_event")
        self._localize_event = self.config.get("localize_event")

        vocab_path = self.config.get(
            "vocab_path", "ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        settings_path = self.config.get(
            "settings_path", "oak_d_orbslam_settings.yaml"
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
        # --- Check if we should switch to localization-only mode ---
        if (
            not self._localization_activated
            and self._localize_event is not None
            and self._localize_event.is_set()
        ):
            if self._slam is not None:
                self._slam.activate_localization_mode()
                self.logger.info("Switched to localization-only mode")
            self._localization_activated = True

        # --- Read a frame ---
        result = self._frame_buf.read_latest()
        if result is None:
            time.sleep(0.01)
            return

        header, rgb, depth = result
        if header.seq_no <= self._last_seq:
            time.sleep(0.005)
            return
        self._last_seq = header.seq_no

        if self._slam is not None:
            # --- Real SLAM tracking ---
            Tcw = self._slam.track_rgbd(rgb, depth, header.produced_ts)
            tracking_ok = self._slam.get_tracking_state() == _TRACKING_OK
            if tracking_ok != self._tracking_ok:
                self._tracking_ok = tracking_ok
                if tracking_ok:
                    self.logger.info("Tracking OK")
                else:
                    self.logger.warning("Tracking lost — withholding poses")
            if tracking_ok:
                self._pose_buf.write(invert_se3(np.asarray(Tcw, dtype=np.float32)))

            # Check for map change (loop closure / global BA)
            if self._slam.map_changed():
                self.logger.info("Map changed (loop closure / global BA)")

                # Publish updated point cloud
                if self._pc_buf is not None:
                    all_pts = self._slam.get_all_map_points()
                    all_pts = np.asarray(all_pts, dtype=np.float32)
                    self._pc_buf.write(all_pts)
                    self.logger.info(
                        f"Published point cloud: {len(all_pts)} points"
                    )

                # Signal loop closure
                if self._loop_event is not None:
                    self._loop_event.set()
        else:
            # --- Stub mode — identity pose ---
            identity = np.eye(4, dtype=np.float32)
            self._pose_buf.write(identity)

    def teardown(self) -> None:
        if self._slam is not None:
            self._slam.shutdown()
            self.logger.info("ORB_SLAM3 shut down")
        else:
            self.logger.info("SLAMWorker (stub) shut down")
