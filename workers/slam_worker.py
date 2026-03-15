"""
SLAMWorker — reads frames from FrameBuffer, runs ORB_SLAM3 RGBD tracking
via the pybind11 wrapper, and writes pose + point-cloud data into shared
buffers.

Operates in two phases:
    Phase A (mapping):  TrackRGBD per frame, accumulate trajectory, publish
                        point cloud when map changes, signal loop closure.
    Phase B (driving):  ActivateLocalizationMode, TrackRGBD for live pose only.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, PoseBuffer
from shared_buffer.imu_buffer import IMUBuffer

# Optional imports — the point-cloud buffer may not exist if not enabled
try:
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None  # type: ignore[misc]


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
    imu_buffer : IMUBuffer | None
        Shared IMUBuffer with vectors from OAK-D camera worker.
    use_imu : bool
        If True, run ORB-SLAM3 in IMU_RGBD mode and pass IMU samples to TrackRGBD.
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
        self._imu_buf: IMUBuffer | None = None
        self._use_imu: bool = False
        self._last_imu_seq: int = 0
        self._last_imu_point: np.ndarray | None = None
        self._last_imu_issue: str = ""
        self._imu_wait_start_ts: float | None = None
        self._imu_wait_last_log_ts: float = 0.0
        self._imu_wait_frames: int = 0
        self._imu_valid_samples: int = 0
        self._imu_min_samples_before_tracking: int = 30
        self._init_phase_start_ts: float | None = None
        self._init_phase_last_log_ts: float = 0.0
        self._init_ready_logged: bool = False
        self._slam = None
        self._last_seq = 0
        self._loop_event: Optional[mp.Event] = None
        self._localize_event: Optional[mp.Event] = None
        self._localization_activated = False
        self._pc_publish_every_s: float = 1.0
        self._last_pc_publish_ts: float = 0.0

    def setup(self) -> None:
        self._frame_buf = self.config["input_buffer"]
        self._pose_buf = self.config["output_buffer"]
        self._pc_buf = self.config.get("pointcloud_buffer")
        self._imu_buf = self.config.get("imu_buffer")
        self._use_imu = bool(self.config.get("use_imu", False) and self._imu_buf is not None)
        self._imu_min_samples_before_tracking = int(
            self.config.get("imu_min_samples_before_tracking", 30)
        )
        self._loop_event = self.config.get("loop_event")
        self._localize_event = self.config.get("localize_event")
        self._pc_publish_every_s = float(self.config.get("publish_pointcloud_every_s", 1.0))

        vocab_path = self.config.get(
            "vocab_path", "ORB_SLAM3/Vocabulary/ORBvoc.txt"
        )
        settings_path = self.config.get(
            "settings_path", "oak_d_orbslam_settings.yaml"
        )

        try:
            import orbslam3_python as _slam_mod

            sensor_type = _slam_mod.RGBD
            imu_ready = self._use_imu and self._settings_support_imu(settings_path)
            if imu_ready and hasattr(_slam_mod, "IMU_RGBD"):
                sensor_type = _slam_mod.IMU_RGBD
            elif self._use_imu:
                self.logger.warning(
                    "IMU requested for SLAM, but settings file lacks IMU calibration/noise keys; "
                    "falling back to RGBD mode"
                )

            self._slam = _slam_mod.System(
                vocab_path, settings_path, sensor_type
            )
            self.logger.info(
                f"ORB_SLAM3 initialised (vocab={vocab_path}, "
                f"settings={settings_path})"
            )
            if sensor_type == getattr(_slam_mod, "IMU_RGBD", None):
                self.logger.info("SLAMWorker IMU mode enabled (IMU_RGBD)")
                self._init_phase_start_ts = time.time()
                self._init_phase_last_log_ts = 0.0
                self._init_ready_logged = False
                self.logger.info(
                    "ORB-SLAM3 inertial startup: waiting for visual-inertial initialization; "
                    "transient backend messages like 'not IMU meas' can appear"
                )
        except ImportError:
            self.logger.warning(
                "orbslam3_python not available — running SLAM stub "
                "(identity poses, no point cloud)"
            )
            self._slam = None

    def step(self) -> None:
        #  Check if we should switch to localization-only mode 
        if (
            not self._localization_activated
            and self._localize_event is not None
            and self._localize_event.is_set()
        ):
            if self._slam is not None:
                self._slam.activate_localization_mode()
                self.logger.info("Switched to localization-only mode")
            self._localization_activated = True

        #  Read a frame 
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
            imu_points = self._collect_imu_points(header.produced_ts)
            if self._use_imu and self._imu_valid_samples < self._imu_min_samples_before_tracking:
                now = time.time()
                if now - self._imu_wait_last_log_ts >= 1.0:
                    self.logger.info(
                        "Waiting for IMU warm-up samples before IMU_RGBD tracking "
                        "(have=%d need=%d)",
                        self._imu_valid_samples,
                        self._imu_min_samples_before_tracking,
                    )
                    self._imu_wait_last_log_ts = now
                time.sleep(0.001)
                return
            if self._use_imu and imu_points is None:
                now = time.time()
                if self._imu_wait_start_ts is None:
                    self._imu_wait_start_ts = now
                    self._imu_wait_last_log_ts = 0.0
                    self._imu_wait_frames = 0
                self._imu_wait_frames += 1

                # Be explicit that startup is waiting on inertial init data.
                if now - self._imu_wait_last_log_ts >= 1.0:
                    wait_s = now - self._imu_wait_start_ts
                    reason = self._last_imu_issue or "no imu sample yet"
                    self.logger.info(
                        "Waiting for ORB-SLAM3 inertial initialization: "
                        "no valid IMU sample yet (wait=%.1fs, skipped_frames=%d, reason=%s)",
                        wait_s,
                        self._imu_wait_frames,
                        reason,
                    )
                    self._imu_wait_last_log_ts = now
                # Avoid calling IMU_RGBD tracking with an empty IMU packet set.
                time.sleep(0.001)
                return

            if self._use_imu and self._imu_wait_start_ts is not None:
                wait_s = time.time() - self._imu_wait_start_ts
                self.logger.info(
                    "ORB-SLAM3 inertial initialization ready: received valid IMU data "
                    "after %.1fs (%d skipped frames)",
                    wait_s,
                    self._imu_wait_frames,
                )
                self._imu_wait_start_ts = None
                self._imu_wait_frames = 0
                self._imu_wait_last_log_ts = 0.0

            #  Real SLAM tracking 
            pose_4x4 = self._slam.track_rgbd(rgb, depth, header.produced_ts, imu_points)
            self._pose_buf.write(np.asarray(pose_4x4, dtype=np.float32))
            self._maybe_publish_pointcloud(force=False)

            if self._use_imu and self._init_phase_start_ts is not None:
                now = time.time()
                state = None
                try:
                    if hasattr(self._slam, "get_tracking_state"):
                        state = int(self._slam.get_tracking_state())
                except Exception:
                    state = None

                if state == 2:
                    if not self._init_ready_logged:
                        self.logger.info(
                            "ORB-SLAM3 inertial initialization complete after %.1fs (tracking_state=%s)",
                            now - self._init_phase_start_ts,
                            state,
                        )
                        self._init_ready_logged = True
                elif now - self._init_phase_last_log_ts >= 1.0:
                    self.logger.info(
                        "ORB-SLAM3 inertial initialization in progress (elapsed=%.1fs, tracking_state=%s)",
                        now - self._init_phase_start_ts,
                        state,
                    )
                    self._init_phase_last_log_ts = now

            # Check for map change (loop closure / global BA)
            if self._slam.map_changed():
                self.logger.info("Map changed (loop closure / global BA)")

                # Publish updated point cloud
                if self._pc_buf is not None:
                    self._maybe_publish_pointcloud(force=True)

                # Signal loop closure
                if self._loop_event is not None:
                    self._loop_event.set()
        else:
            #  Stub mode — identity pose ─
            identity = np.eye(4, dtype=np.float32)
            self._pose_buf.write(identity)

    def _settings_support_imu(self, settings_path: str) -> bool:
        """Return True when ORB-SLAM settings include required IMU keys."""
        required_keys = (
            "IMU.NoiseGyro",
            "IMU.NoiseAcc",
            "IMU.GyroWalk",
            "IMU.AccWalk",
        )

        try:
            if not os.path.exists(settings_path):
                return False
            with open(settings_path, "r", encoding="utf-8", errors="ignore") as fp:
                txt = fp.read()
            return all(k in txt for k in required_keys)
        except Exception:
            return False

    def _collect_imu_points(self, frame_ts: float) -> np.ndarray | None:
        """Collect latest IMU sample and format as (N,7) for the pybind wrapper.

        Shape is [ax, ay, az, gx, gy, gz, t]. Returning None tells the wrapper
        to call RGBD tracking without IMU data.
        """
        if not self._use_imu or self._imu_buf is None:
            self._last_imu_issue = "imu disabled"
            return None

        imu_result = self._imu_buf.read_latest()
        if imu_result is None:
            self._last_imu_issue = "imu buffer empty"
            return self._last_imu_point

        imu_header, imu_vec = imu_result
        if imu_header.seq_no <= self._last_imu_seq:
            self._last_imu_issue = "no newer imu seq"
            return self._last_imu_point

        self._last_imu_seq = imu_header.seq_no
        if imu_vec.shape[0] < 16:
            self._last_imu_issue = "imu vector too short"
            return self._last_imu_point

        # Prefer buffer-header timestamp (host epoch) to keep IMU and frame
        # clocks in the same domain.
        imu_ts = float(imu_header.produced_ts)

        raw_imu_ts = float(imu_vec[15])
        # Normalize timestamps when producers provide microseconds.
        if raw_imu_ts > 1e12:
            raw_imu_ts /= 1e6
        # If vector timestamp is already in the same time base, keep it.
        if abs(raw_imu_ts - frame_ts) < 5.0:
            imu_ts = raw_imu_ts

        # Avoid feeding stale IMU samples from older frames.
        if imu_ts < frame_ts - 1.0:
            self._last_imu_issue = "imu sample stale"
            return self._last_imu_point

        imu_np = np.array(
            [[
                float(imu_vec[4]), float(imu_vec[5]), float(imu_vec[6]),
                float(imu_vec[7]), float(imu_vec[8]), float(imu_vec[9]),
                float(imu_ts),
            ]],
            dtype=np.float64,
        )
        ax, ay, az = float(imu_np[0, 0]), float(imu_np[0, 1]), float(imu_np[0, 2])
        if (ax * ax + ay * ay + az * az) ** 0.5 < 0.1:
            self._last_imu_issue = "acc norm too low"
            return self._last_imu_point

        self._last_imu_issue = ""
        self._last_imu_point = imu_np
        self._imu_valid_samples += 1
        return imu_np

    def _maybe_publish_pointcloud(self, force: bool = False) -> None:
        """Publish map points at a controlled cadence to avoid empty recordings."""
        if self._pc_buf is None or self._slam is None:
            return

        now = time.time()
        if not force and (now - self._last_pc_publish_ts) < self._pc_publish_every_s:
            return

        try:
            all_pts = self._slam.get_all_map_points()
            all_pts = np.asarray(all_pts, dtype=np.float32)
            if all_pts.size == 0:
                return
            self._pc_buf.write(all_pts)
            self._last_pc_publish_ts = now
            if force:
                self.logger.info(f"Published point cloud (map change): {len(all_pts)} points")
            else:
                self.logger.debug(f"Published point cloud (periodic): {len(all_pts)} points")
        except Exception:
            self.logger.exception("Pointcloud publish failed")

    def teardown(self) -> None:
        if self._slam is not None:
            self._slam.shutdown()
            self.logger.info("ORB_SLAM3 shut down")
        else:
            self.logger.info("SLAMWorker (stub) shut down")
