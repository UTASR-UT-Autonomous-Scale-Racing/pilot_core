"""
ControlWorker — operates in one of three modes:

1. **Neural mode** (default): reads segmentation masks + IMU, runs a learned
   control model, and writes normalised steering/throttle into ControlBuffer.

2. **PID mode** (``mode="pid"``): reads live pose from PoseBuffer and track
   boundaries from TrackBoundaryBuffer, runs the TrackPIDController, and
   writes normalised (steering, throttle) to ControlBuffer.

3. **LIDAR wall mode** (``mode="lidar_pid"``): reads the live LIDAR scan each
   step, computes lateral centering error from the forward-facing sector, and
   runs a PID to steer between the walls.  No mapping or global boundary needed.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import ControlBuffer, IMUBuffer, PoseBuffer, SegmentationBuffer

# Optional — only needed in boundary-PID mode
try:
    from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer
except ImportError:
    TrackBoundaryBuffer = None  # type: ignore[misc]

# Optional — only needed in lidar_pid mode
try:
    from shared_buffer import LidarBuffer
except ImportError:
    LidarBuffer = None  # type: ignore[misc,assignment]


class ControlWorker(BaseWorker):
    """Worker process for control-policy inference (neural or PID).

    Config kwargs
    -------------
    mode : str
        ``"neural"`` (default) or ``"pid"``.

    Neural-mode kwargs:
        input_buffers : dict[str, buffer]
            Must contain ``"segments"`` and ``"imu"``.  May contain ``"pose"``.
        output_buffer : ControlBuffer
        model_config : str
        device : str

    PID-mode kwargs:
        pose_buffer : PoseBuffer
        boundary_buffer : TrackBoundaryBuffer
        output_buffer : ControlBuffer
        mapping_done_event : mp.Event
        pid_config : dict  (kp_steer, ki_steer, kd_steer, target_throttle, noise_std)
    """

    def __init__(self, name: str = "control", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._mode: str = "neural"
        # Neural-mode state
        self._seg_buf: SegmentationBuffer | None = None
        self._imu_buf: IMUBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._ctrl_buf: ControlBuffer | None = None
        self._model = None
        self._last_seg_seq = 0
        # Boundary-PID-mode state
        self._boundary_buf: Optional[Any] = None
        self._pid = None
        self._mapping_done_event: Optional[mp.Event] = None
        self._outer_line: Optional[np.ndarray] = None
        self._inner_line: Optional[np.ndarray] = None
        self._last_pose_seq = 0
        # LIDAR-wall-mode state
        self._lidar_buf: Optional[Any] = None
        self._lidar_wall_ctrl = None
        self._last_lidar_seq = 0

    def setup(self) -> None:
        self._mode = self.config.get("mode", "neural")
        self._ctrl_buf = self.config["output_buffer"]

        if self._mode == "pid":
            self._setup_pid()
        elif self._mode == "ftg":
            self._setup_ftg()
        elif self._mode == "lidar_pid":
            self._setup_lidar_pid()
        else:
            self._setup_neural()

    def _setup_neural(self) -> None:
        bufs: Dict[str, Any] = self.config["input_buffers"]
        self._seg_buf = bufs["segments"]
        self._imu_buf = bufs["imu"]
        self._pose_buf = bufs.get("pose")

        model_config_path = self.config["model_config"]
        from utils import load_model_from_config

        self._model, class_name = load_model_from_config(model_config_path)
        self.logger.info(f"Loaded control model: {class_name}")

        device = self.config.get("device", "cpu")
        self.logger.info(f"ControlWorker ready — neural mode (device={device})")

    def _setup_pid(self) -> None:
        from .track_pid_controller import TrackPIDController

        self._pose_buf = self.config["pose_buffer"]
        self._boundary_buf = self.config.get("boundary_buffer")
        self._mapping_done_event = self.config.get("mapping_done_event")
        # BEV axes — must match MappingWorker bev_axes
        # SLAM camera convention (Y-down, Z-fwd): [0, 2]
        # Simulator / ROS robot frame (Z-up, X-fwd): [0, 1]
        self._bev_axes = tuple(self.config.get("bev_axes", [0, 2]))

        pid_cfg = self.config.get("pid_config", {})
        self._pid = TrackPIDController(
            kp_steer=pid_cfg.get("kp_steer", 1.0),
            ki_steer=pid_cfg.get("ki_steer", 0.0),
            kd_steer=pid_cfg.get("kd_steer", 0.1),
            target_throttle=pid_cfg.get("target_throttle", 0.3),
            noise_std=pid_cfg.get("noise_std", 0.05),
        )
        self.logger.info(f"ControlWorker ready — PID mode (bev_axes={self._bev_axes})")

    def _setup_ftg(self) -> None:
        from .track_ftg_controller import TrackFTGController

        self._pose_buf = self.config["pose_buffer"]
        self._boundary_buf = self.config.get("boundary_buffer")
        self._mapping_done_event = self.config.get("mapping_done_event")
        self._bev_axes = tuple(self.config.get("bev_axes", [0, 2]))

        ftg_cfg = self.config.get("ftg_config", {})
        self._pid = TrackFTGController(
            num_beams=ftg_cfg.get("num_beams", 360),
            fov_deg=ftg_cfg.get("fov_deg", 270.0),
            max_range=ftg_cfg.get("max_range", 5.0),
            bubble_radius_deg=ftg_cfg.get("bubble_radius_deg", 30.0),
            smooth_window=ftg_cfg.get("smooth_window", 5),
            target_throttle=ftg_cfg.get("target_throttle", 0.3),
            throttle_corner_scale=ftg_cfg.get("throttle_corner_scale", 0.5),
            noise_std=ftg_cfg.get("noise_std", 0.05),
            gap_strategy=ftg_cfg.get("gap_strategy", "deepest"),
        )
        self.logger.info(f"ControlWorker ready — FTG mode (bev_axes={self._bev_axes})")

    def _setup_lidar_pid(self) -> None:
        from .lidar_wall_controller import LidarWallController

        self._lidar_buf = self.config["lidar_buffer"]
        pid_cfg = self.config.get("pid_config", {})
        self._lidar_wall_ctrl = LidarWallController(
            kp=pid_cfg.get("kp_steer", 1.0),
            ki=pid_cfg.get("ki_steer", 0.0),
            kd=pid_cfg.get("kd_steer", 0.1),
            target_throttle=pid_cfg.get("target_throttle", 0.4),
            sector_deg=pid_cfg.get("sector_deg", 60.0),
            range_min=pid_cfg.get("range_min", 0.1),
            range_max=pid_cfg.get("range_max", 9.5),
            slow_corner_factor=pid_cfg.get("slow_corner_factor", 0.6),
        )
        self.logger.info("ControlWorker ready — LIDAR wall-centering PID mode")

    def step(self) -> None:
        if self._mode in ("pid", "ftg"):
            self._step_pid()
        elif self._mode == "lidar_pid":
            self._step_lidar_pid()
        else:
            self._step_neural()

    #  LIDAR wall-centering PID mode

    def _step_lidar_pid(self) -> None:
        result = self._lidar_buf.read_latest()
        if result is None:
            time.sleep(0.01)
            return

        header, ranges = result
        if header.seq_no <= self._last_lidar_seq:
            time.sleep(0.005)
            return
        self._last_lidar_seq = header.seq_no

        # Returns raw [-1, 1]; normalize to [0, 1] for ControlBuffer
        steer_raw, throttle_raw = self._lidar_wall_ctrl.step(ranges)
        steer_norm = float(np.clip((steer_raw + 1.0) / 2.0, 0.0, 1.0))
        throttle_norm = float(np.clip((throttle_raw + 1.0) / 2.0, 0.0, 1.0))
        print(f"[lidar_pid] steer={steer_raw:+.3f} ({steer_norm:.3f})  "
              f"throttle={throttle_raw:+.3f} ({throttle_norm:.3f})", flush=True)
        self._ctrl_buf.write(steer_norm, throttle_norm)

    #  Neural mode ─

    def _step_neural(self) -> None:
        seg_result = self._seg_buf.read_latest()
        if seg_result is None:
            time.sleep(0.001)
            return

        seg_header, mask = seg_result
        if seg_header.seq_no <= self._last_seg_seq:
            time.sleep(0.001)
            return
        self._last_seg_seq = seg_header.seq_no

        imu_result = self._imu_buf.read_latest()
        imu_vector = (
            imu_result[1] if imu_result is not None else np.zeros(9, dtype=np.float32)
        )

        state = imu_vector.tolist()
        result = self._model.get_action(state, mask)

        if result is not None:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                steering, throttle = float(result[0]), float(result[1])
            else:
                steering, throttle = 0.5, 0.0
            self._ctrl_buf.write(steering, throttle)

    #  PID mode 

    def _step_pid(self) -> None:
        # Wait for mapping to complete
        if (
            self._mapping_done_event is not None
            and not self._mapping_done_event.is_set()
        ):
            time.sleep(0.1)
            return

        # Read boundaries once (they don't change after extraction)
        if self._outer_line is None and self._boundary_buf is not None:
            bnd_result = self._boundary_buf.read_latest()
            if bnd_result is not None:
                _, self._outer_line, self._inner_line = bnd_result
                self.logger.info(
                    f"Loaded boundaries: outer={len(self._outer_line)}, "
                    f"inner={len(self._inner_line)}"
                )

        if self._outer_line is None or self._inner_line is None:
            time.sleep(0.05)
            return

        # Read latest pose
        pose_result = self._pose_buf.read_latest()
        if pose_result is None:
            time.sleep(0.005)
            return

        pose_header, pose, _, _ = pose_result
        if pose_header.seq_no <= self._last_pose_seq:
            time.sleep(0.005)
            return
        self._last_pose_seq = pose_header.seq_no

        # Extract BEV position and heading from 4×4 pose using configured axes.
        # bev_axes=(a0, a1): position = (t[a0], t[a1])
        # Forward direction = vehicle X axis (col 0 of rotation, ROS convention).
        a0, a1 = self._bev_axes
        current_bev = np.array([pose[a0, 3], pose[a1, 3]], dtype=np.float32)
        heading = np.arctan2(pose[a1, 0], pose[a0, 0])

        # PID step — returns raw [-1, 1]; normalize to [0, 1] for ControlBuffer
        steer_raw, throttle_raw = self._pid.step(
            current_bev, heading, self._outer_line, self._inner_line
        )
        steer_norm = float(np.clip((steer_raw + 1.0) / 2.0, 0.0, 1.0))
        throttle_norm = float(np.clip((throttle_raw + 1.0) / 2.0, 0.0, 1.0))
        print(f"[{self._mode}] pos={current_bev}  steer={steer_raw:+.3f} ({steer_norm:.3f})  "
              f"throttle={throttle_raw:+.3f} ({throttle_norm:.3f})", flush=True)
        self._ctrl_buf.write(steer_norm, throttle_norm)

    def teardown(self) -> None:
        self.logger.info(f"ControlWorker ({self._mode} mode) shutting down")
