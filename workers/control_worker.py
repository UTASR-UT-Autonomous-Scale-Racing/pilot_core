"""
ControlWorker — operates in one of three modes:

1. **Neural mode** (default): reads segmentation masks + IMU, runs a learned
   control model, and writes normalised steering/throttle into ControlBuffer.

2. **PID mode** (``mode="pid"``): reads live pose from PoseBuffer and track
   boundaries from TrackBoundaryBuffer, runs the TrackPIDController, and
   writes the result to ControlBuffer.  ``mode="ftg"`` runs the
   Follow-the-Gap controller on the same inputs.

3. **LIDAR wall mode** (``mode="lidar_pid"``): reads the live LIDAR scan each
   step, computes lateral centering error from the forward-facing sector, and
   runs a PID to steer between the walls.  No mapping or global boundary needed.

Output contract: every value written to ControlBuffer is normalised to
``[0, 1]`` with **0.5 = neutral** on both channels, matching
``pilot_controller.Controller`` (linear over each channel's PWM [MIN, MAX])
and the simulator bridge: steering 0 = full left, 1 = full right;
throttle < 0.5 = brake / reverse, > 0.5 = forward.  Controllers work in
signed ``[-1, 1]`` space with **+steering = left** and are converted here with
:func:`signed_to_unit`.
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import numpy as np

from .base_worker import BaseWorker
from .bev import pose_to_bev
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


def signed_to_unit(steering: float, throttle: float) -> tuple[float, float]:
    """Convert signed controller output to the ControlBuffer ``[0, 1]`` contract.

    Signed convention (all track controllers and RL policies): steering
    **+1 = full left** (ROS / REP-103, counter-clockwise), throttle +1 = full
    forward, -1 = full brake / reverse.

    ControlBuffer follows the car's PWM convention (adding to neutral steers
    right): steering 0 = full left, 1 = full right, so steering is negated
    here.  Both channels map 0 → 0.5 (neutral).
    """
    steer_unit = float(np.clip((1.0 - steering) / 2.0, 0.0, 1.0))
    throttle_unit = float(np.clip((throttle + 1.0) / 2.0, 0.0, 1.0))
    return steer_unit, throttle_unit


# Modes driven by the state estimate + mapped track boundaries
TRACK_MODES = ("pure_pursuit", "stanley", "lqr", "mpc", "policy")


def unit_to_signed(steering: float, throttle: float) -> tuple[float, float]:
    """Inverse of :func:`signed_to_unit`."""
    return 1.0 - 2.0 * steering, 2.0 * throttle - 1.0


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
        self._model_takes_timestamp = False
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

        modes = self._modes()
        if self._mode not in modes:
            raise ValueError(f"Unknown control mode {self._mode!r}; options: {sorted(modes)}")
        setup, self._step_fn = modes[self._mode]
        setup()

    def _modes(self) -> Dict[str, tuple]:
        """control_mode → (setup, step)."""
        return {
            "neural": (self._setup_neural, self._step_neural),
            "pid": (self._setup_pid, self._step_pid),
            "ftg": (self._setup_ftg, self._step_pid),
            "lidar_pid": (self._setup_lidar_pid, self._step_lidar_pid),
            **{m: (self._setup_track, self._step_track) for m in TRACK_MODES},
            "depth_ftg": (self._setup_depth_ftg, self._step_depth_ftg),
        }

    def _setup_neural(self) -> None:
        bufs: Dict[str, Any] = self.config["input_buffers"]
        self._seg_buf = bufs["segments"]
        self._imu_buf = bufs["imu"]
        self._pose_buf = bufs.get("pose")

        model_config_path = self.config["model_config"]
        from utils import load_model_from_config

        self._model, class_name = load_model_from_config(model_config_path)
        # Some pilot_controller models take (state, mask, timestamp), others
        # only (state, mask) — detect once rather than per frame.
        params = inspect.signature(self._model.get_action).parameters
        self._model_takes_timestamp = "timestamp" in params
        self.logger.info(f"Loaded control model: {class_name}")

        device = self.config.get("device", "cpu")
        self.logger.info(f"ControlWorker ready — neural mode (device={device})")

    def _setup_pid(self) -> None:
        from .track_pid_controller import TrackPIDController

        self._pose_buf = self.config["pose_buffer"]
        self._boundary_buf = self.config.get("boundary_buffer")
        self._mapping_done_event = self.config.get("mapping_done_event")
        # Ground-plane / forward axes — see pose_to_bev(); must match MappingWorker
        self._bev_axes = tuple(self.config.get("bev_axes", [0, 2]))
        self._forward_axis = int(self.config.get("forward_axis", 2))

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
        self._forward_axis = int(self.config.get("forward_axis", 2))

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
        self._step_fn()

    # --- Track-following modes (pure_pursuit, stanley) ---

    def _setup_track(self) -> None:
        from control.types import VehicleParams

        self._state_buf = self.config["state_buffer"]
        self._boundary_buf = self.config["boundary_buffer"]
        self._mapping_done_event = self.config.get("mapping_done_event")
        self._vehicle = VehicleParams.load(self.config.get("vehicle_params", "vehicle_params.yaml"))
        self._track_cfg = self.config.get("track_config", {})
        self._follower = None
        self._last_state_seq = 0
        self._last_state_time = time.time()
        # Optional map-free fallback while the state estimate is stale
        self._fallback = None
        if self._track_cfg.get("fallback") == "depth_ftg":
            self._fallback = self._make_depth_follower()
            self._fallback_after = float(self._track_cfg.get("fallback_after", 0.3))
            self._fallback_active = False
        self.logger.info(
            f"ControlWorker ready — {self._mode} mode "
            f"(wheelbase={self._vehicle.wheelbase} m, max_steer={self._vehicle.max_steering_angle} rad)"
        )

    def _build_follower(self, outer: np.ndarray, inner: np.ndarray, state) -> None:
        from control.controllers import build_track_follower
        from control.track import TrackModel

        cfg = self._track_cfg
        if self._mode == "policy":
            from control.policy import PolicyFollower

            model = cfg.get("policy", {}).get("model", "policy.onnx")
            self._follower = PolicyFollower(model, outer, inner)
            self.logger.info(f"Policy loaded: {model} (obs_dim={self._follower.meta.obs_dim})")
            return
        track = TrackModel(outer, inner, spacing=cfg.get("spacing", 0.05))
        reversed_ = track.align_direction(state.x, state.y, state.yaw)
        self._follower = build_track_follower(
            self._mode, track, self._vehicle, cfg.get(self._mode, {}), cfg.get("speed", {})
        )
        self.logger.info(
            f"Track model built: length={track.length:.1f} m, "
            f"min half-width={track.half_width.min():.2f} m"
            + (" (direction reversed to match heading)" if reversed_ else "")
        )

    def _step_track(self) -> None:
        if self._mapping_done_event is not None and not self._mapping_done_event.is_set():
            time.sleep(0.1)
            return

        result = self._state_buf.read_latest()
        fresh = result is not None and result[0].seq_no > self._last_state_seq
        if not fresh:
            if self._fallback is not None and time.time() - self._last_state_time > self._fallback_after:
                self._step_fallback()
            else:
                time.sleep(0.002)
            return
        header, state = result
        self._last_state_seq = header.seq_no
        self._last_state_time = time.time()
        if self._fallback is not None and self._fallback_active:
            self._fallback_active = False
            self.logger.info("State estimate back — leaving depth_ftg fallback")

        if self._follower is None:
            bnd = self._boundary_buf.read_latest()
            if bnd is None:
                time.sleep(0.05)
                return
            _, outer, inner = bnd
            self._build_follower(outer, inner, state)

        cmd = self._follower.step(state)
        steer_unit, throttle_unit = signed_to_unit(cmd.steering, cmd.throttle)
        self._ctrl_buf.write(steer_unit, throttle_unit)

        if self._last_state_seq % 50 == 0 and self._mode != "policy":
            pr = self._follower.last_projection
            self.logger.debug(
                f"[{self._mode}] s={pr.s:.1f} e={pr.lateral_error:+.3f} v={state.v:.2f} "
                f"steer={cmd.steering:+.3f} throttle={cmd.throttle:+.3f}"
            )

    # --- Map-free depth Follow-the-Gap (mode, or fallback for track modes) ---

    def _make_depth_follower(self):
        from control.depth_ftg import DepthFTGConfig, DepthGapFollower, Intrinsics
        from control.types import VehicleParams

        self._frame_buf = self.config["frame_buffer"]
        self._last_frame_seq = 0
        K = Intrinsics.from_orbslam_settings(self.config.get("camera_settings", "orbslam_oakd_rgbd.yaml"))
        vehicle = VehicleParams.load(self.config.get("vehicle_params", "vehicle_params.yaml"))
        cfg = DepthFTGConfig(**self.config.get("depth_ftg_config", {}))
        return DepthGapFollower(vehicle, K, cfg)

    def _setup_depth_ftg(self) -> None:
        self._depth_follower = self._make_depth_follower()
        self.logger.info("ControlWorker ready — depth_ftg mode (map-free)")

    def _depth_command(self, follower) -> bool:
        """Run ``follower`` on the newest depth frame; False if none is new."""
        result = self._frame_buf.read_latest()
        if result is None or result[0].seq_no <= self._last_frame_seq:
            return False
        header, _, depth = result
        self._last_frame_seq = header.seq_no
        cmd = follower.step(depth)
        self._ctrl_buf.write(*signed_to_unit(cmd.steering, cmd.throttle))
        return True

    def _step_depth_ftg(self) -> None:
        if not self._depth_command(self._depth_follower):
            time.sleep(0.005)

    def _step_fallback(self) -> None:
        if not self._fallback_active:
            self._fallback_active = True
            self.logger.warning("State estimate stale — driving on depth_ftg fallback")
        if not self._depth_command(self._fallback):
            time.sleep(0.005)

    # --- LIDAR wall-centering PID mode ---

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

        steer_raw, throttle_raw = self._lidar_wall_ctrl.step(ranges)
        steer_unit, throttle_unit = signed_to_unit(steer_raw, throttle_raw)
        self._ctrl_buf.write(steer_unit, throttle_unit)

        if self._last_lidar_seq % 30 == 0:
            self.logger.debug(
                f"[lidar_pid] steer={steer_raw:+.3f} throttle={throttle_raw:+.3f} "
                f"-> buffer ({steer_unit:.3f}, {throttle_unit:.3f})"
            )

    # --- Neural mode ---

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
        if self._model_takes_timestamp:
            result = self._model.get_action(
                state, mask, timestamp=seg_header.produced_ts
            )
        else:
            result = self._model.get_action(state, mask)

        if result is not None:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                steering, throttle = float(result[0]), float(result[1])
            else:
                steering, throttle = 0.5, 0.0
            self._ctrl_buf.write(
                float(np.clip(steering, 0.0, 1.0)),
                float(np.clip(throttle, 0.0, 1.0)),
            )

    # --- PID mode ---

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

        current_bev, heading = pose_to_bev(pose, self._bev_axes, self._forward_axis)

        # PID step — returns raw [-1, 1]; normalize to [0, 1] for ControlBuffer
        steer_raw, throttle_raw = self._pid.step(
            current_bev, heading, self._outer_line, self._inner_line
        )

        # Controllers work in signed space; convert to the ControlBuffer contract
        steer_unit, throttle_unit = signed_to_unit(steer_raw, throttle_raw)
        self._ctrl_buf.write(steer_unit, throttle_unit)

        if self._last_pose_seq % 30 == 0:
            self.logger.debug(
                f"[{self._mode}] pos={current_bev} steer={steer_raw:+.3f} "
                f"throttle={throttle_raw:+.3f} -> buffer ({steer_unit:.3f}, {throttle_unit:.3f})"
            )

    def teardown(self) -> None:
        self.logger.info(f"ControlWorker ({self._mode} mode) shutting down")
