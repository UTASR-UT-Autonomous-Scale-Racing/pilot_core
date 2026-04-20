"""
ControlWorker — runs the configured control mode and writes steering /
throttle into ControlBuffer.

Modes (``mode=``):

* Track following on the state estimate + mapped boundaries:
  ``pure_pursuit`` (default), ``stanley``, ``lqr``, ``mpc``, ``policy``
  (see :mod:`control`).  Optional ``depth_ftg`` fallback while the state
  estimate is stale.
* Map-free, from depth frames: ``depth_ftg`` (Follow-the-Gap), ``vae``
  (learned VAE control model) and ``bc`` (behaviour-cloning CNN).
* Baseline controllers: ``pid`` and ``ftg`` (on raw PoseBuffer poses +
  boundaries), ``lidar_pid`` (LiDAR wall centering, simulator only).

Output contract: every value written to ControlBuffer is normalised to
``[0, 1]`` with **0.5 = neutral** on both channels, matching
:class:`control.pwm.PWMMapper` (linear over each channel's PWM [min, max])
and the simulator bridge: steering 0 = full left, 1 = full right;
throttle < 0.5 = backward, > 0.5 = forward.  Controllers work in signed
``[-1, 1]`` space with **+steering = left** and are converted here with
:func:`signed_to_unit`.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import numpy as np

from .base_worker import BaseWorker
from .bev import pose_to_bev
from shared_buffer import ControlBuffer, PoseBuffer

# Optional — only needed in the pid / ftg modes
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
    """Worker process for vehicle control.

    Common kwargs: ``mode``, ``output_buffer`` (ControlBuffer).

    Track modes: ``state_buffer``, ``boundary_buffer``, ``mapping_done_event``,
    ``track_config`` (config ``track_control``), ``vehicle_params``; for the
    fallback also ``frame_buffer`` / ``camera_settings`` / ``depth_ftg_config``.

    Map-free modes: ``frame_buffer``, ``camera_settings`` + ``depth_ftg_config``
    (``depth_ftg``), ``vae_config`` (``vae``) or ``bc_config`` (``bc``).

    ``pid`` / ``ftg``: ``pose_buffer``, ``boundary_buffer``, ``mapping_done_event``,
    ``pid_config`` / ``ftg_config``, ``bev_axes``, ``forward_axis``.
    ``lidar_pid``: ``lidar_buffer``, ``pid_config``.
    """

    def __init__(self, name: str = "control", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._mode: str = "pure_pursuit"
        self._pose_buf: PoseBuffer | None = None
        self._ctrl_buf: ControlBuffer | None = None
        # pid / ftg mode state
        self._boundary_buf: Optional[Any] = None
        self._pid = None
        self._mapping_done_event: Optional[mp.Event] = None
        self._outer_line: Optional[np.ndarray] = None
        self._inner_line: Optional[np.ndarray] = None
        self._last_pose_seq = 0
        # lidar_pid mode state
        self._lidar_buf: Optional[Any] = None
        self._lidar_wall_ctrl = None
        self._last_lidar_seq = 0

    def setup(self) -> None:
        self._mode = self.config.get("mode", "pure_pursuit")
        self._ctrl_buf = self.config["output_buffer"]

        modes = self._modes()
        if self._mode not in modes:
            raise ValueError(f"Unknown control mode {self._mode!r}; options: {sorted(modes)}")
        setup, self._step_fn = modes[self._mode]
        setup()

    def _modes(self) -> Dict[str, tuple]:
        """control_mode → (setup, step)."""
        return {
            "pid": (self._setup_pid, self._step_pid),
            "ftg": (self._setup_ftg, self._step_pid),
            "lidar_pid": (self._setup_lidar_pid, self._step_lidar_pid),
            **{m: (self._setup_track, self._step_track) for m in TRACK_MODES},
            "depth_ftg": (self._setup_depth_ftg, self._step_depth_ftg),
            "vae": (self._setup_learned_depth, self._step_depth_ftg),
            "bc": (self._setup_learned_depth, self._step_depth_ftg),
        }

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

    # --- Track-following modes (TRACK_MODES) ---

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

    def _setup_learned_depth(self) -> None:
        """``vae`` / ``bc``: an exported depth → steering ONNX model."""
        from control.imitation import DepthPolicyConfig, DepthPolicyFollower

        self._frame_buf = self.config["frame_buffer"]
        self._last_frame_seq = 0
        cfg = dict(self.config.get(f"{self._mode}_config", {}))
        model = cfg.pop("model", f"{self._mode}_policy.onnx")
        self._depth_follower = DepthPolicyFollower(model, DepthPolicyConfig(**cfg))
        self.logger.info(f"ControlWorker ready — {self._mode} mode (map-free, model={model})")

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

    # --- pid / ftg modes (raw pose + boundaries) ---

    def _step_pid(self) -> None:
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

        # Controller output is signed [-1, 1] (steering +1 = left)
        steer_raw, throttle_raw = self._pid.step(
            current_bev, heading, self._outer_line, self._inner_line
        )

        steer_unit, throttle_unit = signed_to_unit(steer_raw, throttle_raw)
        self._ctrl_buf.write(steer_unit, throttle_unit)

        if self._last_pose_seq % 30 == 0:
            self.logger.debug(
                f"[{self._mode}] pos={current_bev} steer={steer_raw:+.3f} "
                f"throttle={throttle_raw:+.3f} -> buffer ({steer_unit:.3f}, {throttle_unit:.3f})"
            )

    def teardown(self) -> None:
        self.logger.info(f"ControlWorker ({self._mode} mode) shutting down")
