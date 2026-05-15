"""
Runs the configured control mode (track modes off state + map, or map-free depth modes)
and writes steering/throttle to the ControlBuffer.
Output contract: [0, 1] with 0.5 = neutral; steering 0 = full left, 1 = full right,
throttle < 0.5 reverse. Controllers use signed [-1, 1] (+steering = left), see signed_to_unit().
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import ControlBuffer


def signed_to_unit(steering: float, throttle: float) -> tuple[float, float]:
    """Signed controller output (steering +1 = full left, REP-103) -> ControlBuffer
    [0, 1], where the car's PWM steers right above neutral, hence the steering flip."""
    steer_unit = float(np.clip((1.0 - steering) / 2.0, 0.0, 1.0))
    throttle_unit = float(np.clip((throttle + 1.0) / 2.0, 0.0, 1.0))
    return steer_unit, throttle_unit


# these use the state estimate + mapped boundaries
TRACK_MODES = ("pure_pursuit", "stanley", "lqr", "mpc", "policy")


def unit_to_signed(steering: float, throttle: float) -> tuple[float, float]:
    """Inverse of signed_to_unit()."""
    return 1.0 - 2.0 * steering, 2.0 * throttle - 1.0


class ControlWorker(BaseWorker):
    """Vehicle control worker. Which kwargs matter depends on the mode.

    Config kwargs:
      mode                one of TRACK_MODES, depth_ftg, vae, bc (default pure_pursuit)
      output_buffer       ControlBuffer
      armed_event         optional; controllers are reset on arming so nothing winds up
      state_buffer, boundary_buffer, mapping_done_event   track modes
      track_config        track_control config section (fallback: depth_ftg)
      vehicle_params      vehicle params YAML
      safety              runtime safety settings, compared with a policy's
      frame_buffer, camera_settings, depth_ftg_config     depth_ftg / fallback
      vae_config, bc_config                               vae / bc
    """

    def __init__(self, name: str = "control", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._mode: str = "pure_pursuit"
        self._ctrl_buf: ControlBuffer | None = None
        self._boundary_buf: Optional[Any] = None
        self._mapping_done_event: Optional[mp.Event] = None
        self._armed_event: Optional[mp.Event] = None
        self._was_armed = False
        self._follower = None
        self._fallback = None
        self._depth_follower = None

    def setup(self) -> None:
        self._mode = self.config.get("mode", "pure_pursuit")
        self._ctrl_buf = self.config["output_buffer"]
        self._armed_event = self.config.get("armed_event")

        modes = self._modes()
        if self._mode not in modes:
            raise ValueError(f"Unknown control mode {self._mode!r}; options: {sorted(modes)}")
        setup, self._step_fn = modes[self._mode]
        setup()

    def _modes(self) -> Dict[str, tuple]:
        """mode name -> (setup, step)"""
        return {
            **{m: (self._setup_track, self._step_track) for m in TRACK_MODES},
            "depth_ftg": (self._setup_depth_ftg, self._step_depth_ftg),
            "vae": (self._setup_learned_depth, self._step_depth_ftg),
            "bc": (self._setup_learned_depth, self._step_depth_ftg),
        }

    def step(self) -> None:
        armed = self._armed_event is None or self._armed_event.is_set()
        if armed and not self._was_armed:
            for follower in (self._follower, self._fallback, self._depth_follower):
                if follower is not None:
                    follower.reset()
        self._was_armed = armed
        self._step_fn()

    # track following

    def _setup_track(self) -> None:
        from control.types import VehicleParams

        self._state_buf = self.config["state_buffer"]
        self._boundary_buf = self.config["boundary_buffer"]
        self._mapping_done_event = self.config.get("mapping_done_event")
        self._vehicle = VehicleParams.load(self.config.get("vehicle_params", "vehicle_params.yaml"))
        self._track_cfg = self.config.get("track_config", {})
        self._last_state_seq = 0
        self._last_state_time = time.monotonic()
        self._next_policy_step = 0.0
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
            # step at the rate the policy was trained at (TrackEnv's 10 Hz for old sidecars)
            hz = self._follower.policy_hz or 10.0
            self._policy_period = 1.0 / hz
            self.logger.info(f"Policy loaded: {model} (obs_dim={self._follower.meta.obs_dim}, {hz:g} Hz)")
            trained = self._follower.safety
            runtime = self.config.get("safety")
            if trained and runtime and any(trained.get(k) != v for k, v in runtime.items() if k in trained):
                self.logger.warning(f"Policy was trained with different safety settings: {trained}")
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
            if self._fallback is not None and time.monotonic() - self._last_state_time > self._fallback_after:
                self._step_fallback()
            else:
                time.sleep(0.002)
            return
        header, state = result
        self._last_state_seq = header.seq_no
        self._last_state_time = time.monotonic()
        if self._fallback is not None and self._fallback_active:
            self._fallback_active = False
            if self._follower is not None:
                self._follower.reset()
            self.logger.info("State estimate back — leaving depth_ftg fallback")

        if self._follower is None:
            bnd = self._boundary_buf.read_latest()
            if bnd is None:
                time.sleep(0.05)
                return
            _, outer, inner = bnd
            self._build_follower(outer, inner, state)

        if self._mode == "policy":
            now = time.monotonic()
            if now < self._next_policy_step:
                return
            self._next_policy_step = now + self._policy_period
        cmd = self._follower.step(state)
        steer_unit, throttle_unit = signed_to_unit(cmd.steering, cmd.throttle)
        self._ctrl_buf.write(steer_unit, throttle_unit)

        if self._last_state_seq % 50 == 0 and self._mode != "policy":
            pr = self._follower.last_projection
            self.logger.debug(
                f"[{self._mode}] s={pr.s:.1f} e={pr.lateral_error:+.3f} v={state.v:.2f} "
                f"steer={cmd.steering:+.3f} throttle={cmd.throttle:+.3f}"
            )

    # depth Follow-the-Gap, either as its own mode or as the track-mode fallback

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
        """vae / bc: both are an exported depth -> steering ONNX model."""
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
        """Returns False if there was no new depth frame."""
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
            self._fallback.reset()
            self.logger.warning("State estimate stale — driving on depth_ftg fallback")
        if not self._depth_command(self._fallback):
            time.sleep(0.005)

    def teardown(self) -> None:
        self.logger.info(f"ControlWorker ({self._mode} mode) shutting down")
