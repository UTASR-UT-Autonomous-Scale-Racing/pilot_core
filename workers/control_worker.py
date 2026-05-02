"""
Runs whichever control mode is configured and writes steering/throttle to
the ControlBuffer.

The track-following modes (pure_pursuit, the default, plus stanley, lqr,
mpc and policy; see control/) drive off the state estimate and the mapped
boundaries, and can optionally drop to depth_ftg while the state estimate
is stale. depth_ftg, vae and bc are map-free and work straight from depth
frames (vae is the learned VAE control model, bc the behaviour-cloning
CNN).

Controllers are reset while the car is disarmed so integrators don't wind
up before the car is allowed to move.

Output contract, since everything downstream depends on it: ControlBuffer
values are [0, 1] with 0.5 = neutral on both channels. That's what
PWMMapper (linear over each channel's PWM [min, max]) and the sim bridge
expect: steering 0 = full left, 1 = full right, throttle < 0.5 backward,
> 0.5 forward. The controllers themselves work in signed [-1, 1] with
+steering = LEFT, and get converted by signed_to_unit() below.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import ControlBuffer


def signed_to_unit(steering: float, throttle: float) -> tuple[float, float]:
    """Signed controller output -> ControlBuffer [0, 1].

    All the track controllers and RL policies use steering +1 = full left
    (ROS / REP-103, CCW positive) and throttle +1 = full forward, -1 = full
    reverse. ControlBuffer follows the car's PWM instead, where going
    above neutral steers right (0 = full left, 1 = full right), hence the
    sign flip on steering. 0 maps to 0.5 on both channels.
    """
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

    always: mode, output_buffer (ControlBuffer), armed_event (optional;
        controllers are reset while it is clear)
    track modes: state_buffer, boundary_buffer, mapping_done_event,
        track_config (the track_control config section), vehicle_params,
        plus frame_buffer / camera_settings / depth_ftg_config if the
        fallback is on
    depth_ftg: frame_buffer, camera_settings, depth_ftg_config
    vae / bc: frame_buffer, vae_config or bc_config
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
        # optional map-free fallback for when the state estimate goes stale
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
