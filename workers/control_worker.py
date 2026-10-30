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
      jump_reset_m        optional; a state this far from the last one resets them too (sim respawn)
      state_buffer, boundary_buffer, mapping_done_event   track modes
      track_config        track_control config section (fallback: depth_ftg;
                          mapping_drive: LidarGapConfig fields, drives the mapping lap)
      lidar_buffer        LidarBuffer, for track_config.mapping_drive (simulator)
      segment_buffer      SegmentationBuffer: without a lidar_buffer, mapping_drive gets an
                          approximate scan from the mask's edge and the frame's depth
                          (control/seg_scan.py; needs frame_buffer)
      seg_scan            seg_scan config: camera_settings + SegScanConfig fields
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
        self._mapping_driver = None
        self._mapping_cmd = None

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
            for follower in (self._follower, self._fallback, self._depth_follower, self._mapping_driver):
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
        self._last_state_xy = None
        self._jump_reset_m = self.config.get("jump_reset_m")
        self._last_state_time = time.monotonic()
        self._next_policy_step = 0.0
        if self._track_cfg.get("fallback") == "depth_ftg":
            self._fallback = self._make_depth_follower()
            self._fallback_after = float(self._track_cfg.get("fallback_after", 0.3))
            self._fallback_active = False
        drive_cfg = self._track_cfg.get("mapping_drive")
        if drive_cfg:
            from control.lidar_gap import LidarGapConfig, LidarGapFollower

            self._lidar_buf = self.config.get("lidar_buffer")
            self._last_lidar_seq = 0
            if self._lidar_buf is None:
                self._setup_seg_scan()
            cfg = dict(drive_cfg)
            cfg.setdefault("speed_per_throttle", self._track_cfg.get("speed", {}).get(
                "longitudinal", {}).get("speed_per_throttle", 3.0))
            self._mapping_driver = LidarGapFollower(self._vehicle, LidarGapConfig(**cfg))
            source = "LiDAR" if self._lidar_buf is not None else "segmentation-edge scan"
            self.logger.info(f"Mapping lap driven by follow-the-gap on the {source} ({cfg})")
        self.logger.info(
            f"ControlWorker ready — {self._mode} mode "
            f"(wheelbase={self._vehicle.wheelbase} m, max_steer={self._vehicle.max_steering_angle} rad)"
        )

    def _build_follower(self, outer: np.ndarray, inner: np.ndarray, state) -> None:
        from control.controllers import build_track_follower
        from control.track import TrackModel
        from control.types import resolve_path

        cfg = self._track_cfg
        if self._mode == "policy":
            from control.policy import PolicyFollower

            model = cfg.get("policy", {}).get("model", "policy.onnx")
            self._follower = PolicyFollower(model, outer, inner)
            # step at the rate the policy was trained at (10 Hz if the sidecar doesn't record it)
            hz = self._follower.policy_hz or 10.0
            self._policy_period = 1.0 / hz
            self.logger.info(f"Policy loaded: {model} (obs_dim={self._follower.meta.obs_dim}, {hz:g} Hz)")
            trained = self._follower.safety
            runtime = self.config.get("safety")
            if trained and runtime and any(trained.get(k) != v for k, v in runtime.items() if k in trained):
                self.logger.warning(f"Policy was trained with different safety settings: {trained}")
            return
        track = TrackModel(outer, inner, spacing=cfg.get("spacing", 0.05))
        raceline = cfg.get("raceline_path")
        if raceline:
            # precomputed by scripts/make_raceline.py from the same map
            track.set_path(np.load(resolve_path(raceline)), spacing=cfg.get("spacing", 0.05))
            self.logger.info(f"Following the racing line {raceline}")
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
            if self._mapping_driver is not None:
                self._step_mapping_drive()
            else:
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
        if self._jump_reset_m is not None and self._last_state_xy is not None and \
                np.hypot(state.x - self._last_state_xy[0], state.y - self._last_state_xy[1]) > self._jump_reset_m:
            # the car was put somewhere else (sim respawn): nothing the controllers integrated applies
            for follower in (self._follower, self._fallback):
                if follower is not None:
                    follower.reset()
            self.logger.info("Pose jump (respawn?) — controllers reset")
        self._last_state_xy = (state.x, state.y)
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

    def _setup_seg_scan(self) -> None:
        """No LiDAR: build the mapping drive's scan from the segmentation mask + depth."""
        from control.depth_ftg import Intrinsics
        from control.seg_scan import SegScanConfig

        self._seg_buf = self.config.get("segment_buffer")
        self._frame_buf = self.config.get("frame_buffer")
        if self._seg_buf is None or self._frame_buf is None:
            raise RuntimeError("mapping_drive needs a scan: a lidar_buffer, or a segment_buffer and "
                               "frame_buffer for the segmentation-edge scan (vision.enabled)")
        cfg = dict(self.config.get("seg_scan", {}))
        self._seg_K = Intrinsics.from_orbslam_settings(
            cfg.pop("camera_settings", self.config.get("camera_settings", "orbslam_oakd_rgbd.yaml")))
        self._seg_cfg = SegScanConfig(**cfg)

    def _read_scan(self):
        """Newest unseen scan as (header, ranges), or None: the LiDAR's, or one built from the
        newest mask and the depth of the frame it was computed from."""
        if self._lidar_buf is not None:
            result = self._lidar_buf.read_latest()
            if result is None or result[0].seq_no <= self._last_lidar_seq:
                return None
            return result
        from control.seg_scan import seg_edge_scan

        seg = self._seg_buf.read_latest()
        if seg is None or seg[0].seq_no <= self._last_lidar_seq:
            return None
        header, mask = seg
        # VisionWorker stamps a mask with its frame's capture time: find that frame's depth
        frames = [self._frame_buf.read_latest()]
        if frames[0] is not None and frames[0][0].produced_ts != header.produced_ts:
            frames = self._frame_buf.read_since(max(0, frames[0][0].seq_no - 3))   # 1-2 behind normally
        depth = next((f[2] for f in frames if f is not None and f[0].produced_ts == header.produced_ts), None)
        if depth is None:
            self._last_lidar_seq = header.seq_no      # its frame has left the ring; wait for the next
            return None
        return header, seg_edge_scan(mask, depth, self._seg_K, self._seg_cfg)

    def _step_mapping_drive(self) -> None:
        """Follow-the-gap until the track is mapped (sim stand-in for the RC lap): on the
        LiDAR, or without one on the segmentation-edge scan."""
        result = self._read_scan()
        now = time.monotonic()
        if result is None:
            # scans come at the sim's ~10 Hz: hold the last command (re-stamped, so the
            # 150 ms watchdog doesn't trip between scans) while the scan is recent
            last = self._mapping_cmd
            if last is not None and now - last[0] < 0.4 and now - last[1] >= 0.02:
                self._ctrl_buf.write(*signed_to_unit(last[2].steering, last[2].throttle))
                self._mapping_cmd = (last[0], now, last[2])
            time.sleep(0.005)
            return
        header, ranges = result
        self._last_lidar_seq = header.seq_no
        st = self._state_buf.read_latest()
        speed = st[1].v if st is not None and time.monotonic() - st[1].t < 0.5 else None
        cmd = self._mapping_driver.step(ranges, speed)
        self._ctrl_buf.write(*signed_to_unit(cmd.steering, cmd.throttle))
        self._mapping_cmd = (header.produced_ts, now, cmd)   # (scan time, last write, command)

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
