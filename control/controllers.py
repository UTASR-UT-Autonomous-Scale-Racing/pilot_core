"""
Track-following controllers: lateral (Pure Pursuit, Stanley) combined with
the curvature-based speed planner into a :class:`TrackFollower`.

All controllers output a signed :class:`Command` (steering +1 = left).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Protocol

import numpy as np

from .speed import G, LongitudinalConfig, LongitudinalController, speed_profile
from .track import Projection, TrackModel
from .types import Command, VehicleParams, VehicleState, wrap_angle


class LateralController(Protocol):
    def reset(self) -> None: ...
    def steer(self, state: VehicleState, track: TrackModel, proj: Projection) -> float:
        """Return the road-wheel angle in radians (+ = left)."""
        ...


@dataclass
class PurePursuit:
    """Geometric path tracking towards a look-ahead point on the centre-line.

    ``L_d = clip(k_v * v + lookahead_min, lookahead_min, lookahead_max)``,
    ``delta = atan(2 L sin(alpha) / L_d)``.
    """

    params: VehicleParams
    lookahead_min: float = 0.6
    lookahead_max: float = 2.0
    k_v: float = 0.5

    def reset(self) -> None:
        pass

    def lookahead(self, v: float) -> float:
        return float(np.clip(self.k_v * abs(v) + self.lookahead_min, self.lookahead_min, self.lookahead_max))

    def steer(self, state: VehicleState, track: TrackModel, proj: Projection) -> float:
        ld = self.lookahead(state.v)
        target = track.point_at(proj.s + ld)
        dx, dy = target[0] - state.x, target[1] - state.y
        alpha = wrap_angle(np.arctan2(dy, dx) - state.yaw)
        dist = max(np.hypot(dx, dy), 1e-3)
        return float(np.arctan2(2.0 * self.params.wheelbase * np.sin(alpha), dist))


@dataclass
class Stanley:
    """Front-axle heading + cross-track error law.

    ``delta = heading_error - atan(k * e / (k_soft + v))`` evaluated at the
    front axle (pose reference is taken as the rear axle).
    """

    params: VehicleParams
    k: float = 1.0
    k_soft: float = 1.0
    k_yaw_rate: float = 0.0    # optional damping on estimated yaw rate

    def reset(self) -> None:
        pass

    def steer(self, state: VehicleState, track: TrackModel, proj: Projection) -> float:
        L = self.params.wheelbase
        fx = state.x + L * np.cos(state.yaw)
        fy = state.y + L * np.sin(state.yaw)
        fp = track.project(fx, fy, hint=proj.idx)
        heading_err = wrap_angle(fp.path_heading - state.yaw)
        cross = np.arctan2(self.k * fp.lateral_error, self.k_soft + abs(state.v))
        return float(heading_err - cross - self.k_yaw_rate * state.yaw_rate)


@dataclass
class SpeedConfig:
    v_max: float = 1.3            # m/s — keep a margin below the safety speed cap (1.5)
    lat_accel_fraction: float = 0.6   # use this fraction of friction_coeff * g for corners
    accel: float = 1.5            # m/s² used by the profile (well below the car's limit)
    decel: float = 1.0            # m/s² — coasting + limited backward input
    preview: float = 0.5          # s — look this far ahead for the speed reference
    longitudinal: LongitudinalConfig = field(default_factory=LongitudinalConfig)


class TrackFollower:
    """Lateral controller + speed planner on a :class:`TrackModel`."""

    def __init__(
        self,
        lateral: LateralController,
        track: TrackModel,
        params: VehicleParams,
        speed: SpeedConfig | None = None,
    ) -> None:
        self.lateral = lateral
        self.track = track
        self.params = params
        self.speed = speed or SpeedConfig()
        self.longitudinal = LongitudinalController(self.speed.longitudinal)
        self._hint: int | None = None
        self._last_t: float | None = None
        self._v_cmd: float | None = None
        self.last_projection: Projection | None = None
        self.v_profile = speed_profile(
            track,
            v_max=min(self.speed.v_max, params.max_speed),
            a_lat_max=self.speed.lat_accel_fraction * params.friction_coeff * G,
            a_accel=self.speed.accel,
            a_decel=self.speed.decel,
        )

    def reset(self) -> None:
        self.lateral.reset()
        self.longitudinal.reset()
        self._hint = None
        self._last_t = None
        self._v_cmd = None

    def v_ref(self, proj: Projection, v: float, dt: float) -> float:
        """Profile speed a short preview ahead, ramped at ``accel`` so the
        reference never jumps ahead of the car (e.g. from standstill)."""
        s_preview = proj.s + max(abs(v), 0.3) * self.speed.preview
        target = float(self.v_profile[self.track.index_at(s_preview)])
        prev = self._v_cmd if self._v_cmd is not None else max(v, 0.0)
        self._v_cmd = min(target, prev + self.speed.accel * dt)
        return self._v_cmd

    def step(self, state: VehicleState) -> Command:
        dt = 0.0 if self._last_t is None else max(state.t - self._last_t, 0.0)
        self._last_t = state.t
        proj = self.track.project(state.x, state.y, hint=self._hint)
        self._hint = proj.idx
        self.last_projection = proj

        delta = self.lateral.steer(state, self.track, proj)
        steering = delta / self.params.max_steering_angle
        dt = min(dt, 0.2)
        throttle = self.longitudinal.step(self.v_ref(proj, state.v, dt), state.v, dt)
        return Command(steering, throttle).clipped()


# --------------------------------------------------------------------- registry

def _pure_pursuit(params: VehicleParams, cfg: Dict[str, Any]) -> LateralController:
    return PurePursuit(params, **cfg)


def _stanley(params: VehicleParams, cfg: Dict[str, Any]) -> LateralController:
    return Stanley(params, **cfg)


LATERAL_CONTROLLERS: Dict[str, Callable[[VehicleParams, Dict[str, Any]], LateralController]] = {
    "pure_pursuit": _pure_pursuit,
    "stanley": _stanley,
}


def build_track_follower(
    mode: str,
    track: TrackModel,
    params: VehicleParams,
    lateral_cfg: Dict[str, Any] | None = None,
    speed_cfg: Dict[str, Any] | None = None,
) -> TrackFollower:
    """Build a :class:`TrackFollower` for ``control_mode`` ``mode``."""
    if mode not in LATERAL_CONTROLLERS:
        raise ValueError(f"Unknown track controller {mode!r}; options: {sorted(LATERAL_CONTROLLERS)}")
    speed_cfg = dict(speed_cfg or {})
    long_cfg = LongitudinalConfig(**speed_cfg.pop("longitudinal", {}))
    speed = SpeedConfig(longitudinal=long_cfg, **speed_cfg)
    return TrackFollower(LATERAL_CONTROLLERS[mode](params, dict(lateral_cfg or {})), track, params, speed)
