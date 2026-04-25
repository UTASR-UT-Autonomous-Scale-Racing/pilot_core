"""
Longitudinal control. Curvature-based speed profile plus a throttle
controller for a drivetrain with no brake: neutral just coasts, and backward
input actively decelerates (and will reverse once stopped, so be careful).

Speed profile with forward/backward passes follows N. R. Kapania,
J. Subosits, J. C. Gerdes, "A Sequential Two-Step Algorithm for Fast
Generation of Vehicle Racing Trajectories", J. Dyn. Sys., Meas., Control
138(9), 2016. For the PI part see K. J. Åström, R. M. Murray, "Feedback Systems",
Princeton University Press, 2008.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import TrackModel

G = 9.81


def speed_profile(
    track: TrackModel,
    v_max: float,
    a_lat_max: float,
    a_accel: float,
    a_decel: float,
) -> np.ndarray:
    """Max speed at each centre-line sample.

    Starts from v <= sqrt(a_lat_max / |kappa|), capped at v_max, then does
    forward (accel) and backward (decel) passes around the loop so the
    profile is actually reachable under those limits.
    """
    kappa = np.abs(track.curvature)
    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(kappa, 1e-6)))
    ds = np.diff(np.concatenate([track.s, [track.length]]))
    n = len(v)
    # two laps of each pass to settle the wrap-around
    for _ in range(2):
        for i in range(n):                      # forward: accel limit
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_accel * ds[i]))
        for i in range(n - 1, -1, -1):          # backward: decel limit
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_decel * ds[i]))
    return v


@dataclass
class LongitudinalConfig:
    # steady-state m/s per unit throttle. calibrate this on the car
    speed_per_throttle: float = 3.0
    kp: float = 0.25                  # throttle per m/s of speed error
    ki: float = 0.15
    integral_limit: float = 0.15      # anti-windup clamp on I, in throttle units
    max_backward: float = 0.3         # most negative throttle used to slow down
    min_speed_for_backward: float = 0.3  # below this, backward input would reverse us


class LongitudinalController:
    """PI on speed, plus steady-state feed-forward."""

    def __init__(self, cfg: LongitudinalConfig | None = None) -> None:
        self.cfg = cfg or LongitudinalConfig()
        self._integral = 0.0

    def reset(self) -> None:
        self._integral = 0.0

    def step(self, v_ref: float, v: float, dt: float) -> float:
        c = self.cfg
        err = v_ref - v
        self._integral = float(np.clip(self._integral + c.ki * err * dt, -c.integral_limit, c.integral_limit))
        u = v_ref / c.speed_per_throttle + c.kp * err + self._integral
        if v < c.min_speed_for_backward:
            u = max(u, 0.0)          # coast rather than start reversing
        return float(np.clip(u, -c.max_backward, 1.0))
