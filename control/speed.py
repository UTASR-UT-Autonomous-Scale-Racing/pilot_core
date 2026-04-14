"""
Longitudinal control: a curvature-based speed profile and a throttle
controller for a drivetrain with **no brake** — neutral coasts, and a
backward input actively decelerates (and would reverse once stopped).

Reference (speed profile, forward/backward passes): N. R. Kapania,
J. Subosits, J. C. Gerdes, "A Sequential Two-Step Algorithm for Fast
Generation of Vehicle Racing Trajectories", J. Dyn. Sys., Meas., Control
138(9), 2016.  PI control: K. J. Åström, R. M. Murray, "Feedback Systems",
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
    """Maximum speed per centre-line sample.

    ``v <= sqrt(a_lat_max / |kappa|)`` capped at ``v_max``, then forward
    (acceleration) and backward (deceleration) passes around the closed
    loop so the profile is reachable with the given limits.
    """
    kappa = np.abs(track.curvature)
    v = np.minimum(v_max, np.sqrt(a_lat_max / np.maximum(kappa, 1e-6)))
    ds = np.diff(np.concatenate([track.s, [track.length]]))
    n = len(v)
    for _ in range(2):  # two laps of each pass settle the wrap-around
        for i in range(n):                      # forward: acceleration limit
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2 * a_accel * ds[i]))
        for i in range(n - 1, -1, -1):          # backward: deceleration limit
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2 * a_decel * ds[i]))
    return v


@dataclass
class LongitudinalConfig:
    speed_per_throttle: float = 3.0   # m/s of steady-state speed per unit throttle (calibrate!)
    kp: float = 0.25                  # throttle per m/s of speed error
    ki: float = 0.15
    integral_limit: float = 0.15      # anti-windup clamp on the I-term (throttle units)
    max_backward: float = 0.3         # most negative throttle used to slow down
    min_speed_for_backward: float = 0.3  # never push backward below this speed (would reverse)


class LongitudinalController:
    """PI speed controller with steady-state feed-forward."""

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
