"""
Follow-the-gap on a 2D LiDAR scan, for driving the mapping lap in the simulator (on
the car a human drives that lap with the RC transmitter). Slow and conservative: it
only has to get round once without touching a wall.

Scan convention (Hokuyo UST-10LX as simulated by AutoDRIVE, REP-103): 1080 ranges from
-135 deg (right) to +135 deg (left), 0 = straight ahead, inf = no return.
Ref: Sezer & Gokasan, "A novel obstacle avoidance algorithm: Follow the Gap Method", 2012.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import Command, VehicleParams

ANGLE_MIN = -2.35619449
ANGLE_INCREMENT = 0.00436332


@dataclass
class LidarGapConfig:
    speed: float = 1.0               # m/s on a clear straight
    min_speed: float = 0.5           # m/s in tight corners
    speed_per_throttle: float = 3.0  # m/s per unit throttle (same calibration as the car)
    max_range: float = 6.0           # m, longer returns are clipped (and inf -> this)
    fov: float = 1.75                # rad either side of straight ahead that is considered
    bubble: float = 0.35             # m, cleared around the closest return
    gap_threshold: float = 1.2       # m, a beam counts as free beyond this
    smooth: int = 9                  # beams, moving average
    centre_weight: float = 0.5       # 0 = aim at the deepest beam of the gap, 1 = its centre
    steer_gain: float = 1.0
    kp_speed: float = 0.3            # throttle per m/s of speed error


class LidarGapFollower:
    """step(ranges, speed) -> Command (signed: steering +1 = left)."""

    def __init__(self, params: VehicleParams, cfg: LidarGapConfig | None = None) -> None:
        self.p = params
        self.cfg = cfg or LidarGapConfig()
        self.last_target = 0.0

    def reset(self) -> None:
        self.last_target = 0.0

    def target_angle(self, ranges: np.ndarray) -> float:
        c = self.cfg
        r = np.nan_to_num(np.asarray(ranges, dtype=np.float64), nan=0.0, posinf=c.max_range)
        r = np.clip(r, 0.0, c.max_range)
        if c.smooth > 1:
            r = np.convolve(r, np.ones(c.smooth) / c.smooth, mode="same")
        ang = ANGLE_MIN + np.arange(len(r)) * ANGLE_INCREMENT
        r = np.where(np.abs(ang) <= c.fov, r, 0.0)

        # clear a bubble around the closest return (car width + margin)
        valid = r > 0.05
        if not valid.any():
            return 0.0
        i_min = int(np.argmin(np.where(valid, r, np.inf)))
        half = np.arctan2(c.bubble, max(r[i_min], 1e-3))
        r = np.where(np.abs(ang - ang[i_min]) <= half, 0.0, r)

        free = r > c.gap_threshold
        if not free.any():
            return float(ang[int(np.argmax(r))])
        # longest run of free beams
        edges = np.diff(np.concatenate([[0], free.astype(int), [0]]))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        k = int(np.argmax(ends - starts))
        s, e = starts[k], ends[k]
        deepest = s + int(np.argmax(r[s:e]))
        centre = (s + e - 1) / 2.0
        idx = (1 - c.centre_weight) * deepest + c.centre_weight * centre
        return float(ANGLE_MIN + idx * ANGLE_INCREMENT)

    def step(self, ranges: np.ndarray, speed: float | None) -> Command:
        c = self.cfg
        target = self.target_angle(ranges)
        self.last_target = target
        steer = float(np.clip(c.steer_gain * target / self.p.max_steering_angle, -1.0, 1.0))
        v_ref = c.min_speed + (c.speed - c.min_speed) * (1.0 - min(abs(steer), 1.0))
        throttle = v_ref / c.speed_per_throttle
        if speed is not None and np.isfinite(speed):
            throttle += c.kp_speed * (v_ref - speed)
        return Command(steer, float(np.clip(throttle, 0.0, 1.0)))
