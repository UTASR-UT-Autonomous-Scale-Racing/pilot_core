"""
Wall-centering PID on the live LIDAR scan. No map needed, so it works on
any track shape: take the forward sector of the scan, get a distance to the
wall on each side, steer to even them out.

Hokuyo UST-10LX, 1080 rays at 0.25 deg each:
    ray 0    -> -135 deg (rear-right)
    ray 540  ->    0 deg (straight ahead)
    ray 1079 -> +135 deg (rear-left)
Positive angles are the LEFT side (y > 0 in the vehicle frame, ROS style).

With half_rays = sector_deg / 0.25 the sectors are
    front_left  = rays 540 .. 540 + half_rays    [0, +sector_deg]
    front_right = rays 540 - half_rays .. 540    [-sector_deg, 0]

Steering out is in [-1, 1], +1 = left.
"""

from __future__ import annotations

import numpy as np


class LidarWallController:
    """PID on the normalised left/right wall distance difference.

    kp/ki/kd are the gains on that lateral error. target_throttle is the base
    throttle in [-1, 1]. sector_deg is the half-angle of the forward sector
    used to find walls (60 deg by default). Rays outside
    [range_min, range_max] are ignored. slow_corner_factor (0-1) is the
    throttle multiplier at full steering lock; it scales linearly in between.
    """

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.1,
        target_throttle: float = 0.4,
        sector_deg: float = 60.0,
        range_min: float = 0.1,
        range_max: float = 9.5,
        slow_corner_factor: float = 0.6,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_throttle = target_throttle
        self.range_min = range_min
        self.range_max = range_max
        self.slow_corner_factor = slow_corner_factor

        # 0.25 deg/ray, centre ray is 540
        half_rays = int(sector_deg / 0.25)
        self._left_start = 540
        self._left_end = min(540 + half_rays, 1080)
        self._right_start = max(540 - half_rays, 0)
        self._right_end = 540

        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, ranges: np.ndarray) -> tuple[float, float]:
        """ranges is the (1080,) float32 scan in metres.

        Returns (steering, throttle), both clipped to [-1, 1].
        """
        left_rays = ranges[self._left_start:self._left_end]
        right_rays = ranges[self._right_start:self._right_end]

        left_valid = left_rays[(left_rays >= self.range_min) & (left_rays <= self.range_max)]
        right_valid = right_rays[(right_rays >= self.range_min) & (right_rays <= self.range_max)]

        # open corner -> no valid returns on that side, treat it as range_max
        left_d = float(np.median(left_valid)) if len(left_valid) > 0 else self.range_max
        right_d = float(np.median(right_valid)) if len(right_valid) > 0 else self.range_max

        # error > 0: closer to the right wall, so steer left (+).
        # error < 0: closer to the left wall, steer right.
        error = (right_d - left_d) / max(left_d + right_d, 0.01)

        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        raw_steer = self.kp * error + self.ki * self._integral + self.kd * derivative
        steer = float(np.clip(raw_steer, -1.0, 1.0))

        # back off the throttle in proportion to |steer|
        throttle = self.target_throttle * (
            1.0 - (1.0 - self.slow_corner_factor) * abs(steer)
        )
        throttle = float(np.clip(throttle, -1.0, 1.0))

        return steer, throttle

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
