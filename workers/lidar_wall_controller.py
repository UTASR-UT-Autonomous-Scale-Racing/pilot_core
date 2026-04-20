"""
LidarWallController — real-time wall-centering PID using the live LIDAR scan.

Works on any track shape (no global map required). At each step it reads the
forward-facing sector of the 2D LIDAR scan, measures the closest wall on each
side, and steers to minimise the lateral offset from the centre line.

Hokuyo UST-10LX ray layout (1080 rays, 0.25° per ray):
    Ray 0   → −135° (rear-right)
    Ray 540 → 0°    (straight ahead)
    Ray 1079→ +135° (rear-left)
    Positive angles = LEFT side (y > 0 in vehicle frame, ROS convention)
    Negative angles = RIGHT side (y < 0 in vehicle frame)

Forward-facing sectors used for wall estimation (``half_rays = sector_deg / 0.25``):
    front_left  = rays 540 → (540 + half_rays)   [0° … +sector_deg]
    front_right = rays (540 − half_rays) → 540   [−sector_deg … 0°]

Output is signed [-1, 1] with steering +1 = LEFT.
"""

from __future__ import annotations

import numpy as np


class LidarWallController:
    """PID wall-centering controller driven by the live LIDAR scan.

    Parameters
    ----------
    kp, ki, kd : float
        PID gains for the lateral centering error.
    target_throttle : float
        Base throttle in [−1, 1].
    sector_deg : float
        Half-angle of the front-facing sector used for wall detection (default 60°).
    range_min, range_max : float
        Valid range limits; rays outside this are ignored.
    slow_corner_factor : float
        Throttle is multiplied by this when a sharp steer is needed (0–1).
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

        # Sector half-angle → ray indices (0.25° per ray, centre ray = 540)
        half_rays = int(sector_deg / 0.25)
        self._left_start = 540
        self._left_end = min(540 + half_rays, 1080)
        self._right_start = max(540 - half_rays, 0)
        self._right_end = 540

        self._integral = 0.0
        self._prev_error = 0.0

    def step(self, ranges: np.ndarray) -> tuple[float, float]:
        """Compute steering and throttle from a 1080-ray LIDAR scan.

        Parameters
        ----------
        ranges : np.ndarray
            Shape ``(1080,)`` float32, values in metres.

        Returns
        -------
        (steering, throttle) — both clipped to [−1, 1].
        """
        left_rays = ranges[self._left_start:self._left_end]
        right_rays = ranges[self._right_start:self._right_end]

        left_valid = left_rays[(left_rays >= self.range_min) & (left_rays <= self.range_max)]
        right_valid = right_rays[(right_rays >= self.range_min) & (right_rays <= self.range_max)]

        # Fallback: if one side has no valid readings (open corner), use range_max
        left_d = float(np.median(left_valid)) if len(left_valid) > 0 else self.range_max
        right_d = float(np.median(right_valid)) if len(right_valid) > 0 else self.range_max

        # Lateral error: positive → closer to right wall → steer left (+)
        #                negative → closer to left wall  → steer right (−)
        error = (right_d - left_d) / max(left_d + right_d, 0.01)

        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        raw_steer = self.kp * error + self.ki * self._integral + self.kd * derivative
        steer = float(np.clip(raw_steer, -1.0, 1.0))

        # Slow down proportionally when steering hard
        throttle = self.target_throttle * (
            1.0 - (1.0 - self.slow_corner_factor) * abs(steer)
        )
        throttle = float(np.clip(throttle, -1.0, 1.0))

        return steer, throttle

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0
