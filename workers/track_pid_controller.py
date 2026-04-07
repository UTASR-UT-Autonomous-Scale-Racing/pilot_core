"""
TrackPIDController — minimal PID that steers the vehicle along the
centre-line of extracted track boundaries.

Output: (steering, throttle) both in [-1, 1], stochastically sampled
from the PID output with configurable Gaussian noise.
"""

from __future__ import annotations

import numpy as np


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference in radians."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


class TrackPIDController:
    """PID controller for autonomous track following.

    Parameters
    ----------
    kp_steer : float
        Proportional gain for steering.
    ki_steer : float
        Integral gain for steering.
    kd_steer : float
        Derivative gain for steering.
    target_throttle : float
        Base throttle in [-1, 1] (default 0.3).
    noise_std : float
        Gaussian noise std for stochastic sampling (default 0.05).
    """

    def __init__(
        self,
        kp_steer: float = 1.0,
        ki_steer: float = 0.0,
        kd_steer: float = 0.1,
        target_throttle: float = 0.3,
        noise_std: float = 0.05,
    ) -> None:
        self.kp = kp_steer
        self.ki = ki_steer
        self.kd = kd_steer
        self.target_throttle = target_throttle
        self.noise_std = noise_std
        self._prev_error = 0.0
        self._integral = 0.0

    def step(
        self,
        current_bev: np.ndarray,
        heading_rad: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> tuple[float, float]:
        """Compute one PID control step.

        Parameters
        ----------
        current_bev : np.ndarray
            ``(2,)`` current vehicle position in BEV (X, Z).
        heading_rad : float
            Vehicle heading angle in the BEV (X, Z) plane.
        outer_line : np.ndarray
            ``(N, 2)`` ordered outer boundary vertices.
        inner_line : np.ndarray
            ``(M, 2)`` ordered inner boundary vertices.

        Returns
        -------
        (steering, throttle) — both clipped to [-1, 1].
        """
        # 1. Find closest point on each boundary
        d_outer = np.linalg.norm(outer_line - current_bev, axis=1)
        d_inner = np.linalg.norm(inner_line - current_bev, axis=1)
        closest_outer = outer_line[d_outer.argmin()]
        closest_inner = inner_line[d_inner.argmin()]
        print(f"Closest Outer Point: {closest_outer}")
        print(f"Closest Inner Point: {closest_inner}")

        # 2. Normal vector to the desired heading
        n_dir = closest_outer - closest_inner
        heading_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        target_heading = np.array([-n_dir[1], n_dir[0]])

        if np.dot(target_heading, heading_vec) < 0:
            target_heading = -target_heading

        # 3. Cross-track error (signed angular offset)
        target_angle = np.arctan2(target_heading[1], target_heading[0])
        error = _angle_diff(target_angle, heading_rad)

        print(f"Target Angle: {target_angle}")
        print(f"Current Angle: {heading_rad}")

        # 4. PID
        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        raw_steer = (
            self.kp * error + self.ki * self._integral + self.kd * derivative
        )
        raw_throttle = self.target_throttle

        # 5. Stochastic sampling
        steer = np.clip(
            raw_steer + np.random.normal(0, self.noise_std), -1.0, 1.0
        )
        throttle = np.clip(
            raw_throttle + np.random.normal(0, self.noise_std * 0.5),
            -1.0,
            1.0,
        )

        return float(steer), float(throttle)

    def reset(self) -> None:
        """Reset integrator and derivative state."""
        self._prev_error = 0.0
        self._integral = 0.0
