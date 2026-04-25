"""
Bare-bones heading PID for track following. The local track direction is
estimated from the nearest outer and inner boundary points and the car is
steered to line up with it.

PID reference: Åström & Murray, "Feedback Systems", Princeton UP, 2008.

Output is (steering, throttle) in [-1, 1], +1 steer = left. Gaussian noise
is added on top of the PID output, so it's stochastic unless noise_std=0.
"""

from __future__ import annotations

import numpy as np


def _angle_diff(a: float, b: float) -> float:
    """a - b wrapped to [-pi, pi)."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


class TrackPIDController:
    """Heading PID. Throttle is constant (target_throttle) plus noise.

    noise_std is the steering noise std; throttle gets half of it.
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
        """One control step.

        Parameters
        ----------
        current_bev : np.ndarray
            (2,) vehicle position in the BEV plane.
        heading_rad : float
            BEV heading, same convention as workers.bev.pose_to_bev.
        outer_line, inner_line : np.ndarray
            (N, 2) / (M, 2) ordered boundary vertices.

        Returns (steering, throttle), clipped to [-1, 1].
        """
        d_outer = np.linalg.norm(outer_line - current_bev, axis=1)
        d_inner = np.linalg.norm(inner_line - current_bev, axis=1)
        closest_outer = outer_line[d_outer.argmin()]
        closest_inner = inner_line[d_inner.argmin()]

        # track direction is perpendicular to inner->outer; flip it if it
        # points against our current heading
        n_dir = closest_outer - closest_inner
        heading_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)])
        target_heading = np.array([-n_dir[1], n_dir[0]])

        if np.dot(target_heading, heading_vec) < 0:
            target_heading = -target_heading

        # positive error = track heads off to our left
        target_angle = np.arctan2(target_heading[1], target_heading[0])
        error = _angle_diff(target_angle, heading_rad)

        self._integral += error
        derivative = error - self._prev_error
        self._prev_error = error

        raw_steer = (
            self.kp * error + self.ki * self._integral + self.kd * derivative
        )
        raw_throttle = self.target_throttle

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
        self._prev_error = 0.0
        self._integral = 0.0
