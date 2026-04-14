"""
Vehicle state estimator: an EKF on a unicycle model
``[x, y, yaw, v, yaw_rate]`` fed by pose measurements (SLAM, ~15 Hz) and,
when available, a direct speed measurement (e.g. simulator ground truth).

Reference: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics",
MIT Press, 2005 (Ch. 3, extended Kalman filter).

The car has no wheel encoders, so speed is inferred from successive poses;
the filter smooths that and lets controllers query the state at any time
between pose updates (``predict_to``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import VehicleState, wrap_angle


@dataclass
class EstimatorConfig:
    pos_std: float = 0.03          # m, pose position measurement noise
    yaw_std: float = 0.03          # rad, pose yaw measurement noise
    speed_std: float = 0.05        # m/s, direct speed measurement noise
    accel_std: float = 2.0         # m/s², process noise on v
    yaw_accel_std: float = 4.0     # rad/s², process noise on yaw rate
    stale_after: float = 0.5       # s without a pose → estimate no longer valid


class PoseEKF:
    def __init__(self, cfg: EstimatorConfig | None = None) -> None:
        self.cfg = cfg or EstimatorConfig()
        self.x: np.ndarray | None = None          # [x, y, yaw, v, w]
        self.P = np.eye(5)
        self.t: float | None = None
        self.last_pose_t: float | None = None

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def reset(self) -> None:
        self.x = None
        self.t = None
        self.last_pose_t = None

    # ------------------------------------------------------------- internals

    def _predict(self, dt: float) -> None:
        if dt <= 0:
            return
        x, y, yaw, v, w = self.x
        c, s = np.cos(yaw), np.sin(yaw)
        self.x = np.array([x + v * c * dt, y + v * s * dt, wrap_angle(yaw + w * dt), v, w])
        F = np.eye(5)
        F[0, 2], F[0, 3] = -v * s * dt, c * dt
        F[1, 2], F[1, 3] = v * c * dt, s * dt
        F[2, 4] = dt
        q_a, q_w = self.cfg.accel_std ** 2, self.cfg.yaw_accel_std ** 2
        Q = np.zeros((5, 5))
        Q[3, 3] = q_a * dt ** 2
        Q[4, 4] = q_w * dt ** 2
        Q[0, 0] = Q[1, 1] = 0.25 * q_a * dt ** 4
        Q[2, 2] = 0.25 * q_w * dt ** 4
        self.P = F @ self.P @ F.T + Q

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, angle_rows: tuple = ()) -> None:
        y = z - H @ self.x
        for r in angle_rows:
            y[r] = wrap_angle(y[r])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        self.P = (np.eye(5) - K @ H) @ self.P

    def _advance(self, t: float) -> None:
        if self.t is not None:
            self._predict(t - self.t)
        self.t = t

    # --------------------------------------------------------------- inputs

    def update_pose(self, t: float, x: float, y: float, yaw: float) -> None:
        if self.x is None:
            self.x = np.array([x, y, wrap_angle(yaw), 0.0, 0.0])
            self.P = np.diag([self.cfg.pos_std ** 2] * 2 + [self.cfg.yaw_std ** 2, 1.0, 1.0])
            self.t = self.last_pose_t = t
            return
        if t < self.t:          # out-of-order sample
            return
        self._advance(t)
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.diag([self.cfg.pos_std ** 2] * 2 + [self.cfg.yaw_std ** 2])
        self._update(np.array([x, y, yaw]), H, R, angle_rows=(2,))
        self.last_pose_t = t

    def update_speed(self, t: float, v: float) -> None:
        if self.x is None or t < self.t:
            return
        self._advance(t)
        H = np.zeros((1, 5))
        H[0, 3] = 1.0
        self._update(np.array([v]), H, np.array([[self.cfg.speed_std ** 2]]))

    # --------------------------------------------------------------- output

    def valid(self, t: float) -> bool:
        return self.last_pose_t is not None and (t - self.last_pose_t) <= self.cfg.stale_after

    def predict_to(self, t: float) -> VehicleState | None:
        """State extrapolated to ``t`` (does not modify the filter)."""
        if self.x is None or not self.valid(t):
            return None
        dt = max(t - self.t, 0.0)
        x, y, yaw, v, w = self.x
        return VehicleState(
            t=t,
            x=float(x + v * np.cos(yaw) * dt),
            y=float(y + v * np.sin(yaw) * dt),
            yaw=float(wrap_angle(yaw + w * dt)),
            v=float(v),
            yaw_rate=float(w),
        )
