"""
EKF on a unicycle model, state [x, y, yaw, v, yaw_rate], fed by SLAM poses (~15 Hz)
and optional direct speed. No wheel encoders, so speed otherwise comes from poses.

Ref: S. Thrun, W. Burgard, D. Fox, "Probabilistic Robotics", MIT Press, 2005, ch. 3.
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
    accel_std: float = 2.0         # m/s^2, process noise on v
    yaw_accel_std: float = 4.0     # rad/s², process noise on yaw rate
    stale_after: float = 0.5       # s without a pose before the estimate is invalid
    jump_reset_m: float | None = None   # a pose this far from the prediction restarts the filter
                                        # there, standing still (a sim respawn after a collision)


class PoseEKF:
    def __init__(self, cfg: EstimatorConfig | None = None) -> None:
        self.cfg = cfg or EstimatorConfig()
        self.x: np.ndarray | None = None          # [x, y, yaw, v, w]
        self.P = np.eye(5)
        self.t: float | None = None
        self.last_pose_t: float | None = None
        self.jumps = 0

    @property
    def initialized(self) -> bool:
        return self.x is not None

    def reset(self) -> None:
        self.x = None
        self.t = None
        self.last_pose_t = None

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

    def update_pose(self, t: float, x: float, y: float, yaw: float) -> None:
        if self.x is None:
            self.x = np.array([x, y, wrap_angle(yaw), 0.0, 0.0])
            self.P = np.diag([self.cfg.pos_std ** 2] * 2 + [self.cfg.yaw_std ** 2, 1.0, 1.0])
            self.t = self.last_pose_t = t
            return
        if t < self.t:  # out of order, drop it
            return
        self._advance(t)
        if self.cfg.jump_reset_m is not None and np.hypot(x - self.x[0], y - self.x[1]) > self.cfg.jump_reset_m:
            # a respawn, not motion: differencing it would give a huge speed and yaw rate
            self.x = None
            self.jumps += 1
            self.update_pose(t, x, y, yaw)
            return
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

    def valid(self, t: float) -> bool:
        return self.last_pose_t is not None and (t - self.last_pose_t) <= self.cfg.stale_after

    def predict_to(self, t: float) -> VehicleState | None:
        """Extrapolate to t without touching the filter state."""
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


@dataclass
class ImuEstimatorConfig:
    pos_std: float = 0.03            # m, pose position measurement noise
    yaw_std: float = 0.02            # rad, pose yaw measurement noise
    acc_std: float = 0.5             # m/s^2, accelerometer noise (white + model error)
    gyro_std: float = 0.02           # rad/s, gyro noise
    acc_bias_walk: float = 0.05      # m/s^2/sqrt(s)
    gyro_bias_walk: float = 0.002    # rad/s/sqrt(s)
    vy_std: float | None = 0.1       # m/s: pseudo-measurement vy = 0 at the IMU (a rear axle barely slips); None = off
    history_s: float = 1.0           # s of IMU kept to apply late poses at their capture time
    stale_after: float = 1.0         # s without a pose before the estimate is invalid
    gate_chi2: float | None = 16.3   # reject a pose whose innovation exceeds this (3 dof, 99.9 %); None = off
    gate_reset_s: float = 2.0        # s of rejected poses after which the filter is re-anchored to SLAM


class ImuEKF:
    """Planar EKF driven by an IMU on the car (update_imu), corrected by SLAM poses.

    State [x, y, yaw, vx, vy, b_gz, b_ax, b_ay]: the IMU point's position and heading in the
    pose frame, its velocity in the body frame (vx forward, vy left: vy is sideslip), and the
    gyro-z / accelerometer biases. IMU samples are in the body frame (x forward, y left,
    z up; level mount, so only ax, ay and gz are used):

        yaw' = gz - b_gz          vx' = ax - b_ax + r vy          vy' = ay - b_ay - r vx

    Poses usually arrive after later IMU samples (SLAM latency): a pose is applied at its
    capture time and the IMU since then is replayed. A pose that disagrees with the IMU's
    prediction beyond gate_chi2 is rejected (SLAM's first poses after a loss can be off by
    a metre); if poses keep disagreeing for gate_reset_s the filter is re-anchored to SLAM.
    Same interface as PoseEKF (update_pose, predict_to, valid), plus update_imu.
    """

    N = 8

    def __init__(self, cfg: ImuEstimatorConfig | None = None) -> None:
        self.cfg = cfg or ImuEstimatorConfig()
        self.reset()

    def reset(self) -> None:
        self.x: np.ndarray | None = None
        self.P = np.eye(self.N)
        self.t: float | None = None
        self.last_pose_t: float | None = None
        self._u: np.ndarray | None = None              # last IMU input [ax, ay, gz]
        self._hist: list = []                          # (t, x, P, u) after each IMU step
        self._pending: list = []                       # IMU samples before the first pose
        self._reject_since: float | None = None
        self.rejected = 0

    @property
    def initialized(self) -> bool:
        return self.x is not None

    # model

    def _step(self, x: np.ndarray, P: np.ndarray, u: np.ndarray | None, dt: float):
        px, py, yaw, vx, vy, bg, bax, bay = x
        ax, ay, gz = u if u is not None else (0.0, 0.0, bg)     # no IMU yet (first pose came first)
        r = gz - bg
        c, s = np.cos(yaw), np.sin(yaw)
        xn = np.array([
            px + (c * vx - s * vy) * dt,
            py + (s * vx + c * vy) * dt,
            wrap_angle(yaw + r * dt),
            vx + (ax - bax + r * vy) * dt,
            vy + (ay - bay - r * vx) * dt,
            bg, bax, bay,
        ])
        F = np.eye(self.N)
        F[0, 2], F[0, 3], F[0, 4] = (-s * vx - c * vy) * dt, c * dt, -s * dt
        F[1, 2], F[1, 3], F[1, 4] = (c * vx - s * vy) * dt, s * dt, c * dt
        F[2, 5] = -dt
        F[3, 4], F[3, 5], F[3, 6] = r * dt, -vy * dt, -dt
        F[4, 3], F[4, 5], F[4, 7] = -r * dt, vx * dt, -dt
        cfg = self.cfg
        Q = np.zeros(self.N)
        Q[2] = (cfg.gyro_std * dt) ** 2
        Q[3] = Q[4] = (cfg.acc_std * dt) ** 2
        Q[5] = cfg.gyro_bias_walk ** 2 * dt
        Q[6] = Q[7] = cfg.acc_bias_walk ** 2 * dt
        return xn, F @ P @ F.T + np.diag(Q)

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray, angle_rows: tuple = ()) -> None:
        y = z - H @ self.x
        for r in angle_rows:
            y[r] = wrap_angle(y[r])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        I_KH = np.eye(self.N) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def _slip_update(self) -> None:
        if self.cfg.vy_std is not None:
            H = np.zeros((1, self.N))
            H[0, 4] = 1.0
            self._update(np.zeros(1), H, np.array([[self.cfg.vy_std ** 2]]))

    def _gate(self, t: float, z: np.ndarray, H: np.ndarray, R: np.ndarray) -> str:
        """For a pose at time t against the current (x, P), at that time: "accept", "reject",
        or "reset" (poses rejected for gate_reset_s: the filter is re-anchored to this one)."""
        if self.cfg.gate_chi2 is None:
            return "accept"
        y = z - H @ self.x
        y[2] = wrap_angle(y[2])
        d2 = float(y @ np.linalg.solve(H @ self.P @ H.T + R, y))
        if d2 <= self.cfg.gate_chi2:
            self._reject_since = None
            return "accept"
        if self._reject_since is None:
            self._reject_since = t
        if t - self._reject_since < self.cfg.gate_reset_s:
            self.rejected += 1
            return "reject"
        self._reject_since = None
        self.x[:3] = z
        self.P[:3, :] = 0.0
        self.P[:, :3] = 0.0
        self.P[:3, :3] = R
        return "reset"

    # inputs

    def update_imu(self, t: float, ax: float, ay: float, gz: float) -> None:
        u = np.array([ax, ay, gz], dtype=np.float64)
        if self.x is None:
            self._pending.append((t, u))
            del self._pending[:-200]
            return
        if t <= self.t:
            return
        self.x, self.P = self._step(self.x, self.P, self._u if self._u is not None else u, t - self.t)
        self._slip_update()
        self.t, self._u = t, u
        self._hist.append((t, self.x.copy(), self.P.copy(), u))
        while self._hist and self._hist[0][0] < t - self.cfg.history_s:
            self._hist.pop(0)

    def update_pose(self, t: float, x: float, y: float, yaw: float) -> None:
        cfg = self.cfg
        z = np.array([x, y, wrap_angle(yaw)])
        H = np.zeros((3, self.N))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        R = np.diag([cfg.pos_std ** 2] * 2 + [cfg.yaw_std ** 2])
        if self.x is None:
            self.x = np.array([x, y, wrap_angle(yaw), 0.0, 0.0, 0.0, 0.0, 0.0])
            self.P = np.diag([cfg.pos_std ** 2] * 2 + [cfg.yaw_std ** 2, 4.0, 1.0, 0.01 ** 2, 0.2 ** 2, 0.2 ** 2])
            self.t = self.last_pose_t = t
            self._u = next((u for tu, u in reversed(self._pending) if tu <= t), None)
            later = [(tu, u) for tu, u in self._pending if tu > t]
            self._pending = []
            self._hist = [(t, self.x.copy(), self.P.copy(), self._u)]
            for tu, u in later:
                self.update_imu(tu, *u)
            return
        if t >= self.t:                                  # no newer IMU yet: coast to t
            if self._u is not None:
                self.x, self.P = self._step(self.x, self.P, self._u, t - self.t)
            self.t = t
            if self._gate(t, z, H, R) == "accept":
                self._update(z, H, R, angle_rows=(2,))
            self._hist = [(t, self.x.copy(), self.P.copy(), self._u)]
            self.last_pose_t = t
            return
        # late pose: rewind to the last IMU step at or before t, apply it there, replay the rest
        k = max((i for i, h in enumerate(self._hist) if h[0] <= t), default=None)
        if k is None:                                    # older than the history: drop it
            return
        t0, x0, P0, u0 = self._hist[k]
        later = self._hist[k + 1:]
        saved = (self.x, self.P, self.t, self._u)
        self.x, self.P = self._step(x0, P0, u0, t - t0) if t > t0 else (x0.copy(), P0.copy())
        verdict = self._gate(t, z, H, R)
        if verdict == "reject":                          # as if it never came
            self.x, self.P, self.t, self._u = saved
            return
        if verdict == "accept":
            self._update(z, H, R, angle_rows=(2,))
        self.t, self._u = t, u0
        self._hist = self._hist[:k + 1]
        for tu, _, _, u in later:
            self.x, self.P = self._step(self.x, self.P, self._u, tu - self.t)
            self._slip_update()
            self.t, self._u = tu, u
            self._hist.append((tu, self.x.copy(), self.P.copy(), u))
        self.last_pose_t = max(self.last_pose_t or t, t)

    def valid(self, t: float) -> bool:
        return self.last_pose_t is not None and (t - self.last_pose_t) <= self.cfg.stale_after

    @property
    def sideslip(self) -> float:
        """rad, angle of the IMU point's velocity to the body x axis"""
        return float(np.arctan2(self.x[4], self.x[3])) if self.x is not None and abs(self.x[3]) > 0.3 else 0.0

    def predict_to(self, t: float) -> VehicleState | None:
        """Extrapolate to t (constant body velocity and yaw rate) without touching the filter."""
        if self.x is None or not self.valid(t):
            return None
        x = self.x.copy()
        if t > self.t and self._u is not None:
            x, _ = self._step(x, self.P, self._u, t - self.t)
        r = float((self._u[2] if self._u is not None else 0.0) - x[5])
        return VehicleState(t=t, x=float(x[0]), y=float(x[1]), yaw=float(x[2]),
                            v=float(np.hypot(x[3], x[4]) * np.sign(x[3])), yaw_rate=r)
