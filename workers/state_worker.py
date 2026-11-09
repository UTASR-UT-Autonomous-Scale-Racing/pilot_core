"""
StateWorker: PoseBuffer poses -> control.PoseEKF -> StateBuffer at a fixed rate.
A stale estimate (no poses for ``stale_after`` s) is not published, so the actuator
watchdog takes the car to neutral.

With a rear_imu_buffer (the Arduino's ICM-20948, IMUWorker's raw output) the estimator is
control.ImuEKF instead: driven by the IMU, corrected by the poses. The IMU's axes are
turned into the car's (x forward, y left, z up) by rear_imu.rotation, the pose's point
(the camera's, for SLAM) is moved to the IMU by rear_imu.lever_m, and the published state
is moved back: controllers see the same point as with PoseEKF, now with the gyro's yaw
rate, through SLAM gaps, and with poses applied at their capture time.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from control.estimator import EstimatorConfig, ImuEKF, ImuEstimatorConfig, PoseEKF
from control.types import VehicleState

from .base_worker import BaseWorker
from .bev import pose_to_bev


class StateWorker(BaseWorker):
    """BEV vehicle state from PoseBuffer poses.

    Config:
      pose_buffer     camera-to-world (SLAM) or vehicle-to-world (sim) poses
      output_buffer   StateBuffer
      bev_axes, forward_axis   passed to workers.bev.pose_to_bev
      rate_hz         publish rate, default 50
      use_pose_speed  also fuse PoseBuffer's speed (sim only; SLAM has none), default off
      estimator       dict of EstimatorConfig overrides
      rear_imu_buffer RawIMUBuffer (IMUWorker raw output): use ImuEKF
      rear_imu        dict: rotation (3x3, sensor -> car axes, default identity),
                      lever_m ([dx, dy], the IMU's position from the pose's point in the car
                      frame, x forward, y left; default [0, 0]), delay_s (subtracted from the
                      IMU's stamps: its filter and read, default 0), estimator (dict of
                      ImuEstimatorConfig overrides)
    """

    def __init__(self, name: str = "state", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._last_pose_seq = 0
        self._next_pub = 0.0
        self._was_valid = False

    def setup(self) -> None:
        self._pose_buf = self.config["pose_buffer"]
        self._state_buf = self.config["output_buffer"]
        self._bev_axes = tuple(self.config.get("bev_axes", [0, 2]))
        self._forward_axis = int(self.config.get("forward_axis", 2))
        self._period = 1.0 / float(self.config.get("rate_hz", 50.0))
        self._use_pose_speed = bool(self.config.get("use_pose_speed", False))
        self._imu_buf = self.config.get("rear_imu_buffer")
        self._imu_seq = 0
        if self._imu_buf is not None:
            ri = self.config.get("rear_imu", {}) or {}
            self._R = np.asarray(ri.get("rotation", np.eye(3)), dtype=np.float64).reshape(3, 3)
            if abs(abs(np.linalg.det(self._R)) - 1.0) > 1e-3 or not np.allclose(self._R @ self._R.T, np.eye(3), atol=1e-3):
                raise ValueError(f"rear_imu.rotation is not a rotation: {self._R.tolist()}")
            self._lever = np.asarray(ri.get("lever_m", [0.0, 0.0]), dtype=np.float64)
            self._delay = float(ri.get("delay_s", 0.0))
            self._ekf = ImuEKF(ImuEstimatorConfig(**(ri.get("estimator") or {})))
        else:
            self._ekf = PoseEKF(EstimatorConfig(**self.config.get("estimator", {})))
        self.logger.info(
            f"StateWorker ready ({'ImuEKF on the rear IMU' if self._imu_buf is not None else 'PoseEKF'}, "
            f"bev_axes={self._bev_axes}, forward_axis={self._forward_axis}, "
            f"rate={1 / self._period:.0f} Hz, pose_speed={self._use_pose_speed})"
        )

    # the pose's point (camera) <-> the IMU's, in the BEV plane
    def _shift(self, x: float, y: float, yaw: float, sign: float) -> tuple:
        c, s = np.cos(yaw), np.sin(yaw)
        dx, dy = self._lever
        return x + sign * (c * dx - s * dy), y + sign * (s * dx + c * dy)

    def _drain_imu(self) -> None:
        for header, sample in self._imu_buf.read_since(self._imu_seq):
            self._imu_seq = header.seq_no
            acc = self._R @ sample[:3].astype(np.float64)
            gyro = self._R @ sample[3:].astype(np.float64)
            self._ekf.update_imu(header.produced_ts - self._delay, acc[0], acc[1], gyro[2])

    def step(self) -> None:
        if self._imu_buf is not None:
            self._drain_imu()
        result = self._pose_buf.read_latest()
        if result is not None:
            header, pose, _, speed = result
            if header.seq_no > self._last_pose_seq:
                self._last_pose_seq = header.seq_no
                (x, y), yaw = pose_to_bev(pose, self._bev_axes, self._forward_axis)
                if self._imu_buf is not None:
                    x, y = self._shift(float(x), float(y), yaw, +1.0)
                self._ekf.update_pose(header.produced_ts, float(x), float(y), yaw)
                if self._use_pose_speed and hasattr(self._ekf, "update_speed"):
                    self._ekf.update_speed(header.produced_ts, float(speed))

        now = time.monotonic()
        if now >= self._next_pub:
            self._next_pub = now + self._period
            state = self._ekf.predict_to(now)
            if state is not None and self._imu_buf is not None:
                x, y = self._shift(state.x, state.y, state.yaw, -1.0)
                state = VehicleState(t=state.t, x=float(x), y=float(y), yaw=state.yaw, v=state.v,
                                     yaw_rate=state.yaw_rate)
            valid = state is not None
            if valid:
                self._state_buf.write(state)
            if valid != self._was_valid:
                self._was_valid = valid
                if valid:
                    self.logger.info("State estimate valid")
                else:
                    self.logger.warning("State estimate stale — no recent poses")
        time.sleep(0.002)

    def teardown(self) -> None:
        self.logger.info("StateWorker shutting down")
