"""
StateWorker: runs PoseBuffer poses through control.PoseEKF and publishes the
state to StateBuffer at a fixed rate.

If poses stop coming in (SLAM lost tracking, say) the estimate goes stale
after ``stale_after`` seconds and we stop publishing. The controller then has
nothing to act on, stops sending commands, and the actuator watchdog takes
the car to neutral.
"""

from __future__ import annotations

import time
from typing import Any

from control.estimator import EstimatorConfig, PoseEKF

from .base_worker import BaseWorker
from .bev import pose_to_bev


class StateWorker(BaseWorker):
    """BEV vehicle state from PoseBuffer poses.

    Config:
      pose_buffer     camera-to-world (SLAM) or vehicle-to-world (sim) poses
      output_buffer   StateBuffer
      bev_axes, forward_axis   passed to workers.bev.pose_to_bev
      rate_hz         publish rate, default 50
      use_pose_speed  also fuse PoseBuffer's speed field. Only the simulator
                      fills it in (SLAM has no speed), hence off by default.
      estimator       dict of EstimatorConfig overrides
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
        self._ekf = PoseEKF(EstimatorConfig(**self.config.get("estimator", {})))
        self.logger.info(
            f"StateWorker ready (bev_axes={self._bev_axes}, forward_axis={self._forward_axis}, "
            f"rate={1 / self._period:.0f} Hz, pose_speed={self._use_pose_speed})"
        )

    def step(self) -> None:
        result = self._pose_buf.read_latest()
        if result is not None:
            header, pose, _, speed = result
            if header.seq_no > self._last_pose_seq:
                self._last_pose_seq = header.seq_no
                (x, y), yaw = pose_to_bev(pose, self._bev_axes, self._forward_axis)
                self._ekf.update_pose(header.produced_ts, float(x), float(y), yaw)
                if self._use_pose_speed:
                    self._ekf.update_speed(header.produced_ts, float(speed))

        now = time.time()
        if now >= self._next_pub:
            self._next_pub = now + self._period
            state = self._ekf.predict_to(now)
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
