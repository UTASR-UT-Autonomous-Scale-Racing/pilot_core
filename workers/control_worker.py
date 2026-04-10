"""
ControlWorker — operates in one of two modes:

1. **Neural mode** (default): reads segmentation masks + IMU, runs a learned
   control model, and writes normalised steering/throttle into ControlBuffer.

2. **PID mode** (``mode="pid"``): reads live pose from PoseBuffer and track
   boundaries from TrackBoundaryBuffer, runs the TrackPIDController, and
   writes the result to ControlBuffer.

Output contract: every value written to ControlBuffer is normalised to
``[0, 1]`` (steering 0 = full left, 0.5 = centre, 1 = full right; throttle
0 = neutral, 1 = max forward), matching ``pilot_controller.Controller``.
Controllers that work in signed ``[-1, 1]`` space are converted here.
"""

from __future__ import annotations

import inspect
import multiprocessing as mp
import time
from typing import Any, Dict, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import ControlBuffer, IMUBuffer, PoseBuffer, SegmentationBuffer

# Optional — only needed in PID mode
try:
    from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer
except ImportError:
    TrackBoundaryBuffer = None  # type: ignore[misc]


def signed_to_unit(steering: float, throttle: float) -> tuple[float, float]:
    """Convert signed controller output to the ControlBuffer ``[0, 1]`` contract.

    Steering ``[-1, 1]`` maps linearly onto ``[0, 1]``.  Throttle is clipped
    to ``[0, 1]`` because the PWM mapper has no reverse range.
    """
    steer_unit = float(np.clip((steering + 1.0) / 2.0, 0.0, 1.0))
    throttle_unit = float(np.clip(throttle, 0.0, 1.0))
    return steer_unit, throttle_unit


def pose_to_bev(pose: np.ndarray) -> tuple[np.ndarray, float]:
    """Extract BEV position ``(X, Z)`` and heading from a camera-to-world pose.

    Heading is the camera's forward axis (third rotation column) projected
    onto the X-Z plane, measured from +X towards +Z — the same convention
    the track controllers use for bearings (``atan2(dz, dx)``).
    """
    position = np.array([pose[0, 3], pose[2, 3]], dtype=np.float32)
    heading = float(np.arctan2(pose[2, 2], pose[0, 2]))
    return position, heading


class ControlWorker(BaseWorker):
    """Worker process for control-policy inference (neural or PID).

    Config kwargs
    -------------
    mode : str
        ``"neural"`` (default) or ``"pid"``.

    Neural-mode kwargs:
        input_buffers : dict[str, buffer]
            Must contain ``"segments"`` and ``"imu"``.  May contain ``"pose"``.
        output_buffer : ControlBuffer
        model_config : str
        device : str

    PID-mode kwargs:
        pose_buffer : PoseBuffer
        boundary_buffer : TrackBoundaryBuffer
        output_buffer : ControlBuffer
        mapping_done_event : mp.Event
        pid_config : dict  (kp_steer, ki_steer, kd_steer, target_throttle, noise_std)
    """

    def __init__(self, name: str = "control", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._mode: str = "neural"
        # Neural-mode state
        self._seg_buf: SegmentationBuffer | None = None
        self._imu_buf: IMUBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._ctrl_buf: ControlBuffer | None = None
        self._model = None
        self._model_takes_timestamp = False
        self._last_seg_seq = 0
        # PID-mode state
        self._boundary_buf: Optional[Any] = None
        self._pid = None
        self._mapping_done_event: Optional[mp.Event] = None
        self._outer_line: Optional[np.ndarray] = None
        self._inner_line: Optional[np.ndarray] = None
        self._last_pose_seq = 0

    def setup(self) -> None:
        self._mode = self.config.get("mode", "neural")
        self._ctrl_buf = self.config["output_buffer"]

        if self._mode == "pid":
            self._setup_pid()
        else:
            self._setup_neural()

    def _setup_neural(self) -> None:
        bufs: Dict[str, Any] = self.config["input_buffers"]
        self._seg_buf = bufs["segments"]
        self._imu_buf = bufs["imu"]
        self._pose_buf = bufs.get("pose")

        model_config_path = self.config["model_config"]
        from utils import load_model_from_config

        self._model, class_name = load_model_from_config(model_config_path)
        # Some pilot_controller models take (state, mask, timestamp), others
        # only (state, mask) — detect once rather than per frame.
        params = inspect.signature(self._model.get_action).parameters
        self._model_takes_timestamp = "timestamp" in params
        self.logger.info(f"Loaded control model: {class_name}")

        device = self.config.get("device", "cpu")
        self.logger.info(f"ControlWorker ready — neural mode (device={device})")

    def _setup_pid(self) -> None:
        from .track_pid_controller import TrackPIDController

        self._pose_buf = self.config["pose_buffer"]
        self._boundary_buf = self.config.get("boundary_buffer")
        self._mapping_done_event = self.config.get("mapping_done_event")

        pid_cfg = self.config.get("pid_config", {})
        self._pid = TrackPIDController(
            kp_steer=pid_cfg.get("kp_steer", 1.0),
            ki_steer=pid_cfg.get("ki_steer", 0.0),
            kd_steer=pid_cfg.get("kd_steer", 0.1),
            target_throttle=pid_cfg.get("target_throttle", 0.3),
            noise_std=pid_cfg.get("noise_std", 0.05),
        )
        self.logger.info("ControlWorker ready — PID mode")

    def step(self) -> None:
        if self._mode == "pid":
            self._step_pid()
        else:
            self._step_neural()

    # --- Neural mode ---

    def _step_neural(self) -> None:
        seg_result = self._seg_buf.read_latest()
        if seg_result is None:
            time.sleep(0.001)
            return

        seg_header, mask = seg_result
        if seg_header.seq_no <= self._last_seg_seq:
            time.sleep(0.001)
            return
        self._last_seg_seq = seg_header.seq_no

        imu_result = self._imu_buf.read_latest()
        imu_vector = (
            imu_result[1] if imu_result is not None else np.zeros(9, dtype=np.float32)
        )

        state = imu_vector.tolist()
        if self._model_takes_timestamp:
            result = self._model.get_action(
                state, mask, timestamp=seg_header.produced_ts
            )
        else:
            result = self._model.get_action(state, mask)

        if result is not None:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                steering, throttle = float(result[0]), float(result[1])
            else:
                steering, throttle = 0.5, 0.0
            self._ctrl_buf.write(
                float(np.clip(steering, 0.0, 1.0)),
                float(np.clip(throttle, 0.0, 1.0)),
            )

    # --- PID mode ---

    def _step_pid(self) -> None:
        # Wait for mapping to complete
        if (
            self._mapping_done_event is not None
            and not self._mapping_done_event.is_set()
        ):
            time.sleep(0.1)
            return

        # Read boundaries once (they don't change after extraction)
        if self._outer_line is None and self._boundary_buf is not None:
            bnd_result = self._boundary_buf.read_latest()
            if bnd_result is not None:
                _, self._outer_line, self._inner_line = bnd_result
                self.logger.info(
                    f"Loaded boundaries: outer={len(self._outer_line)}, "
                    f"inner={len(self._inner_line)}"
                )

        if self._outer_line is None or self._inner_line is None:
            time.sleep(0.05)
            return

        # Read latest pose
        pose_result = self._pose_buf.read_latest()
        if pose_result is None:
            time.sleep(0.005)
            return

        pose_header, pose, _ = pose_result
        if pose_header.seq_no <= self._last_pose_seq:
            time.sleep(0.005)
            return
        self._last_pose_seq = pose_header.seq_no

        current_bev, heading = pose_to_bev(pose)

        # PID step
        steering, throttle = self._pid.step(
            current_bev, heading, self._outer_line, self._inner_line
        )

        # PID works in signed space; convert to the ControlBuffer contract
        steer_unit, throttle_unit = signed_to_unit(steering, throttle)
        self._ctrl_buf.write(steer_unit, throttle_unit)

        if self._last_pose_seq % 30 == 0:
            self.logger.debug(
                f"steering={steering:+.3f} throttle={throttle:+.3f} "
                f"-> buffer ({steer_unit:.3f}, {throttle_unit:.3f})"
            )

    def teardown(self) -> None:
        self.logger.info(f"ControlWorker ({self._mode} mode) shutting down")
