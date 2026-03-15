"""
ControlWorker — operates in one of two modes:

1. **Neural mode** (default): reads segmentation masks + IMU, runs a learned
   control model, and writes normalised steering/throttle into ControlBuffer.

2. **PID mode** (``mode="pid"``): reads live pose from PoseBuffer and track
   boundaries from TrackBoundaryBuffer, runs the TrackPIDController, prints
   normalised (steering, throttle) to terminal, and writes to ControlBuffer.
"""

from __future__ import annotations

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

    #  Neural mode ─

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

        if mask is None or getattr(mask, "size", 0) == 0:
            time.sleep(0.001)
            return

        imu_result = self._imu_buf.read_latest()
        if imu_result is not None:
            imu_vector = np.asarray(imu_result[1], dtype=np.float32)
        else:
            imu_vector = np.zeros(16, dtype=np.float32)
            imu_vector[0] = 1.0

        if imu_vector.shape[0] < 16:
            padded = np.zeros(16, dtype=np.float32)
            padded[: imu_vector.shape[0]] = imu_vector
            imu_vector = padded

        # Keep quaternion valid for downstream math.
        q = imu_vector[:4]
        q_norm = float(np.linalg.norm(q))
        if q_norm <= 1e-6:
            imu_vector[0:4] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            imu_vector[0:4] = q / q_norm

        state = imu_vector.tolist()
        timestamp = float(seg_header.produced_ts)
        try:
            result = self._model.get_action(state, mask, timestamp)
        except TypeError:
            # Backward compatibility with older models that only accept
            # (state, mask).
            result = self._model.get_action(state, mask)

        if result is not None:
            if isinstance(result, (tuple, list)) and len(result) >= 2:
                steering, throttle = float(result[0]), float(result[1])
            else:
                steering, throttle = 0.5, 0.0
            self._ctrl_buf.write(steering, throttle)

    #  PID mode 

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

        # Extract BEV position and heading from 4×4 pose
        current_bev = np.array([pose[0, 3], pose[2, 3]], dtype=np.float32)
        heading = np.arctan2(pose[0, 2], pose[2, 2])

        # PID step
        steering, throttle = self._pid.step(
            current_bev, heading, self._outer_line, self._inner_line
        )

        # Print to terminal
        print(f"steering={steering:+.4f}  throttle={throttle:+.4f}")

        # Write to buffer
        self._ctrl_buf.write(steering, throttle)

    def teardown(self) -> None:
        self.logger.info(f"ControlWorker ({self._mode} mode) shutting down")
