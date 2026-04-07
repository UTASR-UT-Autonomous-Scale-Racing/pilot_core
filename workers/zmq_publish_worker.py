"""
ZMQPublishWorker — Publishes telemetry data from various buffers back to
the simulator for visualization and logging.

Uses topic-based publishing to allow selective subscription.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import ControlBuffer, PoseBuffer, SegmentationBuffer

try:
    from shared_buffer import TrackBoundaryBuffer
except ImportError:
    TrackBoundaryBuffer = None

try:
    from shared_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None

try:
    import zmq
except ImportError:
    zmq = None


class ZMQPublishWorker(BaseWorker):
    """Worker that publishes telemetry data with topic prefixes.

    Config kwargs
    -------------
    port : int
        Port to bind publisher (default: 5557)
    control_buffer : ControlBuffer, optional
        Control commands for echo/logging
    pose_buffer : PoseBuffer, optional
        Vehicle poses for trajectory visualization
    seg_buffer : SegmentationBuffer, optional
        Segmentation masks for overlay
    boundary_buffer : TrackBoundaryBuffer, optional
        Track boundaries for map visualization
    publish : dict
        Configuration for what to publish:
            control: bool
            pose: bool
            segmentation: bool
            boundary: bool
    """

    def __init__(self, name: str = "zmq_publish", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._socket = None
        self._context = None
        self._ctrl_buf: Optional[ControlBuffer] = None
        self._pose_buf: Optional[PoseBuffer] = None
        self._seg_buf: Optional[SegmentationBuffer] = None
        self._boundary_buf: Optional[Any] = None
        self._pc_buf: Optional[Any] = None
        self._publish_config: dict = {}
        self._last_seq = {
            'control': 0,
            'pose': 0,
            'segmentation': 0,
            'boundary': 0,
            'pointcloud': 0,
        }
        self._frame_count = 0

    def setup(self) -> None:
        if zmq is None:
            raise ImportError("pyzmq is required for ZMQPublishWorker")

        port = self.config.get("port", 5557)

        # Initialize ZMQ publisher
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 10)
        self._socket.setsockopt(zmq.LINGER, 1000)

        try:
            self._socket.bind(f"tcp://*:{port}")
            self.logger.info(f"ZMQ telemetry publisher bound to port {port}")
        except Exception as e:
            self.logger.error(f"Failed to bind to port {port}: {e}")
            raise

        # Get buffer references
        self._ctrl_buf = self.config.get("control_buffer")
        self._pose_buf = self.config.get("pose_buffer")
        self._seg_buf = self.config.get("seg_buffer")
        self._boundary_buf = self.config.get("boundary_buffer")
        self._pc_buf = self.config.get("pointcloud_buffer")

        # Get publish configuration
        self._publish_config = self.config.get("publish", {
            "control": True,
            "pose": False,
            "segmentation": False,
            "boundary": False,
        })

        # Allow sockets to initialize
        time.sleep(0.5)
        self.logger.info(f"Publishing: {list(k for k, v in self._publish_config.items() if v)}")

    def step(self) -> None:
        self._frame_count += 1

        # High-frequency data (every frame)
        if self._publish_config.get("control", False) and self._ctrl_buf:
            self._publish_control()

        if self._publish_config.get("pose", False) and self._pose_buf:
            self._publish_pose()

        # Medium-frequency data (every few frames)
        if self._frame_count % 2 == 0:
            if self._publish_config.get("segmentation", False) and self._seg_buf:
                self._publish_segmentation()

        # Low-frequency data (every few frames)
        if self._frame_count % 2 == 0:
            if self._publish_config.get("boundary", False) and self._boundary_buf:
                self._publish_boundary()
            if self._publish_config.get("pointcloud", False) and self._pc_buf:
                self._publish_pointcloud()
        
        time.sleep(0.01)  # Small delay to avoid busy loop

    def _publish_control(self) -> None:
        """Publish control commands (for logging/echo)."""
        try:
            result = self._ctrl_buf.read_latest()
            # print(f"Read control buffer: {result is not None}")
            if result is None:
                return

            header, steering_norm, throttle_norm = result
            if header.seq_no <= self._last_seq['control']:
                return
            self._last_seq['control'] = header.seq_no

            # Convert normalized [0, 1] to simulator range [-1, 1]
            steering = (steering_norm * 2.0) - 1.0
            throttle = (throttle_norm * 2.0) - 1.0

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'steering': float(steering),
                'throttle': float(throttle),
            })
            
            # print(f"Publishing control: {message}")

            self._socket.send_string(f"control {message}", zmq.NOBLOCK)

        except Exception as e:
            self.logger.debug(f"Control publish failed: {e}")

    def _publish_pose(self) -> None:
        """Publish vehicle pose and trajectory."""
        try:
            result = self._pose_buf.read_latest()
            # print(f"Read pose buffer: {result is not None}")
            if result is None:
                return

            header, pose, _, _ = result
            if header.seq_no <= self._last_seq['pose']:
                return
            self._last_seq['pose'] = header.seq_no

            # Extract position and orientation
            position = pose[0:3, 3].tolist()

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'position': position,
                'pose_matrix': pose.flatten().tolist(),
            })

            # print(f"Publishing pose: {message}")

            self._socket.send_string(f"telemetry {message}", zmq.NOBLOCK)

        except Exception as e:
            self.logger.debug(f"Pose publish failed: {e}")

    def _publish_segmentation(self) -> None:
        """Publish segmentation masks."""
        try:
            result = self._seg_buf.read_latest()
            # print(f"[seg] buf read: {result is not None}")
            if result is None:
                return

            header, mask = result
            if header.seq_no <= self._last_seq['segmentation']:
                return
            self._last_seq['segmentation'] = header.seq_no

            # Encode mask as base64 (compressed)
            mask_bytes = mask.astype(np.uint8).tobytes()
            mask_b64 = base64.b64encode(mask_bytes).decode('utf-8')

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'mask': mask_b64,
                'shape': mask.shape,
                'dtype': 'uint8',
            })

            # print(f"Publishing segmentation: seq={header.seq_no}, shape={mask.shape}")

            self._socket.send_string(f"segmentation {message}", zmq.NOBLOCK)

        except Exception as e:
            import traceback
            self.logger.error(f"Segmentation publish failed: {e}")
            traceback.print_exc()

    def _publish_boundary(self) -> None:
        """Publish track boundaries."""
        try:
            result = self._boundary_buf.read_latest()
            # print(f"Read boundary buffer: {result is not None}")
            if result is None:
                return

            header, outer_line, inner_line = result
            if header.seq_no <= self._last_seq['boundary']:
                return
            self._last_seq['boundary'] = header.seq_no

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'outer_line': outer_line.tolist(),
                'inner_line': inner_line.tolist(),
                'complete': True,
            })

            print(f"Publishing boundary: {message}")

            self._socket.send_string(f"map {message}", zmq.NOBLOCK)

            self.logger.info("Published track boundaries to simulator")

        except Exception as e:
            self.logger.debug(f"Boundary publish failed: {e}")

    def _publish_pointcloud(self) -> None:
        """Publish SLAM point cloud as base64-encoded float32 XYZ array."""
        try:
            result = self._pc_buf.read_latest()
            if result is None:
                return

            header, pts = result
            if header.seq_no <= self._last_seq['pointcloud']:
                return
            self._last_seq['pointcloud'] = header.seq_no

            pts_b64 = base64.b64encode(pts.astype(np.float32).tobytes()).decode('utf-8')

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'points': pts_b64,
                'count': len(pts),
            })

            self._socket.send_string(f"pointcloud {message}", zmq.NOBLOCK)

        except Exception as e:
            self.logger.debug(f"Pointcloud publish failed: {e}")

    def teardown(self) -> None:
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self.logger.info("ZMQPublishWorker shut down")
