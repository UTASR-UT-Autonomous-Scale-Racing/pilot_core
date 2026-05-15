"""
Telemetry from the shared buffers to the simulator over ZMQ PUB, one "<topic> <json>"
string per message. Topics: telemetry (pose), segmentation (uint8, 1 = drivable),
map (track boundaries), pointcloud. Commands go out via ActuatorWorker (port 5556).
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import PoseBuffer, SegmentationBuffer

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
    """ZMQ telemetry publisher. All buffers optional.

    Config kwargs:
      port               PUB port, default 5557
      pose_buffer        PoseBuffer
      seg_buffer         SegmentationBuffer
      boundary_buffer    TrackBoundaryBuffer
      pointcloud_buffer  PointCloudBuffer (sent as base64 float32)
      publish            dict of bools: pose / segmentation / boundary / pointcloud
    """

    def __init__(self, name: str = "zmq_publish", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._socket = None
        self._context = None
        self._pose_buf: Optional[PoseBuffer] = None
        self._seg_buf: Optional[SegmentationBuffer] = None
        self._boundary_buf: Optional[Any] = None
        self._pc_buf: Optional[Any] = None
        self._publish_config: dict = {}
        self._last_seq = {
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

        self._pose_buf = self.config.get("pose_buffer")
        self._seg_buf = self.config.get("seg_buffer")
        self._boundary_buf = self.config.get("boundary_buffer")
        self._pc_buf = self.config.get("pointcloud_buffer")

        self._publish_config = self.config.get("publish", {
            "pose": False,
            "segmentation": False,
            "boundary": False,
        })

        # PUB drops anything sent before a subscriber connects, so wait a bit
        time.sleep(0.5)
        self.logger.info(f"Publishing: {list(k for k, v in self._publish_config.items() if v)}")

    def step(self) -> None:
        self._frame_count += 1

        if self._publish_config.get("pose", False) and self._pose_buf:
            self._publish_pose()

        # heavier streams only every other iteration
        if self._frame_count % 2 == 0:
            if self._publish_config.get("segmentation", False) and self._seg_buf:
                self._publish_segmentation()

        if self._frame_count % 2 == 0:
            if self._publish_config.get("boundary", False) and self._boundary_buf:
                self._publish_boundary()
            if self._publish_config.get("pointcloud", False) and self._pc_buf:
                self._publish_pointcloud()

        time.sleep(0.01)

    def _publish_pose(self) -> None:
        try:
            result = self._pose_buf.read_latest()
            if result is None:
                return

            header, pose, _, _ = result
            if header.seq_no <= self._last_seq['pose']:
                return
            self._last_seq['pose'] = header.seq_no

            position = pose[0:3, 3].tolist()

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'position': position,
                'pose_matrix': pose.flatten().tolist(),
            })

            self._socket.send_string(f"telemetry {message}", zmq.NOBLOCK)

        except Exception as e:
            self.logger.debug(f"Pose publish failed: {e}")

    def _publish_segmentation(self) -> None:
        try:
            result = self._seg_buf.read_latest()
            if result is None:
                return

            header, mask = result
            if header.seq_no <= self._last_seq['segmentation']:
                return
            self._last_seq['segmentation'] = header.seq_no

            if mask.dtype != np.uint8:
                mask = (mask > 0).astype(np.uint8)
            mask_b64 = base64.b64encode(np.ascontiguousarray(mask).tobytes()).decode('utf-8')

            message = json.dumps({
                'seq': header.seq_no,
                'timestamp': header.produced_ts,
                'mask': mask_b64,
                'shape': list(mask.shape),
                'dtype': 'uint8',
                'values': 'binary',   # 0 = not track, 1 = drivable track
            })

            self._socket.send_string(f"segmentation {message}", zmq.NOBLOCK)

        except Exception as e:
            import traceback
            self.logger.error(f"Segmentation publish failed: {e}")
            traceback.print_exc()

    def _publish_boundary(self) -> None:
        """Boundaries go out on the "map" topic."""
        try:
            result = self._boundary_buf.read_latest()
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

            self._socket.send_string(f"map {message}", zmq.NOBLOCK)

            self.logger.info("Published track boundaries to simulator")

        except Exception as e:
            self.logger.debug(f"Boundary publish failed: {e}")

    def _publish_pointcloud(self) -> None:
        """Point cloud as base64 of the float32 (N, 3) XYZ buffer."""
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
