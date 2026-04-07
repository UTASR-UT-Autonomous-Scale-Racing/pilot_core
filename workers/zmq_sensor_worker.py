"""
ZMQSensorWorker — Subscribe to sensor data from autodrive_bridge via ZMQ
and populate multiple shared ring buffers (frame, imu, pose).

This worker replaces CameraWorker + IMUWorker when running in simulator mode.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import os

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, IMUBuffer, PoseBuffer, LidarBuffer

try:
    import zmq
except ImportError:
    zmq = None

try:
    import cv2
except ImportError:
    cv2 = None


class ZMQSensorWorker(BaseWorker):
    """Worker that subscribes to ZMQ sensor feed and fills multiple buffers.

    Config kwargs
    -------------
    host : str
        Hostname of autodrive_bridge (default: localhost)
    port : int
        Port number for sensor data (default: 5555)
    frame_buffer : FrameBuffer
        Shared FrameBuffer to write RGB+Depth frames
    imu_buffer : IMUBuffer
        Shared IMUBuffer to write IMU data
    pose_buffer : PoseBuffer, optional
        Shared PoseBuffer to write ground truth poses from simulator
    """

    def __init__(self, name: str = "zmq_sensor", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._socket = None
        self._context = None
        self._frame_buf: FrameBuffer | None = None
        self._imu_buf: IMUBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._lidar_buf: LidarBuffer | None = None
        self._frame_count = 0
        self._last_report_time = 0
        self._depth_max_meters: float = 10.0
        self._depth_logged = False  # log raw depth range once for calibration
        self._pos_logged = False   # log raw simulator position once for scale check
        self._collision_counter = None   # mp.Value('i') shared with gym env
        self._lap_counter = None         # mp.Value('i') shared with gym env
        self._checkpoint_counter = None  # mp.Value('i') shared with gym env
        self._outer_boundary: np.ndarray | None = None
        self._inner_boundary: np.ndarray | None = None

    def setup(self) -> None:
        if zmq is None:
            raise ImportError("pyzmq is required for ZMQSensorWorker")

        host = self.config.get("host", "localhost")
        port = self.config.get("port", 5555)

        # Initialize ZMQ subscriber
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 10)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout
        self._socket.setsockopt(zmq.CONFLATE, 1)  # Keep only latest message

        try:
            self._socket.connect(f"tcp://{host}:{port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.logger.info(f"Connected to ZMQ sensor publisher at {host}:{port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {host}:{port}: {e}")
            raise

        # Get buffer references
        self._frame_buf = self.config["frame_buffer"]
        self._imu_buf = self.config["imu_buffer"]
        self._pose_buf = self.config.get("pose_buffer")
        self._lidar_buf = self.config.get("lidar_buffer")
        self._depth_max_meters = float(self.config.get("depth_max_meters", 10.0))
        self._collision_counter = self.config.get("collision_counter")
        self._lap_counter = self.config.get("lap_counter")
        self._checkpoint_counter = self.config.get("checkpoint_counter")

        # Load and resample boundary lines for distance logging (optional)
        boundary_dir = self.config.get("boundary_save_path", "rl/boundaries")
        boundary_spacing = float(self.config.get("boundary_spacing", 0.1))
        try:
            from workers.boundary_resampler import resample_uniform, smooth_boundary
            outer_path = os.path.join(boundary_dir, "outer_line.npy")
            inner_path = os.path.join(boundary_dir, "inner_line.npy")
            if os.path.exists(outer_path) and os.path.exists(inner_path):
                outer: np.ndarray = resample_uniform(smooth_boundary(np.load(outer_path)[:, :2], 15), boundary_spacing)
                inner: np.ndarray = resample_uniform(smooth_boundary(np.load(inner_path)[:, :2], 15), boundary_spacing)
                self._outer_boundary = outer
                self._inner_boundary = inner
                self.logger.info(
                    f"Loaded+resampled boundaries: {outer.shape[0]} outer, "
                    f"{inner.shape[0]} inner pts (spacing={boundary_spacing}m)"
                )
            else:
                self.logger.info("No boundary files found — distance logging disabled")
        except Exception as e:
            self.logger.warning(f"Could not load boundaries: {e}")

        self._last_report_time = time.time()
        self.logger.info(f"ZMQSensorWorker ready (depth_max_meters={self._depth_max_meters})")

    def step(self) -> None:
        try:
            # Receive sensor data (non-blocking)
            msg_json = self._socket.recv_string(zmq.NOBLOCK)
            data = json.loads(msg_json)

            _fc = data.get('front_camera')
            _dc = data.get('depth_camera')
            if _fc and cv2 is not None:
                try:
                    _fc_arr = np.frombuffer(base64.b64decode(_fc['data']), dtype=np.uint8)
                    _fc_img = cv2.imdecode(_fc_arr, cv2.IMREAD_COLOR)
                    _fc_info = f"shape={_fc_img.shape}" if _fc_img is not None else "decode_failed"
                except Exception as _e:
                    _fc_info = f"decode_error={_e}"
            elif _fc:
                _fc_info = f"PRESENT (cv2 unavailable)"
            else:
                _fc_info = 'NONE'
            if _dc and cv2 is not None:
                try:
                    _dc_arr = np.frombuffer(base64.b64decode(_dc['data']), dtype=np.uint8)
                    _dc_img = cv2.imdecode(_dc_arr, cv2.IMREAD_GRAYSCALE)
                    _dc_info = f"shape={_dc_img.shape}" if _dc_img is not None else "decode_failed"
                except Exception as _e:
                    _dc_info = f"decode_error={_e}"
            elif _dc:
                _dc_info = f"PRESENT (cv2 unavailable)"
            else:
                _dc_info = 'NONE'
            # print(f"[ZMQ] front_camera={_fc_info} | depth_camera={_dc_info}")
            # print(f"[ZMQ] IMU: quat={data.get('orientation_quat')} accel={data.get('linear_acceleration')} throttle={data.get('throttle')} steering={data.get('steering')}")
            # Process and write to buffers
            self._process_frame_data(data)
            self._process_imu_data(data)
            self._process_pose_data(data)
            self._process_lidar_data(data)
            self._process_telemetry_data(data)

            self._frame_count += 1

            # Report stats every second
            now = time.time()
            if now - self._last_report_time >= 1.0:
                elapsed = now - self._last_report_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                # self.logger.debug(f"ZMQ sensor FPS: {fps:.1f}")
                self._frame_count = 0
                self._last_report_time = now

        except zmq.Again:
            # No data available, continue
            time.sleep(0.001)
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to decode JSON: {e}")
            time.sleep(0.001)
        except Exception as e:
            self.logger.error(f"Error processing sensor data: {e}")
            time.sleep(0.01)

    def _process_frame_data(self, data: dict) -> None:
        """Extract and write frame data to FrameBuffer."""
        if 'front_camera' not in data or not data['front_camera']:
            return

        try:
            import cv2
            front_cam = data['front_camera']
            img_bytes = base64.b64decode(front_cam['data'])

            # Decode based on encoding field (JPEG from bridge, or raw bytes legacy)
            encoding = front_cam.get('encoding', 'raw')
            if encoding == 'jpeg':
                nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.frombuffer(img_bytes, dtype=np.uint8).reshape(front_cam['shape'])
                if len(rgb.shape) == 2:
                    rgb = np.stack([rgb, rgb, rgb], axis=-1)

            # Get depth frame if available
            depth = None
            if 'depth_camera' in data and data['depth_camera']:
                try:
                    depth_cam = data['depth_camera']
                    depth_bytes = base64.b64decode(depth_cam['data'])
                    depth_encoding = depth_cam.get('encoding', 'raw')

                    if depth_encoding in ('jpeg', 'png'):
                        # JPEG grayscale → linear float32 depth in metres.
                        # Pixel value 0=near, 255=far_clip.  Tune depth_max_meters
                        # in config_zmq.yaml to match the simulator camera far clip.
                        nparr = np.frombuffer(depth_bytes, dtype=np.uint8)
                        depth = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                        depth = depth / 255.0 * self._depth_max_meters
                        # Log raw pixel range once so the user can verify the scale
                        if not self._depth_logged:
                            raw_u8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                            self.logger.info(
                                f"[depth calibration] raw uint8 pixel range: "
                                f"min={int(raw_u8.min())} max={int(raw_u8.max())} "
                                f"mean={float(raw_u8.mean()):.1f}  "
                                f"→ depth range: "
                                f"{float(depth.min()):.2f}–{float(depth.max()):.2f} m  "
                                f"(depth_max_meters={self._depth_max_meters})"
                            )
                            self._depth_logged = True
                    else:
                        # Legacy raw uint16 path
                        depth_flat = np.frombuffer(depth_bytes, dtype=np.uint16)
                        try:
                            depth = depth_flat.reshape(depth_cam['shape'][:2]).astype(np.float32)
                        except ValueError:
                            side = int(np.sqrt(len(depth_flat)))
                            depth = depth_flat[:side*side].reshape((side, side)).astype(np.float32)

                    if depth.shape != rgb.shape[:2]:
                        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                                           interpolation=cv2.INTER_LINEAR)
                except Exception as e:
                    self.logger.debug(f"Depth decode/resize failed, using zeros: {e}")
                    depth = np.zeros(rgb.shape[:2], dtype=np.float32)
            else:
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)

            # Write to frame buffer
            self._frame_buf.write(rgb, depth)

        except Exception as e:
            self.logger.warning(f"Frame processing failed: {e}")

    def _process_imu_data(self, data: dict) -> None:
        """Extract and write IMU data to IMUBuffer."""
        try:
            # Get quaternion orientation
            if 'orientation_quat' in data:
                quat = np.array(data['orientation_quat'], dtype=np.float32)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            # Get linear acceleration
            if 'linear_acceleration' in data:
                lin_acc = np.array(data['linear_acceleration'], dtype=np.float32)
            else:
                lin_acc = np.zeros(3, dtype=np.float32)

            # Get throttle and steering (normalized to [0, 1])
            throttle = float(data.get('throttle', 0.0))
            steering = float(data.get('steering', 0.0))

            # Convert from [-1, 1] to [0, 1]
            throttle_norm = (throttle + 1.0) / 2.0
            steering_norm = (steering + 1.0) / 2.0
            throttle_norm = np.clip(throttle_norm, 0.0, 1.0)
            steering_norm = np.clip(steering_norm, 0.0, 1.0)

            # Construct IMU vector: [w, x, y, z, ax, ay, az, throttle, steering]
            imu_vector = np.concatenate([
                quat,      # Orientation quaternion (4 floats)
                lin_acc,   # Linear acceleration (3 floats)
                [throttle_norm, steering_norm]  # Control inputs (2 floats)
            ]).astype(np.float32)

            self._imu_buf.write(imu_vector)

        except Exception as e:
            self.logger.warning(f"IMU processing failed: {e}")

    def _process_pose_data(self, data: dict) -> None:
        """Extract and write ground truth pose to PoseBuffer (optional)."""
        if 'position' not in data or 'orientation_quat' not in data:
            return

        if not self._pos_logged:
            self.logger.info(
                f"[sim position check] raw position={data['position']} "
                f"orientation_quat={data['orientation_quat']}"
            )
            self._pos_logged = True

        try:
            pos = np.array(data['position'], dtype=np.float32)

        except Exception as e:
            self.logger.warning(f"Boundary distance check failed: {e}")

        if self._pose_buf is None:
            return

        try:
            pos = np.array(data['position'], dtype=np.float32)
            quat = np.array(data['orientation_quat'], dtype=np.float32)

            # Create 4x4 pose matrix
            pose = np.eye(4, dtype=np.float32)
            pose[0:3, 3] = pos

            # Convert quaternion — bridge sends [x, y, z, w] (ROS convention)
            x, y, z, w = quat[0], quat[1], quat[2], quat[3]

            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)

            pose[0:3, 0:3] = R

            speed = float(data.get('speed', 0.0))
            self._pose_buf.write(pose, None, speed=speed)

        except Exception as e:
            self.logger.warning(f"Pose processing failed: {e}")

    def _process_lidar_data(self, data: dict) -> None:
        """Extract and write 2D LIDAR scan to LidarBuffer."""
        if self._lidar_buf is None:
            return
        if 'lidar' not in data or not data['lidar']:
            return

        try:
            ranges = np.array(data['lidar'], dtype=np.float32)
            # Accept 1080 or 1081 rays (bridge quirk); trim to 1080
            if len(ranges) < 1080:
                self.logger.debug(f"Unexpected LIDAR ray count: {len(ranges)}, skipping")
                return
            self._lidar_buf.write(ranges[:1080])
        except Exception as e:
            self.logger.warning(f"LIDAR processing failed: {e}")

    def _process_telemetry_data(self, data: dict) -> None:
        """Mirror simulator telemetry into shared mp.Values for the gym env."""
        sim_collisions = int(data.get('collision_count', 0))
        sim_laps = int(data.get('lap_count', 0))
        sim_checkpoint = int(data.get('checkpoint_count', 0))

        if self._collision_counter is not None:
            self._collision_counter.value = sim_collisions
        if self._lap_counter is not None:
            self._lap_counter.value = sim_laps
        if self._checkpoint_counter is not None:
            self._checkpoint_counter.value = sim_checkpoint

    def teardown(self) -> None:
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()
        self.logger.info("ZMQSensorWorker shut down")
