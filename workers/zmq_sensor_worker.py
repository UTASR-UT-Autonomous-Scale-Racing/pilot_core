"""
ZMQSensorWorker — subscribes to the JSON sensor feed from autodrive_bridge
and fills the frame, IMU, pose and LIDAR buffers.  Used instead of
CameraWorker + IMUWorker in simulator mode.

Conventions:
    * frame_buf: RGB uint8 + depth float32 in metres (8-bit depth images are
      scaled linearly, 0 = near, 255 = ``depth_max_meters``).
    * Simulator commands are signed [-1, 1] with steering +1 = LEFT; they are
      mapped to [0, 1] (0.5 = neutral, steering 0 = full left) in the IMU
      vector's last two slots.
    * The simulator's ground-truth vehicle-to-world pose (ROS / Z-up frame,
      quaternion sent as [x, y, z, w]) goes to PoseBuffer with its speed.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any


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
    lidar_buffer : LidarBuffer, optional
        Shared LidarBuffer for the 1080-ray 2D scan
    depth_max_meters : float
        Depth mapped to pixel value 255 for 8-bit depth images (default 10.0)
    collision_counter, lap_counter, checkpoint_counter : mp.Value, optional
        Shared integers mirrored from simulator telemetry (for the gym env)
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
        self._depth_logged = False
        self._pos_logged = False
        self._collision_counter = None   # mp.Value('i') shared with gym env
        self._lap_counter = None         # mp.Value('i') shared with gym env
        self._checkpoint_counter = None  # mp.Value('i') shared with gym env

    def setup(self) -> None:
        if zmq is None:
            raise ImportError("pyzmq is required for ZMQSensorWorker")

        host = self.config.get("host", "localhost")
        port = self.config.get("port", 5555)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 10)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)  # ms
        self._socket.setsockopt(zmq.CONFLATE, 1)  # keep only the latest message

        try:
            self._socket.connect(f"tcp://{host}:{port}")
            self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
            self.logger.info(f"Connected to ZMQ sensor publisher at {host}:{port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to {host}:{port}: {e}")
            raise

        self._frame_buf = self.config["frame_buffer"]
        self._imu_buf = self.config["imu_buffer"]
        self._pose_buf = self.config.get("pose_buffer")
        self._lidar_buf = self.config.get("lidar_buffer")
        self._depth_max_meters = float(self.config.get("depth_max_meters", 10.0))
        self._collision_counter = self.config.get("collision_counter")
        self._lap_counter = self.config.get("lap_counter")
        self._checkpoint_counter = self.config.get("checkpoint_counter")

        self._last_report_time = time.time()
        self.logger.info(f"ZMQSensorWorker ready (depth_max_meters={self._depth_max_meters})")

    def step(self) -> None:
        try:
            msg_json = self._socket.recv_string(zmq.NOBLOCK)
            data = json.loads(msg_json)

            self._process_frame_data(data)
            self._process_imu_data(data)
            self._process_pose_data(data)
            self._process_lidar_data(data)
            self._process_telemetry_data(data)

            self._frame_count += 1

            now = time.time()
            if now - self._last_report_time >= 1.0:
                elapsed = now - self._last_report_time
                fps = self._frame_count / elapsed if elapsed > 0 else 0
                self.logger.debug(f"ZMQ sensor FPS: {fps:.1f}")
                self._frame_count = 0
                self._last_report_time = now

        except zmq.Again:
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

            # 'jpeg' (bridge default) or raw uint8 bytes with a 'shape' field
            encoding = front_cam.get('encoding', 'raw')
            if encoding == 'jpeg':
                nparr = np.frombuffer(img_bytes, dtype=np.uint8)
                rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            else:
                rgb = np.frombuffer(img_bytes, dtype=np.uint8).reshape(front_cam['shape'])
                if len(rgb.shape) == 2:
                    rgb = np.stack([rgb, rgb, rgb], axis=-1)

            depth = None
            if 'depth_camera' in data and data['depth_camera']:
                try:
                    depth_cam = data['depth_camera']
                    depth_bytes = base64.b64decode(depth_cam['data'])
                    depth_encoding = depth_cam.get('encoding', 'raw')

                    if depth_encoding in ('jpeg', 'png'):
                        # 8-bit grayscale → linear depth in metres: 0 = near,
                        # 255 = far clip.  depth_max_meters must match the
                        # simulator camera's far clip.
                        nparr = np.frombuffer(depth_bytes, dtype=np.uint8)
                        depth = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                        depth = depth / 255.0 * self._depth_max_meters
                        # Log the pixel → metres mapping of the first frame
                        if not self._depth_logged:
                            raw_u8 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                            self.logger.debug(
                                f"[depth calibration] raw uint8 pixel range: "
                                f"min={int(raw_u8.min())} max={int(raw_u8.max())} "
                                f"mean={float(raw_u8.mean()):.1f}  "
                                f"→ depth range: "
                                f"{float(depth.min()):.2f}–{float(depth.max()):.2f} m  "
                                f"(depth_max_meters={self._depth_max_meters})"
                            )
                            self._depth_logged = True
                    else:
                        # Raw uint16 depth, written without unit conversion
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

            self._frame_buf.write(rgb, depth)

        except Exception as e:
            self.logger.warning(f"Frame processing failed: {e}")

    def _process_imu_data(self, data: dict) -> None:
        """Extract and write IMU data to IMUBuffer."""
        try:
            if 'orientation_quat' in data:
                # Bridge sends [x, y, z, w] (ROS); IMU vector layout is [w, x, y, z]
                x, y, z, w = data['orientation_quat']
                quat = np.array([w, x, y, z], dtype=np.float32)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            if 'linear_acceleration' in data:
                lin_acc = np.array(data['linear_acceleration'], dtype=np.float32)
            else:
                lin_acc = np.zeros(3, dtype=np.float32)

            # Simulator throttle / steering are signed [-1, 1]
            throttle = float(data.get('throttle', 0.0))
            steering = float(data.get('steering', 0.0))

            throttle_norm = (throttle + 1.0) / 2.0
            steering_norm = (1.0 - steering) / 2.0  # simulator +1 = left → buffer 0 = left
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
            self.logger.debug(
                f"[sim position check] raw position={data['position']} "
                f"orientation_quat={data['orientation_quat']}"
            )
            self._pos_logged = True

        if self._pose_buf is None:
            return

        try:
            pos = np.array(data['position'], dtype=np.float32)
            quat = np.array(data['orientation_quat'], dtype=np.float32)

            pose = np.eye(4, dtype=np.float32)
            pose[0:3, 3] = pos

            # Quaternion → rotation matrix; bridge sends [x, y, z, w] (ROS)
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
