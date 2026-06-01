"""
Decoders shared by simulator links: a sensor dict (see AutoDRIVEWorker's
autodrive_to_internal) -> frame (RGB uint8, depth float32 m), IMU (sim commands
remapped to [0, 1] in the last two slots), ground-truth pose (Z-up, vehicle-to-world),
optional StateBuffer, LIDAR, and the sim's lap / collision / checkpoint counters.
Subclasses own the transport (setup / step / teardown).
"""

from __future__ import annotations

import base64
import time
from typing import Any


import numpy as np

from control.types import VehicleState

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, IMUBuffer, PoseBuffer, LidarBuffer, StateBuffer

try:
    import cv2
except ImportError:
    cv2 = None


class SimDecoder(BaseWorker):
    """Sim sensor dicts -> shared buffers (decoders only; no transport).

    Config kwargs:
      frame_buffer        FrameBuffer; None skips all image decoding
      imu_buffer          IMUBuffer
      pose_buffer         PoseBuffer (ground truth)
      state_buffer        StateBuffer, VehicleState from GT pose + sim speed
      lidar_buffer        LidarBuffer, 1080-ray scan
      depth_max_meters    metres at pixel 255 for 8-bit depth, default 15.0 (sim far clip)
      depth_raw_scale     metres per unit for raw uint16 depth (else zero depth + error)
      no_message_warn_s   warn after this long without messages, default 1.0
      collision_counter, lap_counter, checkpoint_counter   mp.Values for the gym env
    """

    def __init__(self, name: str = "sim", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._imu_buf: IMUBuffer | None = None
        self._pose_buf: PoseBuffer | None = None
        self._lidar_buf: LidarBuffer | None = None
        self._state_buf: StateBuffer | None = None
        self._frame_count = 0
        self._last_report_time = 0
        self._depth_max_meters: float = 15.0
        self._depth_raw_scale: float | None = None
        self._depth_logged = False
        self._raw_depth_error_logged = False
        self._last_msg_time = 0.0
        self._last_silence_warn = 0.0
        self._silent = False
        self._no_msg_warn_s = 1.0
        self._pos_logged = False
        # mp.Value('i') each, shared with the gym env
        self._collision_counter = None
        self._lap_counter = None
        self._checkpoint_counter = None

    def _check_silence(self, now: float) -> None:
        """Warn when the sim has gone quiet (repeats every 5 s while it stays quiet)."""
        gap = now - self._last_msg_time
        if gap > self._no_msg_warn_s and (not self._silent or now - self._last_silence_warn > 5.0):
            self.logger.warning(f"No sensor message from the simulator for {gap:.1f} s")
            self._silent = True
            self._last_silence_warn = now

    def _process_frame_data(self, data: dict) -> None:
        if 'front_camera' not in data or not data['front_camera']:
            return

        try:
            import cv2
            front_cam = data['front_camera']
            img_bytes = base64.b64decode(front_cam['data'])

            # either 'jpeg' (the AutoDRIVE sim) or raw uint8 + a 'shape' field
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
                        # linear 0..255 -> 0..depth_max_meters (must match the sim far clip)
                        nparr = np.frombuffer(depth_bytes, dtype=np.uint8)
                        depth = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE).astype(np.float32)
                        depth = depth / 255.0 * self._depth_max_meters
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
                        depth = self._decode_raw_depth(depth_bytes, depth_cam)

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

    def _decode_raw_depth(self, depth_bytes: bytes, depth_cam: dict) -> np.ndarray:
        """Raw uint16 depth -> metres. Needs depth_raw_scale and the frame's
        'shape'; anything else raises (the caller substitutes zeros)."""
        if self._depth_raw_scale is None:
            if not self._raw_depth_error_logged:
                self.logger.error("Raw uint16 depth received but depth_raw_scale (m per unit) "
                                  "is not configured; depth is zeroed")
                self._raw_depth_error_logged = True
            raise ValueError("depth_raw_scale not set")
        shape = depth_cam.get('shape')
        if not shape:
            raise ValueError("raw depth frame has no 'shape'")
        depth_flat = np.frombuffer(depth_bytes, dtype=np.uint16)
        return depth_flat.reshape(tuple(shape[:2])).astype(np.float32) * self._depth_raw_scale

    def _process_imu_data(self, data: dict) -> None:
        if self._imu_buf is None:
            return
        try:
            if 'orientation_quat' in data:
                # the sim sends [x, y, z, w] (ROS order), the IMU vector wants [w, x, y, z]
                x, y, z, w = data['orientation_quat']
                quat = np.array([w, x, y, z], dtype=np.float32)
            else:
                quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            if 'linear_acceleration' in data:
                lin_acc = np.array(data['linear_acceleration'], dtype=np.float32)
            else:
                lin_acc = np.zeros(3, dtype=np.float32)

            # sim values are signed [-1, 1]
            throttle = float(data.get('throttle', 0.0))
            steering = float(data.get('steering', 0.0))

            throttle_norm = (throttle + 1.0) / 2.0
            steering_norm = (1.0 - steering) / 2.0  # sim +1 (left) -> buffer 0 (left)
            throttle_norm = np.clip(throttle_norm, 0.0, 1.0)
            steering_norm = np.clip(steering_norm, 0.0, 1.0)

            imu_vector = np.concatenate([
                quat,
                lin_acc,
                [throttle_norm, steering_norm]
            ]).astype(np.float32)

            self._imu_buf.write(imu_vector)

        except Exception as e:
            self.logger.warning(f"IMU processing failed: {e}")

    def _process_pose_data(self, data: dict) -> None:
        """Ground-truth pose, only if a pose_buffer was given."""
        if 'position' not in data or 'orientation_quat' not in data:
            return

        if not self._pos_logged:
            self.logger.debug(
                f"[sim position check] raw position={data['position']} "
                f"orientation_quat={data['orientation_quat']}"
            )
            self._pos_logged = True

        if self._pose_buf is None and self._state_buf is None:
            return

        try:
            pos = np.array(data['position'], dtype=np.float32)
            quat = np.array(data['orientation_quat'], dtype=np.float32)

            pose = np.eye(4, dtype=np.float32)
            pose[0:3, 3] = pos

            # quat -> R. Again [x, y, z, w] order from the sim
            x, y, z, w = quat[0], quat[1], quat[2], quat[3]

            R = np.array([
                [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
                [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
                [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
            ], dtype=np.float32)

            pose[0:3, 0:3] = R

            speed = float(data.get('speed', 0.0))
            now = time.monotonic()
            if self._pose_buf is not None:
                self._pose_buf.write(pose, None, speed=speed, ts=now)
            if self._state_buf is not None:
                # sim world is Z-up: ground plane is X, Y; yaw from the body X axis
                yaw = float(np.arctan2(R[1, 0], R[0, 0]))
                self._state_buf.write(
                    VehicleState(t=now, x=float(pos[0]), y=float(pos[1]), yaw=yaw, v=speed), ts=now)

        except Exception as e:
            self.logger.warning(f"Pose processing failed: {e}")

    def _process_lidar_data(self, data: dict) -> None:
        if self._lidar_buf is None:
            return
        if 'lidar' not in data or not data['lidar']:
            return

        try:
            ranges = np.array(data['lidar'], dtype=np.float32)
            # AutoDRIVE sends 1081 rays (-135..+135 deg inclusive); the buffer holds 1080
            if len(ranges) < 1080:
                self.logger.debug(f"Unexpected LIDAR ray count: {len(ranges)}, skipping")
                return
            self._lidar_buf.write(ranges[:1080])
        except Exception as e:
            self.logger.warning(f"LIDAR processing failed: {e}")

    def _process_telemetry_data(self, data: dict) -> None:
        """Copy the sim's collision/lap/checkpoint counts into the shared mp.Values."""
        sim_collisions = int(data.get('collision_count', 0))
        sim_laps = int(data.get('lap_count', 0))
        sim_checkpoint = int(data.get('checkpoint_count', 0))

        if self._collision_counter is not None:
            self._collision_counter.value = sim_collisions
        if self._lap_counter is not None:
            self._lap_counter.value = sim_laps
        if self._checkpoint_counter is not None:
            self._checkpoint_counter.value = sim_checkpoint
