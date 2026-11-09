"""
OAK-D capture (DepthAI v3): RGB uint8 + aligned depth float32 (metres) -> FrameBuffer.
produced_ts is the capture time: DepthAI timestamps use steady_clock, i.e. time.monotonic().
The on-board IMU is streamed from here (optional imu_buffer) because only one
process can open a DepthAI device.
"""

from __future__ import annotations

import os
import re
import time
from datetime import timedelta
from typing import Any, Optional

import cv2
import numpy as np

from .base_worker import BaseWorker, FatalWorkerError
from shared_buffer import FrameBuffer

try:
    import depthai as dai
except ImportError:
    dai = None  # so tests can import this without depthai installed


def read_settings_camera(path: str) -> Optional[dict]:
    """Device id and intrinsics recorded in an ORB-SLAM3 settings file
    written by scripts/gen_orbslam_settings.py (None if unreadable)."""
    from control.types import resolve_path

    path = resolve_path(path) if path else path
    if not path or not os.path.isfile(path):
        return None
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        out: dict = {}
        for key in ("fx", "fy", "cx", "cy"):
            node = fs.getNode(f"Camera1.{key}")
            if node.empty():
                return None
            out[key] = float(node.real())
        node = fs.getNode("Pilot.deviceId")
        out["device_id"] = node.string() if not node.empty() else None
    finally:
        fs.release()
    if not out["device_id"]:
        # older generated files only have it in the header comment
        with open(path) as f:
            m = re.search(r"#\s*Device:\s*(\S+)", f.read())
        out["device_id"] = m.group(1) if m else None
    return out


def intrinsics_mismatch(settings: dict, K, tol: float = 0.02) -> list[str]:
    """Names of fx/fy/cx/cy that differ from the 3x3 K by more than tol (relative)."""
    live = {"fx": K[0][0], "fy": K[1][1], "cx": K[0][2], "cy": K[1][2]}
    return [
        f"{k} {settings[k]:.1f} vs live {live[k]:.1f}"
        for k in live
        if abs(settings[k] - live[k]) > tol * abs(live[k])
    ]


class CameraWorker(BaseWorker):
    """OAK-D RGB + depth capture.

    Config kwargs:
      output_buffer        FrameBuffer
      fps                  default 30
      resolution           [w, h] for RGB and depth, default [640, 400]
      imu_buffer           stream the on-board IMU into it if given
      raw_imu_buffer       also stream accel + all three gyro axes (RawIMUBuffer), for
                           ORB-SLAM3's RGB-D-inertial mode
      imu_rate_hz          default 100
      max_exposure_us      auto-exposure limit, default 4000 (0 = DepthAI default)
      settings_path        ORB-SLAM3 settings to check against, default orbslam_oakd_rgbd.yaml
      extended_disparity   default True (closer min depth, no sub-pixel)
      frame_timeout_s      re-open after no frame this long, default 1.0
      startup_timeout_s    same for the first frame after (re)opening, default 5.0
      max_step_errors      consecutive step errors before re-opening, default 5
      max_reopen_attempts  failed re-opens before exiting, default 3
    """

    def __init__(self, name: str = "camera", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._pipeline = None
        self._sync_q = None
        self._frame_count: int = 0
        self._imu_buf = None
        self._raw_imu_buf = None
        self._imu_q = None
        self._has_rotation_vector = False
        self._step_errors = 0
        self._last_frame_t = 0.0
        self._got_frame = False
        self._startup_timeout = 5.0
        self._intrinsics_checked = False

    def setup(self) -> None:
        if dai is None:
            raise RuntimeError("depthai is not installed; CameraWorker needs it for the OAK-D")
        self._frame_buf = self.config["output_buffer"]
        self._imu_buf = self.config.get("imu_buffer")
        self._raw_imu_buf = self.config.get("raw_imu_buffer")
        self._frame_timeout = float(self.config.get("frame_timeout_s", 1.0))
        self._startup_timeout = float(self.config.get("startup_timeout_s", 5.0))
        self._max_step_errors = int(self.config.get("max_step_errors", 5))
        self._max_reopen = int(self.config.get("max_reopen_attempts", 3))
        self._settings = read_settings_camera(
            self.config.get("settings_path", "orbslam_oakd_rgbd.yaml")
        )
        self._start_device()
        self._check_device_id()

    def _start_device(self) -> None:
        """Build and start the pipeline (also used to re-open the camera)."""
        fps = self.config.get("fps", 30)
        width, height = self.config.get("resolution", [640, 400])
        size = (width, height)
        max_exp_us = int(self.config.get("max_exposure_us", 4000))

        self._pipeline = self._open_pipeline()
        device = self._pipeline.getDefaultDevice()
        platform = device.getPlatform()

        cams = {
            socket: self._pipeline.create(dai.node.Camera).build(socket, sensorFps=fps)
            for socket in (
                dai.CameraBoardSocket.CAM_A,   # RGB
                dai.CameraBoardSocket.CAM_B,   # left
                dai.CameraBoardSocket.CAM_C,   # right
            )
        }
        if max_exp_us > 0:
            # short exposures: less motion blur for ORB features and stereo
            for cam in cams.values():
                cam.initialControl.setAutoExposureLimit(max_exp_us)
        cam_rgb = cams[dai.CameraBoardSocket.CAM_A]

        stereo = self._pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)
        if self.config.get("extended_disparity", True):
            # halves the minimum depth; DepthAI says it's incompatible with sub-pixel
            stereo.setSubpixel(False)
            stereo.setExtendedDisparity(True)

        cams[dai.CameraBoardSocket.CAM_B].requestOutput(size, fps=fps).link(stereo.left)
        cams[dai.CameraBoardSocket.CAM_C].requestOutput(size, fps=fps).link(stereo.right)

        # undistorted, otherwise the aligned depth won't line up with it
        rgb_out = cam_rgb.requestOutput(size, fps=fps, enableUndistortion=True)

        if platform == dai.Platform.RVC4:
            align = self._pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            rgb_out.link(align.inputAlignTo)
            depth_out = align.outputAligned
        else:
            # RVC2: StereoDepth can do the alignment itself
            rgb_out.link(stereo.inputAlignTo)
            depth_out = stereo.depth

        # pair rgb/depth if timestamps are within half a frame period
        sync = self._pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(seconds=1.0 / (2 * fps)))
        rgb_out.link(sync.inputs["rgb"])
        depth_out.link(sync.inputs["depth"])
        # small non-blocking queue; step() always takes the newest group
        self._sync_q = sync.out.createOutputQueue(maxSize=2, blocking=False)

        self._imu_q = None
        if self._imu_buf is not None or self._raw_imu_buf is not None:
            self._setup_imu(device, self.config.get("imu_rate_hz", 100))

        self._pipeline.start()
        self._last_frame_t = time.monotonic()
        self._got_frame = False
        self._step_errors = 0
        self.logger.info(
            f"OAK-D camera ready ({platform}) @ {fps} FPS, {width}×{height}, "
            f"AE limit {max_exp_us} µs"
            + (" + on-board IMU" if self._imu_q is not None else "")
        )
        self._warn_slow_link(device, fps)

    def _warn_slow_link(self, device, fps: int) -> None:
        """Log a warning on a USB 2 link above 15 FPS: RGB-D frames drop and lag there."""
        try:
            speed = device.getUsbSpeed()
            slow = speed in (dai.UsbSpeed.LOW, dai.UsbSpeed.FULL, dai.UsbSpeed.HIGH)
        except Exception:  # RVC4 over Ethernet, or a mocked device
            return
        if slow and fps > 15:
            self.logger.warning(
                f"OAK-D link is {speed.name}, not USB 3: {fps} FPS RGB-D will drop frames and lag "
                "(measured 25 Hz, ~320 ms at 30 FPS). Use a USB 3 port/cable or fps: 15."
            )

    def _open_pipeline(self, attempts: int = 5, backoff_s: float = 1.0):
        """dai.Pipeline() with retries.

        When a pipeline closes the OAK-D reboots and can't be reopened for a
        moment ("Couldn't open stream"). Shows up on worker restarts.
        """
        for attempt in range(1, attempts + 1):
            try:
                return dai.Pipeline()
            except RuntimeError as exc:
                if attempt == attempts:
                    raise
                self.logger.warning(
                    f"OAK-D not ready ({exc}) — retry {attempt}/{attempts - 1} "
                    f"in {backoff_s * attempt:.0f}s"
                )
                time.sleep(backoff_s * attempt)

    def _stop_pipeline(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def _reopen(self, reason: str) -> None:
        """Stop and re-open the camera; give up after max_reopen_attempts."""
        self.logger.warning(f"{reason} — re-opening the OAK-D")
        self._stop_pipeline()
        for attempt in range(1, self._max_reopen + 1):
            time.sleep(2.0 * attempt)
            try:
                self._start_device()
                self.logger.info(f"OAK-D re-opened (attempt {attempt})")
                return
            except Exception as exc:
                self._stop_pipeline()
                self.logger.warning(
                    f"Re-open attempt {attempt}/{self._max_reopen} failed: {exc}"
                )
        # the process exits and the orchestrator restarts it
        raise FatalWorkerError(f"OAK-D could not be re-opened after {self._max_reopen} attempts")

    def _check_device_id(self) -> None:
        if self._settings is None:
            self.logger.warning(
                "No ORB-SLAM3 settings to check against the camera "
                "(run scripts/gen_orbslam_settings.py)"
            )
            return
        live_id = self._pipeline.getDefaultDevice().getDeviceId()
        if self._settings["device_id"] and self._settings["device_id"] != live_id:
            self.logger.warning(
                f"ORB-SLAM3 settings were generated for OAK-D {self._settings['device_id']} "
                f"but {live_id} is connected — regenerate them with "
                "scripts/gen_orbslam_settings.py"
            )

    def _check_intrinsics(self, rgb_frame) -> None:
        """Compare the settings' intrinsics with the first frame's (>2 % warns)."""
        self._intrinsics_checked = True
        if self._settings is None:
            return
        try:
            K = np.asarray(rgb_frame.getTransformation().getIntrinsicMatrix())
        except Exception as exc:
            self.logger.debug(f"No intrinsics on the RGB frame ({exc})")
            return
        bad = intrinsics_mismatch(self._settings, K)
        if bad:
            self.logger.warning(
                "ORB-SLAM3 settings don't match the live RGB intrinsics ("
                + ", ".join(bad)
                + ") — regenerate them with scripts/gen_orbslam_settings.py"
            )

    def _setup_imu(self, device, rate_hz: int) -> None:
        """Accelerometer + calibrated gyro always; the rotation vector only on
        a BNO08x (the BMI270 on newer OAK-Ds has no sensor fusion)."""
        imu_type = str(device.getConnectedIMU())
        if imu_type.upper() in ("", "NONE"):
            self.logger.warning("No IMU on this OAK-D — not streaming IMU data")
            return
        sensors = [dai.IMUSensor.ACCELEROMETER, dai.IMUSensor.GYROSCOPE_CALIBRATED]
        self._has_rotation_vector = imu_type.upper().startswith("BNO")
        if self._has_rotation_vector:
            sensors.append(dai.IMUSensor.ROTATION_VECTOR)
        else:
            self.logger.info(
                f"OAK-D IMU is {imu_type}: accel + gyro only, quaternion slots are NaN"
            )
        imu = self._pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(sensors, rate_hz)
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        self._imu_q = imu.out.createOutputQueue(maxSize=50, blocking=False)

    def _drain_imu(self) -> None:
        """Push pending IMU packets as [qw, qx, qy, qz, ax, ay, az, gx, gy] (gyro x/y
        where the I2C path has throttle/steering feedback), stamped with the
        accelerometer capture time. Quaternion is NaN without a rotation vector.
        With a raw_imu_buffer, also [ax, ay, az, gx, gy, gz] per packet."""
        for imu_data in self._imu_q.tryGetAll():
            for packet in imu_data.packets:
                acc = packet.acceleroMeter
                gyro = packet.gyroscope
                if self._has_rotation_vector:
                    rv = packet.rotationVector
                    quat = (rv.real, rv.i, rv.j, rv.k)
                else:
                    quat = (np.nan,) * 4
                ts = acc.getTimestamp().total_seconds()
                if self._imu_buf is not None:
                    vec = np.array(
                        [*quat, acc.x, acc.y, acc.z, gyro.x, gyro.y], dtype=np.float32
                    )
                    self._imu_buf.write(vec, ts=ts)
                if self._raw_imu_buf is not None:
                    # a packet pairs an accel and a gyro sample whose own stamps can differ
                    # by up to a sample period: stamp the pair with their mean
                    t_pair = 0.5 * (ts + gyro.getTimestamp().total_seconds())
                    self._raw_imu_buf.write(
                        np.array([acc.x, acc.y, acc.z, gyro.x, gyro.y, gyro.z], dtype=np.float32),
                        ts=t_pair)

    def step(self) -> None:
        try:
            got = self._step_device()
            self._step_errors = 0
        except Exception:
            self._step_errors += 1
            if self._step_errors >= self._max_step_errors:
                self._reopen(f"{self._step_errors} step errors in a row")
                return
            raise
        timeout = self._frame_timeout if self._got_frame else self._startup_timeout
        if got:
            self._last_frame_t = time.monotonic()
            self._got_frame = True
        elif time.monotonic() - self._last_frame_t > timeout:
            self._reopen(f"no frame for {timeout:.1f}s")

    def _step_device(self) -> bool:
        """Publish the newest synced frame, if any. True if one was published."""
        if self._imu_q is not None:
            self._drain_imu()

        groups = self._sync_q.tryGetAll()
        if not groups:
            time.sleep(0.001)
            return False
        msg_group = groups[-1]  # newest; older ones are stale

        rgb_frame = msg_group["rgb"]
        depth_frame = msg_group["depth"]
        ts = rgb_frame.getTimestamp().total_seconds()  # capture time, monotonic clock

        # DepthAI gives BGR, vision models want RGB
        rgb = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2RGB)
        depth = depth_frame.getFrame().astype(np.float32)
        depth /= 1000.0  # mm -> m

        self._frame_buf.write(rgb, depth, ts=ts)

        if not self._intrinsics_checked:
            self._check_intrinsics(rgb_frame)

        self._frame_count += 1
        if self._frame_count % 100 == 1:
            self.logger.debug(
                f"frame #{self._frame_count}  rgb={rgb.shape}  "
                f"depth={depth.shape} range=[{depth.min():.2f}, {depth.max():.2f}]m  "
                f"latency={1000 * (time.monotonic() - ts):.0f}ms"
            )
        return True

    def teardown(self) -> None:
        if self._pipeline is not None:
            self._stop_pipeline()
            self.logger.info("OAK-D pipeline stopped")
