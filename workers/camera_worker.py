"""
CameraWorker — captures RGB + aligned Depth frames from an OAK-D camera via
DepthAI v3 and writes them into a FrameBuffer.

Uses the DepthAI v3 builder API with three camera nodes (RGB + stereo pair)
and a StereoDepth node for depth estimation:
    color  = pipeline.create(dai.node.Camera).build(CAM_A)
    left   = pipeline.create(dai.node.Camera).build(CAM_B)
    right  = pipeline.create(dai.node.Camera).build(CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

When an ``imu_buffer`` is supplied, the on-board IMU (BNO086 / BMI270) is
added to the *same* pipeline.  A DepthAI device can only be opened by one
process, so the OAK-D IMU must live here rather than in a separate worker.

This worker does **no** model inference — it only captures and publishes frames.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import cv2
import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer

try:
    import depthai as dai
except ImportError:
    dai = None  # allow import on machines without depthai for testing


class CameraWorker(BaseWorker):
    """Worker process that captures OAK-D RGB+Depth frames via DepthAI v3.

    Falls back to synthetic FPV frames when depthai is unavailable or
    ``synthetic=True`` is passed in config (dev-local mode).

    Config kwargs (passed via ``start.py``)
    ----------------------------------------
    output_buffer : FrameBuffer
        Shared FrameBuffer to write into.
    fps : int
        Target camera FPS (default 30).
    resolution : list[int]
        ``[width, height]`` for RGB and depth output (default [640, 400]).
    synthetic : bool
        Force synthetic frame generation (default False).
    imu_buffer : IMUBuffer | None
        If given, stream the OAK-D on-board IMU into this buffer.
    imu_rate_hz : int
        IMU report rate (default 100).
    """

    def __init__(self, name: str = "camera", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._pipeline = None
        self._sync_q = None
        self._frame_count: int = 0
        self._synthetic = False
        self._synth_gen = None
        self._imu_buf = None
        self._imu_q = None

    # --- lifecycle ---

    def setup(self) -> None:
        import os
        use_synthetic = (
            self.config.get("synthetic", False)
            or os.environ.get("PILOT_SYNTHETIC", "") == "1"
            or dai is None
        )
        if use_synthetic:
            self._setup_synthetic()
            return

        if dai is None:
            raise ImportError("depthai >= 3.0 is required for CameraWorker")

        fps = self.config.get("fps", 30)
        width, height = self.config.get("resolution", [640, 400])
        size = (width, height)

        self._frame_buf = self.config["output_buffer"]

        # --- DepthAI v3 pipeline ---
        self._pipeline = self._open_pipeline()
        platform = self._pipeline.getDefaultDevice().getPlatform()

        # Colour camera (CAM_A)
        cam_rgb = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps
        )
        # Stereo pair (CAM_B = left, CAM_C = right)
        cam_left = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=fps
        )
        cam_right = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=fps
        )

        # StereoDepth node
        stereo = self._pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)

        # Link stereo pair → StereoDepth
        cam_left.requestOutput(size, fps=fps).link(stereo.left)
        cam_right.requestOutput(size, fps=fps).link(stereo.right)

        # RGB output — undistorted so stereo depth aligns to it exactly
        rgb_out = cam_rgb.requestOutput(size, fps=fps, enableUndistortion=True)

        # Align depth to RGB
        if platform == dai.Platform.RVC4:
            align = self._pipeline.create(dai.node.ImageAlign)
            stereo.depth.link(align.input)
            rgb_out.link(align.inputAlignTo)
            depth_out = align.outputAligned
        else:
            # RVC2 — align via stereo node directly
            rgb_out.link(stereo.inputAlignTo)
            depth_out = stereo.depth

        # Sync RGB + depth so we get matched pairs
        sync = self._pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(seconds=1.0 / (2 * fps)))
        rgb_out.link(sync.inputs["rgb"])
        depth_out.link(sync.inputs["depth"])
        self._sync_q = sync.out.createOutputQueue()

        self._imu_buf = self.config.get("imu_buffer")
        if self._imu_buf is not None:
            self._setup_imu(self.config.get("imu_rate_hz", 100))

        self._pipeline.start()
        self.logger.info(
            f"OAK-D camera ready ({platform}) @ {fps} FPS, {width}×{height}"
            + (" + on-board IMU" if self._imu_q is not None else "")
        )

    def _open_pipeline(self, attempts: int = 5, backoff_s: float = 1.0):
        """Open the device, retrying while it re-enumerates.

        After a pipeline is closed the OAK-D reboots and briefly cannot be
        reopened ("Couldn't open stream"), e.g. on a worker restart.
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

    def _setup_imu(self, rate_hz: int) -> None:
        """Add the on-board IMU node to the camera pipeline."""
        imu = self._pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [
                dai.IMUSensor.ACCELEROMETER,
                dai.IMUSensor.GYROSCOPE_CALIBRATED,
                dai.IMUSensor.ROTATION_VECTOR,
            ],
            rate_hz,
        )
        imu.setBatchReportThreshold(1)
        imu.setMaxBatchReports(10)
        self._imu_q = imu.out.createOutputQueue(maxSize=50, blocking=False)

    def _drain_imu(self) -> None:
        """Publish all pending IMU packets.

        Vector layout matches the I2C IMU path:
        ``[qw, qx, qy, qz, ax, ay, az, gx, gy]`` (accel in m/s², gyro rad/s).
        The final two slots carry throttle/steering feedback on the I2C
        path; the OAK-D has no such feedback, so gyro x/y are reported.
        """
        for imu_data in self._imu_q.tryGetAll():
            for packet in imu_data.packets:
                acc = packet.acceleroMeter
                gyro = packet.gyroscope
                rv = packet.rotationVector
                vec = np.array(
                    [rv.real, rv.i, rv.j, rv.k, acc.x, acc.y, acc.z, gyro.x, gyro.y],
                    dtype=np.float32,
                )
                self._imu_buf.write(vec)

    def step(self) -> None:
        if self._synthetic:
            self._step_synthetic()
            return

        if self._imu_q is not None:
            self._drain_imu()

        msg_group = self._sync_q.tryGet()
        if msg_group is None:
            time.sleep(0.001)
            return

        rgb_frame = msg_group["rgb"]
        depth_frame = msg_group["depth"]

        # DepthAI delivers BGR; FrameBuffer carries RGB (vision models expect RGB)
        rgb = cv2.cvtColor(rgb_frame.getCvFrame(), cv2.COLOR_BGR2RGB)
        depth = depth_frame.getFrame().astype(np.float32)
        depth /= 1000.0  # mm → metres

        self._frame_buf.write(rgb, depth)

        self._frame_count += 1
        if self._frame_count % 100 == 1:
            self.logger.debug(
                f"frame #{self._frame_count}  rgb={rgb.shape}  "
                f"depth={depth.shape} range=[{depth.min():.2f}, {depth.max():.2f}]m"
            )

    # --- synthetic mode ---

    def _setup_synthetic(self) -> None:
        """Init synthetic FPV frame generator (no hardware needed)."""
        from synthetic import SyntheticFrameGenerator

        fps = self.config.get("fps", 30)
        width, height = self.config.get("resolution", [640, 400])
        self._frame_buf = self.config["output_buffer"]
        self._synthetic = True
        self._synth_gen = SyntheticFrameGenerator(width=width, height=height)
        self._target_dt = 1.0 / fps
        self.logger.info(
            f"Synthetic camera ready @ {fps} FPS, {width}×{height}"
        )

    def _step_synthetic(self) -> None:
        rgb, depth = self._synth_gen.next_fast()
        self._frame_buf.write(rgb, depth)
        self._frame_count += 1
        if self._frame_count % 100 == 1:
            self.logger.debug(
                f"synth frame #{self._frame_count}  rgb={rgb.shape}  "
                f"depth range=[{depth.min():.2f}, {depth.max():.2f}]m"
            )
        time.sleep(self._target_dt)

    def teardown(self) -> None:
        if self._synthetic:
            self.logger.info("Synthetic camera stopped")
            return
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self.logger.info("OAK-D pipeline stopped")
