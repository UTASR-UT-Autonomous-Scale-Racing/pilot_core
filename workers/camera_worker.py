"""
CameraWorker — captures RGB + aligned Depth frames from an OAK-D camera via
DepthAI v3 and writes them into a FrameBuffer.

Uses the DepthAI v3 builder API with three camera nodes (RGB + stereo pair)
and a StereoDepth node for depth estimation:
    color  = pipeline.create(dai.node.Camera).build(CAM_A)
    left   = pipeline.create(dai.node.Camera).build(CAM_B)
    right  = pipeline.create(dai.node.Camera).build(CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

This worker does **no** model inference — it only captures and publishes frames.
"""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer

try:
    from shared_buffer.imu_buffer import IMUBuffer
except Exception:
    IMUBuffer = None  # type: ignore

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
    imu_buffer : IMUBuffer | None
        Optional IMUBuffer. When provided, IMU samples are read from OAK-D
        in the same DepthAI v3 pipeline and written as 16-float vectors.
    fps : int
        Target camera FPS (default 30).
    resolution : list[int]
        ``[width, height]`` for RGB and depth output (default [640, 400]).
    synthetic : bool
        Force synthetic frame generation (default False).
    """

    def __init__(self, name: str = "camera", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._pipeline = None
        self._device = None
        self._sync_q = None
        self._rgb_q = None
        self._depth_q = None
        self._imu_q = None
        self._imu_buf = None
        self._remote_connector = None
        self._rgbd_vis_cfg: dict[str, Any] = {}
        self._rgbd_vis_requested: bool = False
        self._latest_rgb_msg = None
        self._latest_depth_msg = None
        self._frame_count: int = 0
        self._imu_count: int = 0
        self._synthetic = False
        self._synth_gen = None

    #  lifecycle ─

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
        self._imu_buf = self.config.get("imu_buffer")
        self._rgbd_vis_cfg = self.config.get("rgbd_visualization", {}) or {}
        self._rgbd_vis_requested = bool(self._rgbd_vis_cfg.get("enabled", False))

        self._frame_buf = self.config["output_buffer"]

        # DepthAI pipeline: require v3 builder API.
        self._pipeline = dai.Pipeline()
        platform = "unknown"
        if hasattr(self._pipeline, "getDefaultDevice"):
            try:
                platform = str(self._pipeline.getDefaultDevice().getPlatform())
            except Exception:
                platform = "unknown"

        if not hasattr(dai.node.Camera, "build"):
            raise RuntimeError(
                "DepthAI v3 builder API is required (dai.node.Camera.build missing)"
            )
        self._setup_pipeline_builder_v3(size=size, fps=fps)

        if hasattr(self._pipeline, "start"):
            self._pipeline.start()
        else:
            self._device = dai.Device(self._pipeline)
            if self._rgb_q is None:
                self._rgb_q = self._device.getOutputQueue("rgb", maxSize=8, blocking=False)
            if self._depth_q is None:
                self._depth_q = self._device.getOutputQueue("depth", maxSize=8, blocking=False)
            if self._imu_buf is not None and self._imu_q is None:
                self._imu_q = self._device.getOutputQueue("imu", maxSize=32, blocking=False)

        if self._remote_connector is not None:
            try:
                self._remote_connector.registerPipeline(self._pipeline)
                self.logger.info(
                    "RGBD remote visualization enabled on ws=%s http=%s",
                    self._rgbd_vis_cfg.get("web_socket_port", 8765),
                    self._rgbd_vis_cfg.get("http_port", 8082),
                )
            except Exception as exc:
                self.logger.warning(
                    "Failed to register RGBD remote visualization pipeline: %s",
                    exc,
                )
                self._remote_connector = None

        self.logger.info(
            f"OAK-D camera ready ({platform}) @ {fps} FPS, {width}×{height}"
        )
        if self._imu_q is not None:
            self.logger.info("OAK-D IMU stream enabled in CameraWorker")

    def step(self) -> None:
        if self._synthetic:
            self._step_synthetic()
            return

        if self._remote_connector is not None:
            try:
                key = self._remote_connector.waitKey(1)
                if key == ord("q"):
                    self.logger.info("Remote visualization requested quit via 'q'")
            except Exception:
                pass

        if self._sync_q is not None:
            msg_group = self._sync_q.tryGet()
            if msg_group is None:
                time.sleep(0.001)
                return
            rgb_frame = msg_group["rgb"]
            depth_frame = msg_group["depth"]
        else:
            if self._rgb_q is None or self._depth_q is None:
                time.sleep(0.001)
                return
            rgb_try = self._rgb_q.tryGet()
            depth_try = self._depth_q.tryGet()
            if rgb_try is not None:
                self._latest_rgb_msg = rgb_try
            if depth_try is not None:
                self._latest_depth_msg = depth_try
            if self._latest_rgb_msg is None or self._latest_depth_msg is None:
                time.sleep(0.001)
                return
            rgb_frame = self._latest_rgb_msg
            depth_frame = self._latest_depth_msg

        rgb = rgb_frame.getCvFrame().astype(np.uint8)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        depth = depth_frame.getFrame().astype(np.float32)
        depth /= 1000.0  # mm → metres

        self._frame_buf.write(rgb, depth)
        self._drain_imu_queue()

        self._frame_count += 1
        if self._frame_count % 100 == 1:
            self.logger.debug(
                f"frame #{self._frame_count}  rgb={rgb.shape}  "
                f"depth={depth.shape} range=[{depth.min():.2f}, {depth.max():.2f}]m"
            )

    # ── synthetic mode ─────────────────────────────────────────────

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

    def _drain_imu_queue(self) -> None:
        """Read all pending IMU packets and write 16-float vectors."""
        if self._imu_q is None or self._imu_buf is None:
            return

        while True:
            imu_data = self._imu_q.tryGet()
            if imu_data is None:
                break

            for packet in imu_data.packets:
                accel = packet.acceleroMeter
                gyro = packet.gyroscope
                vec = np.array(
                    [
                        0.0, 0.0, 0.0, 0.0,  # quaternion placeholder (w,x,y,z)
                        float(accel.x), float(accel.y), float(accel.z),
                        float(gyro.x), float(gyro.y), float(gyro.z),
                        0.0, 0.0, 0.0,  # magnetometer placeholder (mx,my,mz)
                        0.0, 0.0,  # raw throttle, steering placeholders
                        float(time.time()),
                    ],
                    dtype=np.float32,
                )
                self._imu_buf.write(vec)
                self._imu_count += 1

                if self._imu_count % 200 == 1:
                    self.logger.debug(
                        f"imu #{self._imu_count} "
                        f"accel=[{accel.x:.4f},{accel.y:.4f},{accel.z:.4f}] "
                        f"gyro=[{gyro.x:.4f},{gyro.y:.4f},{gyro.z:.4f}]"
                    )

    def _setup_pipeline_builder_v3(self, size: tuple[int, int], fps: int) -> None:
        """DepthAI v3 builder API path."""
        cam_rgb = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps
        )
        cam_left = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=fps
        )
        cam_right = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=fps
        )

        stereo = self._pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)

        self._setup_remote_rgbd_visualization(
            color_camera=cam_rgb,
            depth_source=stereo,
            size=size,
            fps=fps,
        )

        if self._imu_buf is not None:
            imu = self._pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                200,
            )
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            self._imu_q = imu.out.createOutputQueue(maxSize=32, blocking=False)

        cam_left.requestOutput(size, fps=fps).link(stereo.left)
        cam_right.requestOutput(size, fps=fps).link(stereo.right)
        rgb_out = cam_rgb.requestOutput(size, fps=fps)

        # Keep depth in stereo camera coordinates to avoid hard-failing on
        # devices that lack full extrinsic calibration links for alignment.
        depth_out = stereo.depth

        sync = self._pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(seconds=1.0 / (2 * fps)))
        rgb_out.link(sync.inputs["rgb"])
        depth_out.link(sync.inputs["depth"])
        self._sync_q = sync.out.createOutputQueue()

    def _setup_remote_rgbd_visualization(
        self,
        color_camera: Any,
        depth_source: Any,
        size: tuple[int, int],
        fps: int,
    ) -> None:
        """Optional remote RGBD visualisation via DepthAI RemoteConnection."""
        if not self._rgbd_vis_requested:
            return

        if not hasattr(dai, "RemoteConnection") or not hasattr(dai.node, "RGBD"):
            self.logger.warning(
                "RGBD visualization requested but RemoteConnection/RGBD node is unavailable"
            )
            return

        depth_source_name = str(self._rgbd_vis_cfg.get("depth_source", "stereo")).lower()
        if depth_source_name != "stereo":
            self.logger.warning(
                "rgbd_visualization.depth_source=%s not supported in live camera worker; "
                "falling back to stereo",
                depth_source_name,
            )

        ws_port = int(self._rgbd_vis_cfg.get("web_socket_port", 8765))
        http_port = int(self._rgbd_vis_cfg.get("http_port", 8082))

        try:
            self._remote_connector = dai.RemoteConnection(
                webSocketPort=ws_port,
                httpPort=http_port,
            )
            rgbd = self._pipeline.create(dai.node.RGBD).build(
                color_camera,
                depth_source,
                size,
                fps,
            )
            self._remote_connector.addTopic("pcl", rgbd.pcl, "common")
        except Exception as exc:
            self.logger.warning("Failed to enable RGBD visualization: %s", exc)
            self._remote_connector = None

    def _setup_pipeline_classic(self, size: tuple[int, int], fps: int) -> None:
        """Classic DepthAI API fallback path."""
        raise NotImplementedError("Classic DepthAI API is not implemented yet")
        width, height = size

        cam_rgb = self._pipeline.create(dai.node.ColorCamera)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam_rgb.setInterleaved(False)
        cam_rgb.setPreviewSize(width, height)
        cam_rgb.setFps(fps)

        cam_left = self._pipeline.create(dai.node.MonoCamera)
        cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        cam_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_left.setFps(fps)

        cam_right = self._pipeline.create(dai.node.MonoCamera)
        cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        cam_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        cam_right.setFps(fps)

        stereo = self._pipeline.create(dai.node.StereoDepth)

        cam_left.out.link(stereo.left)
        cam_right.out.link(stereo.right)

        xout_rgb = self._pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = self._pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # Queues are bound after device start in setup().

        if self._imu_buf is not None:
            imu = self._pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(
                [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW],
                200,
            )
            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)

            xout_imu = self._pipeline.create(dai.node.XLinkOut)
            xout_imu.setStreamName("imu")
            imu.out.link(xout_imu.input)
            # Queue is bound after device start in setup().

    def teardown(self) -> None:
        if self._synthetic:
            self.logger.info("Synthetic camera stopped")
            return
        if self._pipeline is not None:
            if hasattr(self._pipeline, "stop"):
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
            self.logger.info("OAK-D pipeline stopped")
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
