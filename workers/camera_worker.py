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


RVC2_PRESETS: set[str] = {
    "FAST_ACCURACY",
    "FAST_DENSITY",
    "DEFAULT",
    "FACE",
    "HIGH_DETAIL",
    "ROBOTICS",
}
RVC4_PRESETS: set[str] = {"ACCURACY", "DENSITY"}

PRESET_CANDIDATES: dict[str, list[str]] = {
    "FAST_ACCURACY": ["FAST_ACCURACY"],
    "FAST_DENSITY": ["FAST_DENSITY"],
    "DEFAULT": ["DEFAULT"],
    "FACE": ["FACE"],
    "HIGH_DETAIL": ["HIGH_DETAIL", "HIGH_DENSITY"],
    "ROBOTICS": ["ROBOTICS"],
    "ACCURACY": ["ACCURACY", "HIGH_ACCURACY"],
    "DENSITY": ["DENSITY", "HIGH_DENSITY"],
}


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
        self._imu_sensor_names: list[str] = []
        self._imu_accel: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._imu_gyro: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._imu_mag: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._imu_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self._imu_seen_accel: bool = False
        self._imu_seen_gyro: bool = False
        self._platform: Any = None
        self._depth_preset_requested: str = "FAST_ACCURACY"
        self._depth_preset_effective: str = "FAST_ACCURACY"
        self._depth_preset_sdk: str = "FAST_ACCURACY"
        self._stereo_size: tuple[int, int] = (1280, 800)
        self._sync_threshold_ms: float = 25.0
        self._rgb_undistort: bool = True
        self._xlink_chunk_size: int = 0
        self._current_dot_ir: float = 0.8
        self._current_flood_ir: float = 0.8
        self._adaptive_ir: bool = True
        self._ir_update_interval_s: float = 0.5
        self._last_ir_update: float = 0.0
        self._synthetic = False
        self._synth_gen = None

    #  lifecycle ─

    def setup(self) -> None:
        import os

        fps = self.config.get("fps", 30)
        width, height = self.config.get("resolution", [640, 400])
        size = (width, height)
        stereo_w, stereo_h = self.config.get("stereo_resolution", [1280, 800])
        self._stereo_size = (int(stereo_w), int(stereo_h))
        self._imu_buf = self.config.get("imu_buffer")
        self._rgbd_vis_cfg = self.config.get("rgbd_visualization", {}) or {}
        self._rgbd_vis_requested = bool(self._rgbd_vis_cfg.get("enabled", False))
        self._depth_preset_requested = str(
            self.config.get("depth_preset", "FAST_ACCURACY")
        ).strip().upper()
        self._sync_threshold_ms = float(self.config.get("sync_threshold_ms", 25.0))
        self._rgb_undistort = bool(self.config.get("rgb_undistort", True))
        self._xlink_chunk_size = int(self.config.get("xlink_chunk_size", 0))
        self._current_dot_ir = float(self.config.get("ir_dot_intensity", 0.8))
        self._current_flood_ir = float(self.config.get("ir_flood_intensity", 0.8))
        self._adaptive_ir = bool(self.config.get("adaptive_ir", True))
        self._ir_update_interval_s = float(self.config.get("ir_update_interval_s", 0.5))

        use_synthetic = (
            self.config.get("synthetic", False)
            or os.environ.get("PILOT_SYNTHETIC", "") == "1"
            or dai is None
        )
        if use_synthetic:
            if self._rgbd_vis_requested:
                # Keep the setup path observable when visualization is enabled,
                # even if the worker falls back to synthetic mode.
                self._setup_remote_rgbd_visualization(
                    color_camera=None,
                    depth_source=None,
                    size=size,
                    fps=fps,
                )
            self._setup_synthetic()
            return

        if dai is None:
            raise ImportError("depthai >= 3.0 is required for CameraWorker")

        self._frame_buf = self.config["output_buffer"]

        # DepthAI pipeline: require v3 builder API.
        self._pipeline = dai.Pipeline()
        self._pipeline.setXLinkChunkSize(self._xlink_chunk_size)
        platform = "unknown"
        if hasattr(self._pipeline, "getDefaultDevice"):
            try:
                self._platform = self._pipeline.getDefaultDevice().getPlatform()
                platform = str(self._platform)
            except Exception:
                platform = "unknown"

        if not hasattr(dai.node.Camera, "build"):
            raise RuntimeError(
                "DepthAI v3 builder API is required (dai.node.Camera.build missing)"
            )
        self._setup_pipeline_builder_v3(size=size, fps=fps)

        if hasattr(self._pipeline, "start"):
            self._pipeline.start()
            if self._device is None and hasattr(self._pipeline, "getDefaultDevice"):
                try:
                    self._device = self._pipeline.getDefaultDevice()
                except Exception:
                    self._device = None
        else:
            self._device = dai.Device(self._pipeline)
            if self._rgb_q is None:
                self._rgb_q = self._device.getOutputQueue("rgb", maxSize=8, blocking=False)
            if self._depth_q is None:
                self._depth_q = self._device.getOutputQueue("depth", maxSize=8, blocking=False)
            if self._imu_buf is not None and self._imu_q is None:
                self._imu_q = self._device.getOutputQueue("imu", maxSize=32, blocking=False)

        try:
            self._apply_ir_levels(self._current_dot_ir, self._current_flood_ir)
        except Exception:
            pass

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
        self.logger.info(
            "Camera settings: depth_preset(requested=%s,effective=%s,sdk=%s) "
            "stereo=%s sync_threshold_ms=%.1f rgb_undistort=%s xlink_chunk_size=%d "
            "ir(dot=%.2f,flood=%.2f,adaptive=%s)",
            self._depth_preset_requested,
            self._depth_preset_effective,
            self._depth_preset_sdk,
            self._stereo_size,
            self._sync_threshold_ms,
            self._rgb_undistort,
            self._xlink_chunk_size,
            self._current_dot_ir,
            self._current_flood_ir,
            self._adaptive_ir,
        )
        if self._imu_q is not None:
            self.logger.info("OAK-D IMU stream enabled in CameraWorker")
            if self._imu_sensor_names:
                self.logger.info(
                    "OAK-D IMU sensors: %s",
                    ", ".join(self._imu_sensor_names),
                )

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

        now = time.time()
        if self._adaptive_ir and (now - self._last_ir_update) >= self._ir_update_interval_s:
            mean_luma = float(rgb.mean())
            dot_ir, flood_ir = self._ir_levels_from_luma(mean_luma)
            if (
                abs(dot_ir - self._current_dot_ir) >= 0.05
                or abs(flood_ir - self._current_flood_ir) >= 0.05
            ):
                self._apply_ir_levels(dot_ir, flood_ir)
            self._last_ir_update = now

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

        def _first_attr(obj: Any, names: list[str]) -> Any:
            for n in names:
                if hasattr(obj, n):
                    return getattr(obj, n)
            return None

        def _xyz(src: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
            if src is None:
                return default
            x = getattr(src, "x", default[0])
            y = getattr(src, "y", default[1])
            z = getattr(src, "z", default[2])
            return float(x), float(y), float(z)
            # sehsuuehhsueh suue ud euu8pYYUs.
        while True:
            imu_data = self._imu_q.tryGet()
            if imu_data is None:
                break

            for packet in imu_data.packets:
                accel = _first_attr(packet, ["acceleroMeter", "accelerometer"])
                gyro = _first_attr(packet, ["gyroscope", "gyroScope", "gyroscopeCalibrated"])
                mag = _first_attr(packet, ["magneticField", "magnetometer"])
                rot = _first_attr(packet, ["rotationVector", "rotation"])

                if accel is not None:
                    self._imu_accel = _xyz(accel, self._imu_accel)
                    self._imu_seen_accel = True
                if gyro is not None:
                    self._imu_gyro = _xyz(gyro, self._imu_gyro)
                    self._imu_seen_gyro = True
                if mag is not None:
                    self._imu_mag = _xyz(mag, self._imu_mag)
                if rot is not None:
                    # DepthAI rotation vectors are quaternion-like (real, i, j, k).
                    self._imu_quat = (
                        float(getattr(rot, "real", self._imu_quat[0])),
                        float(getattr(rot, "i", self._imu_quat[1])),
                        float(getattr(rot, "j", self._imu_quat[2])),
                        float(getattr(rot, "k", self._imu_quat[3])),
                    )

                # Publish only after at least one accel and gyro sample have been seen.
                if not (self._imu_seen_accel and self._imu_seen_gyro):
                    continue

                ax, ay, az = self._imu_accel
                gx, gy, gz = self._imu_gyro
                mx, my, mz = self._imu_mag
                qw, qx, qy, qz = self._imu_quat

                # Use host epoch time to align with FrameBuffer produced_ts.
                ts = float(time.time())

                vec = np.array(
                    [
                        qw, qx, qy, qz,
                        ax, ay, az,
                        gx, gy, gz,
                        mx, my, mz,
                        0.0, 0.0,  # OAK-D IMU path: throttle/steering are always 0
                        ts,
                    ],
                    dtype=np.float32,
                )
                self._imu_buf.write(vec)
                self._imu_count += 1

                if self._imu_count % 200 == 1:
                    self.logger.debug(
                        f"imu #{self._imu_count} "
                        f"accel=[{ax:.4f},{ay:.4f},{az:.4f}] "
                        f"gyro=[{gx:.4f},{gy:.4f},{gz:.4f}] "
                        f"mag=[{mx:.4f},{my:.4f},{mz:.4f}] "
                        f"quat=[{qw:.4f},{qx:.4f},{qy:.4f},{qz:.4f}]"
                    )

    @staticmethod
    def _is_rvc4_platform(platform: Any) -> bool:
        return "RVC4" in str(platform)

    @staticmethod
    def _ir_levels_from_luma(mean_luma: float) -> tuple[float, float]:
        if mean_luma >= 185.0:
            return 0.05, 0.10
        if mean_luma >= 150.0:
            return 0.20, 0.25
        if mean_luma >= 110.0:
            return 0.45, 0.50
        if mean_luma >= 75.0:
            return 0.65, 0.70
        return 0.85, 0.90

    def _resolve_depth_preset(self) -> tuple[Any, str, str]:
        mode_cls = dai.node.StereoDepth.PresetMode
        available: dict[str, Any] = {
            name: getattr(mode_cls, name)
            for name in dir(mode_cls)
            if name.isupper() and not name.startswith("_")
        }

        requested = self._depth_preset_requested
        if requested not in PRESET_CANDIDATES:
            requested = "FAST_ACCURACY"

        effective = requested
        if self._is_rvc4_platform(self._platform) and requested in RVC2_PRESETS:
            effective = "ACCURACY"
        elif not self._is_rvc4_platform(self._platform) and requested in RVC4_PRESETS:
            effective = "FAST_ACCURACY"

        for candidate in PRESET_CANDIDATES[effective]:
            if candidate in available:
                return available[candidate], effective, candidate

        for fallback in ("FAST_ACCURACY", "DEFAULT", "ACCURACY", "DENSITY"):
            for candidate in PRESET_CANDIDATES[fallback]:
                if candidate in available:
                    return available[candidate], effective, candidate

        raise RuntimeError("No compatible StereoDepth preset found in this DepthAI build")

    def _apply_ir_levels(self, dot: float, flood: float) -> None:
        if self._device is None:
            return
        self._device.setIrLaserDotProjectorIntensity(dot)
        self._device.setIrFloodLightIntensity(flood)
        self._current_dot_ir = dot
        self._current_flood_ir = flood

    def _setup_pipeline_builder_v3(self, size: tuple[int, int], fps: int) -> None:
        """DepthAI v3 builder API path."""
        # Keep node creation order explicit to mirror the intended graph:
        # CAM_A (color), CAM_B/CAM_C (mono pair), StereoDepth, Sync, RGBD, IMU.
        cam_rgb = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps
        )
        cam_1 = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=fps
        )
        cam_2 = self._pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=fps
        )

        stereo = self._pipeline.create(dai.node.StereoDepth)
        selected_preset, effective_name, sdk_name = self._resolve_depth_preset()
        stereo.setDefaultProfilePreset(selected_preset)
        self._depth_preset_effective = effective_name
        self._depth_preset_sdk = sdk_name
        stereo.setLeftRightCheck(True)
        stereo.setRectification(True)

        align = (
            self._pipeline.create(dai.node.ImageAlign)
            if self._is_rvc4_platform(self._platform)
            else None
        )

        self._setup_remote_rgbd_visualization(
            color_camera=cam_rgb,
            depth_source=stereo,
            size=size,
            fps=fps,
        )

        if self._imu_buf is not None:
            imu = self._pipeline.create(dai.node.IMU)

            # Enable accel+gyro together so packets contain the inertial pair
            # required by SLAM. Add optional sensors when available.
            self._imu_sensor_names = []
            enabled_pair = False
            enabled_sensors: list[Any] = []
            sensor_pairs = [
                ("ACCELEROMETER_RAW", "GYROSCOPE_RAW"),
                ("ACCELEROMETER_RAW", "GYROSCOPE_CALIBRATED"),
                ("ACCELEROMETER", "GYROSCOPE_RAW"),
                ("ACCELEROMETER", "GYROSCOPE_CALIBRATED"),
            ]
            for accel_name, gyro_name in sensor_pairs:
                accel_sensor = getattr(dai.IMUSensor, accel_name, None)
                gyro_sensor = getattr(dai.IMUSensor, gyro_name, None)
                if accel_sensor is None or gyro_sensor is None:
                    continue
                enabled_sensors = [accel_sensor, gyro_sensor]
                self._imu_sensor_names = [accel_name, gyro_name]
                enabled_pair = True
                break

            if not enabled_pair:
                self.logger.warning(
                    "Failed to enable accel+gyro IMU pair; IMU stream may be incomplete"
                )

            for sensor_name in ["MAGNETOMETER_CALIBRATED", "ROTATION_VECTOR"]:
                sensor = getattr(dai.IMUSensor, sensor_name, None)
                if sensor is not None:
                    enabled_sensors.append(sensor)
                    self._imu_sensor_names.append(sensor_name)

            if enabled_sensors:
                try:
                    imu.enableIMUSensor(enabled_sensors, 200)
                except Exception:
                    self.logger.warning(
                        "Failed to enable requested IMU sensors in combined mode"
                    )

            imu.setBatchReportThreshold(1)
            imu.setMaxBatchReports(10)
            self._imu_q = imu.out.createOutputQueue(maxSize=32, blocking=False)

        # Match stereo_example settings: CAM_B -> left, CAM_C -> right.
        cam_1.requestOutput(self._stereo_size, fps=fps).link(stereo.left)
        cam_2.requestOutput(self._stereo_size, fps=fps).link(stereo.right)
        try:
            rgb_out = cam_rgb.requestOutput(size, fps=fps, enableUndistortion=self._rgb_undistort)
        except TypeError:
            rgb_out = cam_rgb.requestOutput(size, fps=fps)

        if align is not None:
            stereo.depth.link(align.input)
            rgb_out.link(align.inputAlignTo)
            depth_out = align.outputAligned
        else:
            depth_out = stereo.depth
            if hasattr(stereo, "inputAlignTo"):
                rgb_out.link(stereo.inputAlignTo)
            elif hasattr(stereo, "setDepthAlign"):
                stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

        sync = self._pipeline.create(dai.node.Sync)
        sync.setSyncThreshold(timedelta(milliseconds=self._sync_threshold_ms))
        rgb_out.link(sync.inputs["rgb"])
        try:
            sync.inputs["rgb"].setBlocking(False)
        except Exception:
            pass
        depth_out.link(sync.inputs["depth"])
        self._sync_q = sync.out.createOutputQueue(maxSize=4, blocking=False)

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

        if self._pipeline is None or color_camera is None or depth_source is None:
            self.logger.warning(
                "RGBD visualization requested, but camera pipeline is unavailable "
                "(synthetic mode or DepthAI unavailable)."
            )
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
