#!/usr/bin/env python3
"""
start.py — Central orchestrator for the pilot_core autonomous racecar pipeline.

Responsibilities:
    1. Parse config.yaml + CLI arguments
    2. Create shared-memory ring buffers in the parent process
    3. Spawn each worker as a multiprocessing.Process with kwargs
    4. Monitor worker health (restart on crash)
    5. Handle SIGINT / SIGTERM for graceful shutdown

Usage:
    python start.py                            # uses config.yaml defaults
    python start.py --config my_config.yaml    # custom config
    python start.py --recording                # enable recording
    python start.py --debug                    # verbose logging
    python start.py --enable-slam              # enable SLAM + mapping pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import yaml

#  Shared buffers 
from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    IMUBuffer,
    PointCloudBuffer,
    PoseBuffer,
    SegmentationBuffer,
    TrackBoundaryBuffer,
)

# Workers 
from workers import (
    ActuatorWorker,
    BaseWorker,
    BufferRecorderWorker,
    CameraWorker,
    ControlWorker,
    IMUWorker,
    MappingWorker,
    MapPlotWorker,
    RecorderWorker,
    SLAMWorker,
    VisionWorker,
    WebPublishWorker,
)

# 
# Config helpers
# 

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config file and return as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pilot_core — autonomous racecar inference pipeline"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--recording", "-r",
        action="store_true",
        help="Enable data recording",
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--enable-slam",
        action="store_true",
        help="Enable SLAM + mapping + PID pipeline",
    )
    parser.add_argument(
        "--oakd-imu",
        action="store_true",
        help="Use OAK-D on-board IMU instead of I2C Arduino IMU",
    )
    parser.add_argument(
        "--no-actuator-i2c",
        action="store_true",
        help="Disable actuator I2C writes (run actuator worker in dummy mode)",
    )
    parser.add_argument(
        "--web-publish",
        action="store_true",
        help="Enable WebPublishWorker regardless of config file",
    )
    parser.add_argument(
        "--web-all",
        action="store_true",
        help="Publish all supported web payload sections",
    )
    parser.add_argument("--pose", action="store_true", help="Publish pose")
    parser.add_argument("--trajectory", action="store_true", help="Publish trajectory")
    parser.add_argument("--control", action="store_true", help="Publish control")
    parser.add_argument("--track", action="store_true", help="Publish track boundaries")
    parser.add_argument("--imu", action="store_true", help="Publish IMU")
    parser.add_argument("--pointcloud", action="store_true", help="Publish point cloud")
    parser.add_argument(
        "--depth-preset",
        type=str,
        help="Stereo preset (FAST_ACCURACY, FAST_DENSITY, DEFAULT, FACE, HIGH_DETAIL, ROBOTICS, ACCURACY, DENSITY)",
    )
    parser.add_argument(
        "--stereo-resolution",
        type=str,
        help="Stereo mono resolution as WxH (for example 1280x800)",
    )
    parser.add_argument(
        "--sync-threshold-ms",
        type=float,
        help="RGB/depth sync threshold in milliseconds",
    )
    parser.add_argument(
        "--ir-dot-intensity",
        type=float,
        help="Initial IR dot projector intensity [0.0..1.0]",
    )
    parser.add_argument(
        "--ir-flood-intensity",
        type=float,
        help="Initial IR flood intensity [0.0..1.0]",
    )
    parser.add_argument(
        "--xlink-chunk-size",
        type=int,
        help="DepthAI XLink chunk size (0 disables chunking)",
    )
    parser.add_argument(
        "--ir-update-interval-s",
        type=float,
        help="Adaptive IR update interval in seconds",
    )
    parser.add_argument(
        "--adaptive-ir",
        action="store_true",
        help="Enable adaptive IR based on scene brightness",
    )
    parser.add_argument(
        "--no-adaptive-ir",
        action="store_true",
        help="Disable adaptive IR updates",
    )
    parser.add_argument(
        "--rgb-undistort",
        action="store_true",
        help="Enable RGB undistortion on camera output",
    )
    parser.add_argument(
        "--no-rgb-undistort",
        action="store_true",
        help="Disable RGB undistortion on camera output",
    )
    return parser.parse_args()


def _parse_resolution_arg(raw: str) -> List[int]:
    value = raw.lower().replace(" ", "")
    if "x" not in value:
        raise ValueError("resolution must be in WxH format")
    w_str, h_str = value.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError("resolution values must be positive")
    return [w, h]


def _load_track_map(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load pre-mapped boundaries from a .npz or .json map file."""
    lower = path.lower()
    if lower.endswith(".npz"):
        with np.load(path) as data:
            outer = np.asarray(data["outer"], dtype=np.float32)
            inner = np.asarray(data["inner"], dtype=np.float32)
        return outer, inner
    if lower.endswith(".json"):
        with open(path, "r") as fp:
            payload = json.load(fp)
        outer = np.asarray(payload.get("outer", []), dtype=np.float32)
        inner = np.asarray(payload.get("inner", []), dtype=np.float32)
        return outer, inner
    raise ValueError("Unsupported map format; use .npz or .json")


def _resolve_recording_path(path_template: str) -> str:
        """Resolve recording path templates and user home.

        Supported placeholders:
            - {datetime}: replaced with local timestamp YYYYmmdd_HHMMSS.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path_template.replace("{datetime}", ts)
        return os.path.abspath(os.path.expanduser(path))


# 
# Global state for signal handling
# 

_workers: List[BaseWorker] = []
_processes: List[mp.Process] = []
_buffers: List[Any] = []
_shutting_down = False
MAX_RESTARTS = 3


def _shutdown(signum=None, frame=None) -> None:
    """Gracefully stop all workers and clean up shared memory."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    logging.info("Shutdown signal received — stopping workers …")

    # Signal all workers to stop
    for w in _workers:
        w.stop()

    # Wait for processes to finish
    for p in _processes:
        p.join(timeout=5)
        if p.is_alive():
            logging.warning(f"Worker {p.name} did not exit in time — terminating")
            p.terminate()
            p.join(timeout=2)

    # Release shared memory
    for buf in _buffers:
        try:
            buf.unlink()
        except Exception:
            pass

    logging.info("All workers stopped. Exiting.")
    sys.exit(0)


# 
# Main
# 

def main() -> None:
    args = parse_args()

    # Ensure all mp primitives (Event, Value, Process) use the spawn context
    # so that semaphores/locks are properly shared with spawned child processes.
    mp.set_start_method("spawn", force=True)

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("start")

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Merge CLI overrides
    if args.recording:
        config["recording"] = True
    if args.enable_slam:
        config["enable_slam"] = True
    if args.oakd_imu:
        config["use_oakd_imu"] = True
    if args.no_actuator_i2c:
        config["actuator_dummy"] = True
    if args.web_publish:
        config.setdefault("web_publish", {})
        config["web_publish"]["enabled"] = True
    if args.depth_preset:
        config["depth_preset"] = str(args.depth_preset).upper()
    if args.stereo_resolution:
        config["stereo_resolution"] = _parse_resolution_arg(args.stereo_resolution)
    if args.sync_threshold_ms is not None:
        config["sync_threshold_ms"] = float(args.sync_threshold_ms)
    if args.ir_dot_intensity is not None:
        config["ir_dot_intensity"] = float(args.ir_dot_intensity)
    if args.ir_flood_intensity is not None:
        config["ir_flood_intensity"] = float(args.ir_flood_intensity)
    if args.xlink_chunk_size is not None:
        config["xlink_chunk_size"] = int(args.xlink_chunk_size)
    if args.ir_update_interval_s is not None:
        config["ir_update_interval_s"] = float(args.ir_update_interval_s)
    if args.adaptive_ir:
        config["adaptive_ir"] = True
    if args.no_adaptive_ir:
        config["adaptive_ir"] = False
    if args.rgb_undistort:
        config["rgb_undistort"] = True
    if args.no_rgb_undistort:
        config["rgb_undistort"] = False

    web_flag_map = {
        "pose": args.pose,
        "trajectory": args.trajectory,
        "control": args.control,
        "track": args.track,
        "imu": args.imu,
        "pointcloud": args.pointcloud,
    }
    selected_web = [name for name, enabled in web_flag_map.items() if enabled]
    if args.web_all:
        selected_web = list(web_flag_map.keys())

    if selected_web:
        config.setdefault("web_publish", {})
        config["web_publish"]["enabled"] = True
        config["web_publish"]["include_buffers"] = selected_web

    enable_slam = config.get("enable_slam", False)
    control_mode = config.get("control_mode", "neural")
    mapping_cfg = config.get("mapping", {})
    pointcloud_only_mode = bool(mapping_cfg.get("pointcloud_only", False))

    #  1. Create shared buffers 
    buf_cfg = config.get("buffers", {})

    frame_buf = FrameBuffer(
        slots=buf_cfg.get("frame", {}).get("slots", 8),
        name="frame_buf",
        create=True,
    )
    imu_buf = IMUBuffer(
        slots=buf_cfg.get("imu", {}).get("slots", 32),
        name="imu_buf",
        create=True,
    )
    seg_buf = SegmentationBuffer(
        slots=buf_cfg.get("segmentation", {}).get("slots", 4),
        name="seg_buf",
        create=True,
    )
    pose_buf = PoseBuffer(
        slots=buf_cfg.get("pose", {}).get("slots", 8),
        name="pose_buf",
        create=True,
    )
    ctrl_buf = ControlBuffer(
        slots=buf_cfg.get("control", {}).get("slots", 8),
        name="ctrl_buf",
        create=True,
    )

    global _buffers
    _buffers = [frame_buf, imu_buf, seg_buf, pose_buf, ctrl_buf]

    # SLAM-specific buffers
    pc_buf = None
    boundary_buf = None
    if enable_slam:
        pc_buf = PointCloudBuffer(
            slots=buf_cfg.get("pointcloud", {}).get("slots", 4),
            name="pc_buf",
            create=True,
        )
        _buffers.append(pc_buf)
        if not pointcloud_only_mode:
            boundary_buf = TrackBoundaryBuffer(
                slots=buf_cfg.get("boundary", {}).get("slots", 2),
                name="boundary_buf",
                create=True,
            )
            _buffers.append(boundary_buf)

    logger.info("Shared buffers created")

    mapping_enabled = bool(mapping_cfg.get("enabled", True)) and not pointcloud_only_mode
    loaded_prebuilt_map = False

    if enable_slam and pointcloud_only_mode:
        logger.info(
            "Pointcloud-only mode enabled: map/boundary generation disabled; recording pointcloud stream only"
        )

    #  2. Create multiprocessing events for pipeline coordination 
    loop_event = mp.Event() if enable_slam else None
    mapping_done_event = mp.Event() if enable_slam else None
    localize_event = mp.Event() if enable_slam else None

    if enable_slam and boundary_buf is not None:
        map_load_path = mapping_cfg.get("load_path")
        if map_load_path:
            try:
                outer_line, inner_line = _load_track_map(map_load_path)
                if len(outer_line) == 0 or len(inner_line) == 0:
                    raise ValueError("Loaded map has empty boundaries")
                boundary_buf.write(outer_line, inner_line)
                mapping_done_event.set()
                if mapping_cfg.get("set_localize_on_load", True):
                    localize_event.set()
                mapping_enabled = False
                loaded_prebuilt_map = True
                logger.info(
                    "Loaded prebuilt map from %s (outer=%d, inner=%d)",
                    map_load_path,
                    len(outer_line),
                    len(inner_line),
                )
            except Exception as exc:
                logger.error("Failed to load prebuilt map %s: %s", map_load_path, exc)
    elif enable_slam and pointcloud_only_mode and mapping_cfg.get("load_path"):
        logger.info("Ignoring mapping.load_path because mapping.pointcloud_only=true")

    #  3. Resolve config values 
    seg_config_path = config.get(
        "segmentation_config_path", "pilot_vision/models/default/config.yaml"
    )
    ctrl_config_path = config.get(
        "control_config_path", "pilot_controller/models/default/config.yaml"
    )
    resolution = config.get("resolution", [640, 400])
    fps = config.get("fps", 30)
    device = config.get("device", "cuda:0")
    use_oakd_imu = config.get("use_oakd_imu", False)
    i2c_address = config.get("i2c_address")
    i2c_bus = config.get("i2c_bus")
    slam_cfg = config.get("slam", {})
    map_plot_cfg = config.get("map_plot", {})
    rgbd_vis_cfg = config.get("rgbd_visualization", {})
    pid_cfg = config.get("pid", {})
    startup_gate_cfg = config.get("startup_gate", {}) or {}
    startup_gate_enabled = bool(startup_gate_cfg.get("enabled", True))
    startup_gate_timeout_s = float(startup_gate_cfg.get("timeout_s", 25.0))
    startup_gate_event = mp.Event() if startup_gate_enabled else None

    if isinstance(i2c_address, str):
        i2c_address = int(i2c_address, 0)

    effective_control_mode = control_mode
    if control_mode == "pid" and enable_slam and not mapping_enabled and not loaded_prebuilt_map:
        logger.warning(
            "PID mode requested with mapping disabled and no prebuilt map. "
            "Falling back to neural control mode."
        )
        effective_control_mode = "neural"

    #  4. Define workers   
      
    use_webcam = config.get("use_webcam", False)
    use_synthetic = config.get("synthetic", False)
    global _workers

    # When use_oakd_imu is True, integrate IMU into the camera pipeline
    # to avoid OAK-D device conflicts (only one process can own the device).
    camera_kwargs = dict(
        name="camera",
        output_buffer=frame_buf,
        fps=fps,
        resolution=resolution,
        stereo_resolution=config.get("stereo_resolution", [1280, 800]),
        depth_preset=config.get("depth_preset", "FAST_ACCURACY"),
        sync_threshold_ms=config.get("sync_threshold_ms", 25.0),
        ir_dot_intensity=config.get("ir_dot_intensity", 0.8),
        ir_flood_intensity=config.get("ir_flood_intensity", 0.8),
        adaptive_ir=config.get("adaptive_ir", True),
        ir_update_interval_s=config.get("ir_update_interval_s", 0.5),
        rgb_undistort=config.get("rgb_undistort", True),
        xlink_chunk_size=config.get("xlink_chunk_size", 0),
        synthetic=use_synthetic,
        rgbd_visualization=rgbd_vis_cfg,
    )
    if use_oakd_imu:
        camera_kwargs["imu_buffer"] = imu_buf
        logger.info("OAK-D IMU enabled — IMU data captured via CameraWorker")

    _workers = [
        CameraWorker(**camera_kwargs),
    ]

    # Only add a separate IMUWorker when NOT using OAK-D IMU
    if not use_oakd_imu:
        _workers.append(
            IMUWorker(
                name="imu",
                output_buffer=imu_buf,
                i2c_address=i2c_address,
                i2c_bus=i2c_bus,
                use_oakd_imu=False,
                synthetic=use_synthetic,
            ),
        )

    _workers.append(
        VisionWorker(
            name="vision",
            input_buffer=frame_buf,
            output_buffer=seg_buf,
            model_config=seg_config_path,
            device=device,
            use_tensorrt=config.get("use_tensorrt", False),
            debug=args.debug,
        ),
    )

    # Control worker — mode depends on config
    if effective_control_mode == "pid" and enable_slam:
        _workers.append(
            ControlWorker(
                name="control",
                mode="pid",
                output_buffer=ctrl_buf,
                pose_buffer=pose_buf,
                boundary_buffer=boundary_buf,
                mapping_done_event=mapping_done_event,
                pid_config=pid_cfg,
            )
        )
    else:
        _workers.append(
            ControlWorker(
                name="control",
                mode="neural",
                input_buffers={
                    "segments": seg_buf,
                    "imu": imu_buf,
                    "pose": pose_buf,
                },
                output_buffer=ctrl_buf,
                model_config=ctrl_config_path,
                device=device,
            )
        )

    _workers.append(
        ActuatorWorker(
            name="actuator",
            input_buffer=ctrl_buf,
            dummy_mode=bool(config.get("actuator_dummy", False)),
            i2c_address=i2c_address,
            i2c_bus=i2c_bus,
        )
    )

    # SLAM + Mapping workers
    if enable_slam:
        slam_use_imu = bool(use_oakd_imu and not use_synthetic)
        if use_oakd_imu and use_synthetic:
            logger.info(
                "Synthetic mode active: disabling SLAM IMU tracking (RGBD-only synthetic run)"
            )
        elif use_oakd_imu:
            logger.info(
                "Real OAK-D + IMU feed detected: forcing SLAM IMU_RGBD tracking"
            )
        _workers.append(
            SLAMWorker(
                name="slam",
                input_buffer=frame_buf,
                output_buffer=pose_buf,
                pointcloud_buffer=pc_buf,
                imu_buffer=imu_buf if slam_use_imu else None,
                use_imu=slam_use_imu,
                vocab_path=slam_cfg.get(
                    "vocabulary", "ORB_SLAM3/Vocabulary/ORBvoc.txt"
                ),
                settings_path=slam_cfg.get(
                    "settings", "oak_d_orbslam_settings.yaml"
                ),
                loop_event=loop_event,
                localize_event=localize_event,
            )
        )
        if mapping_enabled:
            _workers.append(
                MappingWorker(
                    name="mapping",
                    pose_buffer=pose_buf,
                    segment_buffer=seg_buf,
                    pointcloud_buffer=pc_buf,
                    boundary_buffer=boundary_buf,
                    loop_event=loop_event,
                    mapping_done_event=mapping_done_event,
                    extract_on_loop_only=mapping_cfg.get("extract_on_loop_only", False),
                    auto_extract_timeout_s=mapping_cfg.get("auto_extract_timeout_s", 20.0),
                    alpha_shape=mapping_cfg.get("alpha_shape", 25.0),
                    boundary_min_points=mapping_cfg.get("boundary_min_points", 50),
                    map_output_path=mapping_cfg.get("save_path"),
                )
            )

        if map_plot_cfg.get("enabled", False):
            _workers.append(
                MapPlotWorker(
                    name="map_plot",
                    pose_buffer=pose_buf,
                    pointcloud_buffer=pc_buf,
                    boundary_buffer=boundary_buf,
                    loop_event=loop_event,
                    mapping_done_event=mapping_done_event,
                    fps=map_plot_cfg.get("fps", 10),
                    save_dir=map_plot_cfg.get("save_dir", "/tmp/pilot_map"),
                    max_trail_laps=map_plot_cfg.get("max_trail_laps", 5),
                    save_video=map_plot_cfg.get("record_video", False),
                    video_filename=map_plot_cfg.get("video_filename", "map_recording.mp4"),
                )
            )

    # Optional: Per-buffer recorder workers
    if config.get("recording", False):
        rec_cfg = config.get("recording_config", {})
        rec_dir_template = config.get(
            "recording_path", "~/recordings/recording_{datetime}"
        )
        rec_dir = _resolve_recording_path(rec_dir_template)
        logger.info("Recording output directory: %s", rec_dir)
        rec_sample_k = rec_cfg.get("sample_k", 1)
        rec_top_n = rec_cfg.get("top_n", 0)

        # Map every active buffer to a dedicated BufferRecorderWorker.
        # Each recorder samples every k-th sequence number and caps at
        # top_n records (0 = unlimited).  Per-buffer overrides are read
        # from recording_config.<buffer_name>.{sample_k, top_n}.
        rec_buffers: Dict[str, Any] = {
            "frame": frame_buf,
            "imu": imu_buf,
            "segmentation": seg_buf,
            "pose": pose_buf,
            "control": ctrl_buf,
        }
        if enable_slam:
            if pc_buf is not None:
                rec_buffers["pointcloud"] = pc_buf
            if boundary_buf is not None:
                rec_buffers["boundary"] = boundary_buf

        for buf_name, buf_obj in rec_buffers.items():
            per_buf_cfg = rec_cfg.get(buf_name, {})
            k = per_buf_cfg.get("sample_k", rec_sample_k)
            n = per_buf_cfg.get("top_n", rec_top_n)
            _workers.append(
                BufferRecorderWorker(
                    name=f"rec_{buf_name}",
                    buffer=buf_obj,
                    buffer_name=buf_name,
                    output_dir=rec_dir,
                    sample_k=k,
                    top_n=n,
                )
            )

    # Optional: Web telemetry publisher
    web_cfg = config.get("web_publish", {})
    if web_cfg.get("enabled", False):
        _workers.append(
            WebPublishWorker(
                name="web_publish",
                pose_buffer=pose_buf,
                imu_buffer=imu_buf,
                pointcloud_buffer=pc_buf,
                boundary_buffer=boundary_buf,
                control_buffer=ctrl_buf,
                endpoint=web_cfg.get("endpoint", "http://localhost:8080/api/telemetry"),
                publish_interval=web_cfg.get("interval", 1.0),
                timeout=web_cfg.get("timeout", 2.0),
                vehicle_id=web_cfg.get("vehicle_id", "pilot_1"),
                max_trajectory_pts=web_cfg.get("max_trajectory_pts", 500),
                max_pointcloud_pts=web_cfg.get("max_pointcloud_pts", 2000),
                include_buffers=web_cfg.get("include_buffers"),
            )
        )
        logger.info(
            "Web publish worker enabled → %s (include=%s)",
            web_cfg.get("endpoint"),
            web_cfg.get("include_buffers", "default"),
        )

    #  5. Register signal handlers ─
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if startup_gate_enabled and startup_gate_event is not None:
        gate_exempt_workers = {"camera", "imu", "slam"}
        for w in _workers:
            if w.name in gate_exempt_workers:
                continue
            w.config["startup_gate_event"] = startup_gate_event
            w.config["startup_gate_timeout_s"] = startup_gate_timeout_s
        logger.info(
            "Startup gate enabled (timeout=%.1fs): non-core workers wait for camera/IMU/SLAM readiness",
            startup_gate_timeout_s,
        )

    #  6. Spawn worker processes ─
    global _processes
    _restart_counts: Dict[str, int] = {w.name: 0 for w in _workers}
    for w in _workers:
        p = mp.Process(target=w.run, name=w.name, daemon=False)
        p.start()
        _processes.append(p)
        logger.info(f"Spawned worker: {w.name} (pid={p.pid})")

    logger.info(f"All {len(_workers)} workers running.  Press Ctrl+C to stop.")
    if enable_slam:
        if loaded_prebuilt_map:
            logger.info("SLAM pipeline enabled — using prebuilt map in localization mode")
        elif mapping_enabled:
            logger.info(
                "SLAM pipeline enabled — mapping until loop closure, "
                "then PID control"
            )
        elif pointcloud_only_mode:
            logger.info("SLAM pipeline enabled — pointcloud-only mode (no map/boundaries)")
        else:
            logger.info("SLAM pipeline enabled — mapping disabled by config")

    #  7. Monitor loop 
    try:
        _terminal_gate_done = loaded_prebuilt_map or not mapping_enabled
        startup_gate_last_log_ts = 0.0
        while not _shutting_down:
            if startup_gate_enabled and startup_gate_event is not None and not startup_gate_event.is_set():
                frame_ready = frame_buf.read_latest() is not None
                imu_ready = imu_buf.read_latest() is not None
                pose_ready = (not enable_slam) or (pose_buf.read_latest() is not None)
                slam_alive = True
                if enable_slam:
                    slam_alive = any(
                        w.name == "slam" and p.is_alive()
                        for w, p in zip(_workers, _processes)
                    )

                if frame_ready and imu_ready and pose_ready and slam_alive:
                    startup_gate_event.set()
                    logger.info(
                        "Startup gate opened: frame=%s imu=%s pose=%s slam_alive=%s",
                        frame_ready,
                        imu_ready,
                        pose_ready,
                        slam_alive,
                    )
                else:
                    now = time.time()
                    if now - startup_gate_last_log_ts >= 1.0:
                        logger.info(
                            "Startup gate waiting: frame=%s imu=%s pose=%s slam_alive=%s",
                            frame_ready,
                            imu_ready,
                            pose_ready,
                            slam_alive,
                        )
                        startup_gate_last_log_ts = now

            # Check if mapping is done → prompt for terminal confirmation
            if (
                enable_slam
                and mapping_done_event is not None
                and mapping_done_event.is_set()
                and not _terminal_gate_done
            ):
                logger.info("=" * 60)
                logger.info("Track mapping complete!")
                logger.info("=" * 60)
                try:
                    input("Press ENTER to start PID control …")
                except EOFError:
                    pass
                # Switch SLAM to localization-only mode
                if localize_event is not None:
                    localize_event.set()
                _terminal_gate_done = True
                logger.info("PID control activated — vehicle driving")

            for i, (w, p) in enumerate(zip(_workers, _processes)):
                if not p.is_alive() and not _shutting_down:
                    exit_code = p.exitcode
                    _restart_counts[w.name] = _restart_counts.get(w.name, 0) + 1

                    if _restart_counts[w.name] > MAX_RESTARTS:
                        if _restart_counts[w.name] == MAX_RESTARTS + 1:
                            logger.error(
                                f"Worker {w.name} exceeded {MAX_RESTARTS} "
                                f"restarts — disabling"
                            )
                        continue
                    logger.warning(
                        f"Worker {w.name} (pid={p.pid}) exited with code "
                        f"{exit_code} — restarting "
                        f"({_restart_counts[w.name]}/{MAX_RESTARTS}) …"
                    )
                    w._stop_event = mp.Event()
                    new_p = mp.Process(target=w.run, name=w.name, daemon=False)
                    new_p.start()
                    _processes[i] = new_p
                    logger.info(
                        f"Restarted worker: {w.name} (new pid={new_p.pid})"
                    )
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
