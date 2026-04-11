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
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from typing import Any, Dict, List

import numpy as np
import yaml

# --- Shared buffers ---
from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    IMUBuffer,
    LidarBuffer,
    PointCloudBuffer,
    PoseBuffer,
    SegmentationBuffer,
    TrackBoundaryBuffer,
)

# --- Workers ---
from workers import (
    ActuatorWorker,
    BaseWorker,
    BufferRecorderWorker,
    CameraWorker,
    ControlWorker,
    IMUWorker,
    LidarProjectionWorker,
    MappingWorker,
    SLAMWorker,
    VisionWorker,
    WebPublishWorker,
    ZMQPublishWorker,
    ZMQSensorWorker,
)

# --------------------------------------------------------------------------
# Config helpers
# --------------------------------------------------------------------------

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
    return parser.parse_args()


# --------------------------------------------------------------------------
# Global state for signal handling
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

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

    # Every worker is its own process; letting each one spin up a full
    # BLAS/OpenMP thread pool oversubscribes small CPUs badly.  Children
    # inherit this environment when spawned.
    threads = str(config.get("threads_per_worker", 1))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, threads)

    # Merge CLI overrides
    if args.recording:
        config["recording"] = True
    if args.enable_slam:
        config["enable_slam"] = True
    if args.oakd_imu:
        config["use_oakd_imu"] = True

    enable_slam = config.get("enable_slam", False)
    use_saved_map = config.get("use_saved_map", False)
    saved_map_path = config.get("saved_map_path", "rl/boundaries")
    control_mode = config.get("control_mode", "neural")
    use_zmq = config.get("use_zmq", False)
    use_zmq_pose = config.get("use_zmq_pose", False)

    # --- 1. Create shared buffers ---
    buf_cfg = config.get("buffers", {})
    resolution = config.get("resolution", [640, 400])

    # Size the frame buffer for the actual resolution: meta(16) + RGB uint8 + depth float32
    _h, _w = resolution[0], resolution[1]
    _frame_max_payload = 16 + _h * _w * 3 + _h * _w * 4
    frame_buf = FrameBuffer(
        slots=buf_cfg.get("frame", {}).get("slots", 8),
        name="frame_buf",
        create=True,
        max_payload=_frame_max_payload,
    )
    imu_buf = IMUBuffer(
        slots=buf_cfg.get("imu", {}).get("slots", 32),
        name="imu_buf",
        create=True,
    )
    # Size the seg buffer for the actual frame resolution (metadata=8 bytes + H*W*float32)
    _seg_h, _seg_w = resolution[0], resolution[1]
    _seg_max_payload = 8 + _seg_h * _seg_w * 4
    seg_buf = SegmentationBuffer(
        slots=buf_cfg.get("segmentation", {}).get("slots", 4),
        name="seg_buf",
        create=True,
        max_payload=_seg_max_payload,
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
    lidar_buf = None
    if enable_slam or use_saved_map:
        if enable_slam:
            pc_buf = PointCloudBuffer(
                slots=buf_cfg.get("pointcloud", {}).get("slots", 4),
                name="pc_buf",
                create=True,
            )
        boundary_buf = TrackBoundaryBuffer(
            slots=buf_cfg.get("boundary", {}).get("slots", 2),
            name="boundary_buf",
            create=True,
        )
        _buffers.extend([b for b in [pc_buf, boundary_buf] if b is not None])
    needs_lidar = use_zmq and (
        control_mode == "lidar_pid"
        or (enable_slam and config.get("use_zmq_pose", False))
    )
    if needs_lidar:
        lidar_buf = LidarBuffer(
            slots=buf_cfg.get("lidar", {}).get("slots", 16),
            name="lidar_buf",
            create=True,
        )
        _buffers.append(lidar_buf)

    logger.info("Shared buffers created")

    # --- 2. Create multiprocessing events for pipeline coordination ---
    loop_event = mp.Event() if enable_slam else None
    mapping_done_event = mp.Event() if (enable_slam or use_saved_map) else None
    localize_event = mp.Event() if enable_slam else None

    # Load pre-saved map boundaries and write into boundary_buf immediately
    if use_saved_map and boundary_buf is not None:
        outer_path = os.path.join(saved_map_path, "outer_line.npy")
        inner_path = os.path.join(saved_map_path, "inner_line.npy")
        if os.path.exists(outer_path) and os.path.exists(inner_path):
            outer_line = np.load(outer_path).astype(np.float32)
            inner_line = np.load(inner_path).astype(np.float32)
            boundary_buf.write(outer_line, inner_line)
            if mapping_done_event is not None:
                mapping_done_event.set()
            logger.info(f"Loaded saved map from {saved_map_path} ({len(outer_line)} outer, {len(inner_line)} inner pts)")
        else:
            logger.warning(f"use_saved_map=true but no .npy files found at {saved_map_path}")

    # --- 3. Resolve config values ---
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
    mapping_cfg = config.get("mapping", {})
    pid_cfg = config.get("pid", {})
    ftg_cfg = config.get("ftg", {})

    if isinstance(i2c_address, str):
        i2c_address = int(i2c_address, 0)

    # --- 4. Define workers ---

    use_synthetic = (
        config.get("synthetic", False) or os.environ.get("PILOT_SYNTHETIC") == "1"
    )
    zmq_host = config.get("zmq_host", "localhost")
    zmq_sensor_port = config.get("zmq_sensor_port", 5555)
    zmq_control_port = config.get("zmq_control_port", 5556)
    zmq_telemetry_port = config.get("zmq_telemetry_port", 5557)

    global _workers
    _workers = []

    # Sensor input workers
    if use_zmq:
        # Simulator mode: one ZMQ subscriber replaces the hardware sensors
        _workers.append(
            ZMQSensorWorker(
                name="zmq_sensor",
                host=zmq_host,
                port=zmq_sensor_port,
                frame_buffer=frame_buf,
                imu_buffer=imu_buf,
                pose_buffer=pose_buf if use_zmq_pose else None,
                lidar_buffer=lidar_buf,
                depth_max_meters=config.get("depth_max_meters", 10.0),
            )
        )
        logger.info(f"ZMQ sensor mode enabled (host={zmq_host}, port={zmq_sensor_port})")
    else:
        # Hardware mode.  The OAK-D can only be opened by one process, so its
        # IMU is streamed by the camera worker; otherwise a dedicated IMU
        # worker reads I2C / synthetic data.
        oakd_imu_via_camera = use_oakd_imu and not use_synthetic
        _workers.append(
            CameraWorker(
                name="camera",
                output_buffer=frame_buf,
                fps=fps,
                resolution=resolution,
                synthetic=use_synthetic,
                imu_buffer=imu_buf if oakd_imu_via_camera else None,
            )
        )
        if not oakd_imu_via_camera:
            _workers.append(
                IMUWorker(
                    name="imu",
                    output_buffer=imu_buf,
                    i2c_address=i2c_address,
                    i2c_bus=i2c_bus,
                    synthetic=use_synthetic,
                )
            )

    # Vision worker — skip when using LIDAR mapping (no segmentation needed)
    lidar_mapping_mode = use_zmq and use_zmq_pose and enable_slam
    if not lidar_mapping_mode:
        _workers.append(
            VisionWorker(
                name="vision",
                input_buffer=frame_buf,
                output_buffer=seg_buf,
                model_config=seg_config_path,
                device=device,
                use_tensorrt=config.get("use_tensorrt", False),
            )
        )

    # Control worker — mode depends on config
    # PID mode can use either SLAM or ZMQ ground truth pose
    has_pose_source = enable_slam or use_zmq_pose
    if control_mode == "lidar_pid":
        _workers.append(
            ControlWorker(
                name="control",
                mode="lidar_pid",
                output_buffer=ctrl_buf,
                lidar_buffer=lidar_buf,
                pid_config=pid_cfg,
            )
        )
    elif control_mode == "pid" and has_pose_source:
        _workers.append(
            ControlWorker(
                name="control",
                mode="pid",
                output_buffer=ctrl_buf,
                pose_buffer=pose_buf,
                boundary_buffer=boundary_buf,
                mapping_done_event=mapping_done_event,
                pid_config=pid_cfg,
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
            )
        )
    elif control_mode == "ftg" and has_pose_source:
        _workers.append(
            ControlWorker(
                name="control",
                mode="ftg",
                output_buffer=ctrl_buf,
                pose_buffer=pose_buf,
                boundary_buffer=boundary_buf,
                mapping_done_event=mapping_done_event,
                ftg_config=ftg_cfg,
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
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

    # Actuator worker (hardware I2C and/or ZMQ control publisher)
    _workers.append(
        ActuatorWorker(
            name="actuator",
            input_buffer=ctrl_buf,
            i2c_address=i2c_address,
            i2c_bus=i2c_bus,
            zmq_enabled=use_zmq,
            zmq_port=zmq_control_port,
        )
    )

    # SLAM + Mapping workers
    if enable_slam and use_zmq_pose:
        # Ground-truth pose mode: project 2D LIDAR scan to world-frame 3D points
        lidar_cfg = config.get("lidar", {})
        _workers.append(
            LidarProjectionWorker(
                name="lidar_proj",
                lidar_buffer=lidar_buf,
                pose_buffer=pose_buf,
                output_buffer=pc_buf,
                range_min=lidar_cfg.get("range_min", 0.1),
                range_max=lidar_cfg.get("range_max", 9.5),
                voxel_size=lidar_cfg.get("voxel_size", 0.10),
                move_threshold=lidar_cfg.get("move_threshold", 0.05),
            )
        )
        logger.info("Using ZMQ ground truth pose + LIDAR projection for point cloud")
    elif enable_slam:
        _workers.append(
            SLAMWorker(
                name="slam",
                input_buffer=frame_buf,
                output_buffer=pose_buf,
                pointcloud_buffer=pc_buf,
                vocab_path=slam_cfg.get(
                    "vocabulary", "ORB_SLAM3/Vocabulary/ORBvoc.txt"
                ),
                settings_path=slam_cfg.get(
                    "settings", "oak_d_orbslam_settings.yaml"
                ),
                loop_event=loop_event,
                localize_event=localize_event,
                warmup_frames=slam_cfg.get("warmup_frames", 15),
            )
        )

    if enable_slam and not use_saved_map:
        _workers.append(
            MappingWorker(
                name="mapping",
                pose_buffer=pose_buf,
                segment_buffer=seg_buf,
                pointcloud_buffer=pc_buf,
                boundary_buffer=boundary_buf,
                loop_event=loop_event,
                mapping_done_event=mapping_done_event,
                alpha_shape=mapping_cfg.get("alpha_shape", 25.0),
                boundary_min_points=mapping_cfg.get("boundary_min_points", 50),
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
                boundary_save_path=mapping_cfg.get("boundary_save_path"),
            )
        )

    # Optional: Per-buffer recorder workers
    if config.get("recording", False):
        rec_cfg = config.get("recording_config", {})
        rec_dir = config.get("recording_path", "/tmp/pilot_recordings")
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
                boundary_buffer=boundary_buf,
                control_buffer=ctrl_buf,
                endpoint=web_cfg.get("endpoint", "http://localhost:8080/api/telemetry"),
                publish_interval=web_cfg.get("interval", 1.0),
                timeout=web_cfg.get("timeout", 2.0),
                vehicle_id=web_cfg.get("vehicle_id", "pilot_1"),
                max_trajectory_pts=web_cfg.get("max_trajectory_pts", 500),
            )
        )
        logger.info("Web publish worker enabled → %s", web_cfg.get("endpoint"))

    # Optional: ZMQ telemetry publisher (for simulator visualization)
    zmq_publish_cfg = config.get("zmq_publish", {})
    if use_zmq and zmq_publish_cfg.get("enabled", False):
        _workers.append(
            ZMQPublishWorker(
                name="zmq_publish",
                port=zmq_telemetry_port,
                control_buffer=ctrl_buf,
                pose_buffer=pose_buf if zmq_publish_cfg.get("pose", False) else None,
                seg_buffer=seg_buf if zmq_publish_cfg.get("segmentation", False) else None,
                boundary_buffer=boundary_buf if zmq_publish_cfg.get("boundary", False) else None,
                pointcloud_buffer=pc_buf if zmq_publish_cfg.get("pointcloud", False) else None,
                publish=zmq_publish_cfg,
            )
        )
        logger.info(f"ZMQ telemetry publisher enabled on port {zmq_telemetry_port}")

    # --- 5. Register signal handlers ---
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # --- 6. Spawn worker processes ---
    _restart_counts: Dict[str, int] = {w.name: 0 for w in _workers}
    for w in _workers:
        p = mp.Process(target=w.run, name=w.name, daemon=False)
        p.start()
        _processes.append(p)
        logger.info(f"Spawned worker: {w.name} (pid={p.pid})")

    logger.info(f"All {len(_workers)} workers running.  Press Ctrl+C to stop.")
    if enable_slam:
        logger.info(
            "SLAM pipeline enabled — mapping until loop closure, "
            "then PID control"
        )

    # --- 7. Monitor loop ---
    # Inform user of manual loop-closure trigger when SLAM is enabled
    if enable_slam:
        logger.info("SLAM mapping active. Press 'L' + ENTER to manually signal loop closure if SLAM doesn't auto-detect it.")

    try:
        _terminal_gate_done = False
        while not _shutting_down:
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

            # Manual loop closure trigger: type 'L' + ENTER in the terminal
            if (
                enable_slam
                and loop_event is not None
                and not loop_event.is_set()
                and not _terminal_gate_done
            ):
                import select, sys as _sys
                if select.select([_sys.stdin], [], [], 0)[0]:
                    line = _sys.stdin.readline().strip().upper()
                    if line == 'L':
                        loop_event.set()
                        logger.info("Manual loop closure triggered — extracting boundaries …")

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
