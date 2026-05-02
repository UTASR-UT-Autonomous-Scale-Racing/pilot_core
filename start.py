#!/usr/bin/env python3
"""
Entry point for the pilot_core pipeline. Reads config.yaml plus CLI flags,
creates the shared-memory ring buffers in this (parent) process, spawns one
multiprocessing.Process per worker, restarts workers that crash, and cleans
up on SIGINT / SIGTERM / SIGHUP.

Usage (the config is required, so the car never runs a profile by accident):
    python start.py --config config.yaml       # Jetson on the car
    python start.py --config config_cpu.yaml   # workstation + OAK-D (map-free depth_ftg)
    python start.py --config config.yaml --enable-slam   # SLAM + mapping + track control
    python start.py --config config.yaml --recording     # record the buffers
    python start.py --config config.yaml --debug         # verbose logging
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

from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    IMUBuffer,
    LidarBuffer,
    PointCloudBuffer,
    PoseBuffer,
    SegmentationBuffer,
    StateBuffer,
    TrackBoundaryBuffer,
)

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
    StateWorker,
    VisionWorker,
    ZMQPublishWorker,
    ZMQSensorWorker,
)
from workers.control_worker import TRACK_MODES

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pilot_core — autonomous racecar inference pipeline"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML configuration file (config.yaml on the car, config_cpu.yaml on a workstation)",
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
        help="Enable SLAM + mapping (needed by the track-following modes)",
    )
    parser.add_argument(
        "--oakd-imu",
        action="store_true",
        help="Use OAK-D on-board IMU instead of I2C Arduino IMU",
    )
    return parser.parse_args()


# module-level so the signal handler can reach them

_workers: List[BaseWorker] = []
_processes: List[mp.Process] = []
_buffers: List[Any] = []
_shutting_down = False
_armed_event = None
_keyboard = None
MAX_RESTARTS = 3
# a restart of any of these disarms the car: the command chain was interrupted
CRITICAL_WORKERS = {"actuator", "control", "state", "camera", "slam", "zmq_sensor"}


def _release_buffers() -> None:
    for buf in _buffers:
        try:
            buf.unlink()
        except Exception:
            pass


def _shutdown(signum=None, frame=None) -> None:
    """Stop all workers and unlink the shared memory."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    logging.info("Shutdown signal received — stopping workers …")

    # disarm first so the actuator goes neutral while everything else winds down
    if _armed_event is not None:
        _armed_event.clear()
    if _keyboard is not None:
        _keyboard.close()

    for w in _workers:
        w.stop()

    # actuator first, so its neutral write goes out before anything else is torn down
    order = sorted(_processes, key=lambda p: p.name != "actuator")
    for p in order:
        p.join(timeout=5)
        if p.is_alive():
            logging.warning(f"Worker {p.name} did not exit in time — terminating")
            p.terminate()
            p.join(timeout=2)

    _release_buffers()

    logging.info("All workers stopped. Exiting.")
    sys.exit(0)


class KeyboardControl:
    """Non-blocking single-key terminal input for the operator.

    On a TTY we switch to cbreak so keys act without Enter; close() puts the
    terminal back. With piped stdin each line counts as one key. EOF is
    ignored and the car just stays in whatever state it's in.
    """

    def __init__(self) -> None:
        self._fd = None
        self._old = None
        self._eof = False
        try:
            if sys.stdin.isatty():
                import termios
                import tty

                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
        except Exception:
            self._fd = None

    def poll(self) -> List[str]:
        import select

        keys: List[str] = []
        if self._eof:
            return keys
        while select.select([sys.stdin], [], [], 0)[0]:
            if self._fd is not None:
                ch = os.read(self._fd, 1).decode(errors="ignore")
                if ch:
                    keys.append(ch.lower())
                    continue
            else:
                line = sys.stdin.readline()
                if line:
                    keys.append(line.strip().lower()[:1] or " ")
                    continue
            self._eof = True
            break
        return keys

    def close(self) -> None:
        if self._fd is not None and self._old is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._fd = None


def main() -> None:
    args = parse_args()

    # spawn everywhere, so Event/Value/Process all share one context and the
    # semaphores/locks actually get shared with the children
    mp.set_start_method("spawn", force=True)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("start")

    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Every worker is its own process. If each one spins up a full BLAS/OpenMP
    # thread pool, small CPUs get badly oversubscribed. Children inherit this env.
    threads = str(config.get("threads_per_worker", 1))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, threads)

    # CLI flags win over the config
    if args.recording:
        config["recording"] = True
    if args.enable_slam:
        config["enable_slam"] = True
    if args.oakd_imu:
        config["use_oakd_imu"] = True

    enable_slam = config.get("enable_slam", False)
    use_saved_map = config.get("use_saved_map", False)
    saved_map_path = config.get("saved_map_path", "rl/boundaries")
    control_mode = config.get("control_mode", "pure_pursuit")
    use_zmq = config.get("use_zmq", False)
    use_zmq_pose = config.get("use_zmq_pose", False)

    # shared buffers
    buf_cfg = config.get("buffers", {})
    resolution = config.get("resolution", [640, 400])

    # frame payload = meta(16) + W*H*3 RGB uint8 + W*H float32 depth in metres
    width, height = resolution
    _frame_max_payload = 16 + width * height * 3 + width * height * 4
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
    # Seg buffer payload: meta(8) + W*H float32 mask
    _seg_max_payload = 8 + width * height * 4
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
    needs_lidar = use_zmq and enable_slam and use_zmq_pose
    if needs_lidar:
        lidar_buf = LidarBuffer(
            slots=buf_cfg.get("lidar", {}).get("slots", 16),
            name="lidar_buf",
            create=True,
        )
        _buffers.append(lidar_buf)

    # state estimate only makes sense if something is producing poses
    has_pose_source = enable_slam or use_zmq_pose
    state_buf = None
    if has_pose_source:
        state_buf = StateBuffer(
            slots=buf_cfg.get("state", {}).get("slots", 8),
            name="state_buf",
            create=True,
        )
        _buffers.append(state_buf)

    logger.info("Shared buffers created")

    loop_event = mp.Event() if enable_slam else None
    mapping_done_event = mp.Event() if (enable_slam or use_saved_map) else None
    localize_event = mp.Event() if enable_slam else None

    # saved map: push the boundaries into boundary_buf right away
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

    resolution = config.get("resolution", [640, 400])
    fps = config.get("fps", 30)
    vision_cfg = config.get("vision", {})
    use_oakd_imu = config.get("use_oakd_imu", False)
    i2c_address = config.get("i2c_address")
    i2c_bus = config.get("i2c_bus")
    slam_cfg = config.get("slam", {})
    mapping_cfg = config.get("mapping", {})

    if isinstance(i2c_address, str):
        i2c_address = int(i2c_address, 0)

    # workers

    # one drivetrain calibration, shared by everything that produces a speed
    speed_per_throttle = (
        config.get("track_control", {}).get("speed", {}).get("longitudinal", {}).get("speed_per_throttle", 3.0)
    )
    depth_ftg_cfg = dict(config.get("depth_ftg", {}))
    depth_ftg_cfg.setdefault("speed_per_throttle", speed_per_throttle)
    vae_cfg = dict(config.get("vae", {}))
    vae_cfg.setdefault("speed_per_throttle", speed_per_throttle)
    bc_cfg = dict(config.get("bc", {}))
    bc_cfg.setdefault("speed_per_throttle", speed_per_throttle)

    use_synthetic = (
        config.get("synthetic", False) or os.environ.get("PILOT_SYNTHETIC") == "1"
    )
    zmq_host = config.get("zmq_host", "localhost")
    zmq_sensor_port = config.get("zmq_sensor_port", 5555)
    zmq_control_port = config.get("zmq_control_port", 5556)
    zmq_telemetry_port = config.get("zmq_telemetry_port", 5557)

    global _workers
    _workers = []

    # sensors
    if use_zmq:
        # sim: a single ZMQ subscriber stands in for all the hardware sensors
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
        # Hardware. Only one process can open the OAK-D, so if we're using its
        # IMU the camera worker streams it. Otherwise a separate IMU worker
        # reads I2C (or synthetic data).
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

    # Drivable-track segmentation. Nothing consumes seg_buf yet (it is recorded
    # and published as telemetry), so it is off unless vision.enabled is set.
    if vision_cfg.get("enabled", False):
        _workers.append(
            VisionWorker(
                name="vision",
                input_buffer=frame_buf,
                output_buffer=seg_buf,
                mode=vision_cfg.get("model", "ground_plane"),
                onnx_model=vision_cfg.get("onnx_model", "track_seg.onnx"),
                threshold=vision_cfg.get("threshold"),
                camera_settings=slam_cfg.get("settings", "orbslam_oakd_rgbd.yaml"),
                ground_plane=vision_cfg.get("ground_plane", {}),
            )
        )

    # Control. The track modes need a pose source (SLAM or sim ground truth);
    # depth_ftg / vae / bc only need frame_buf.
    global _armed_event
    _armed_event = mp.Event()
    if control_mode in TRACK_MODES and has_pose_source:
        _workers.append(
            ControlWorker(
                name="control",
                mode=control_mode,
                output_buffer=ctrl_buf,
                armed_event=_armed_event,
                state_buffer=state_buf,
                boundary_buffer=boundary_buf,
                mapping_done_event=mapping_done_event,
                track_config=config.get("track_control", {}),
                vehicle_params=config.get("vehicle_params", "vehicle_params.yaml"),
                frame_buffer=frame_buf,
                camera_settings=slam_cfg.get("settings", "orbslam_oakd_rgbd.yaml"),
                depth_ftg_config=depth_ftg_cfg,
            )
        )
    elif control_mode in ("depth_ftg", "vae", "bc"):
        _workers.append(
            ControlWorker(
                name="control",
                mode=control_mode,
                output_buffer=ctrl_buf,
                armed_event=_armed_event,
                frame_buffer=frame_buf,
                camera_settings=slam_cfg.get("settings", "orbslam_oakd_rgbd.yaml"),
                depth_ftg_config=depth_ftg_cfg,
                vae_config=vae_cfg,
                bc_config=bc_cfg,
                vehicle_params=config.get("vehicle_params", "vehicle_params.yaml"),
            )
        )
    else:
        needs_pose = control_mode in TRACK_MODES
        hint = (
            " — it needs a pose source: run with --enable-slam (or use_zmq_pose in the simulator)"
            if needs_pose else
            " — valid modes: " + ", ".join((*TRACK_MODES, "depth_ftg", "vae", "bc"))
        )
        _release_buffers()
        raise SystemExit(f"control_mode {control_mode!r} cannot run with this configuration{hint}")

    if has_pose_source:
        _workers.append(
            StateWorker(
                name="state",
                pose_buffer=pose_buf,
                output_buffer=state_buf,
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
                use_pose_speed=use_zmq_pose,
                estimator=config.get("estimator", {}),
            )
        )

    # Safety supervisor. Unless it's set explicitly, the steady-state throttle
    # bound uses the same drivetrain gain as the speed controller.
    safety_cfg = dict(config.get("safety", {}))
    safety_cfg.setdefault("speed_per_throttle", speed_per_throttle)

    # actuator: I2C to the hardware and/or ZMQ control out to the sim.
    # Stays neutral until the operator arms from the terminal
    _workers.append(
        ActuatorWorker(
            name="actuator",
            input_buffer=ctrl_buf,
            i2c_address=i2c_address,
            i2c_bus=i2c_bus,
            zmq_enabled=use_zmq,
            zmq_port=zmq_control_port,
            armed_event=_armed_event,
            state_buffer=state_buf,
            safety=safety_cfg,
            pwm=config.get("pwm", {}),
        )
    )

    # SLAM + mapping
    if enable_slam and use_zmq_pose:
        # ground-truth pose mode: 2D LIDAR scan -> world-frame 3D points
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
                    "settings", "orbslam_oakd_rgbd.yaml"
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
                boundary_min_points=mapping_cfg.get("boundary_min_points", 50),
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
                boundary_save_path=mapping_cfg.get("boundary_save_path"),
            )
        )

    if config.get("recording", False):
        rec_cfg = config.get("recording_config", {})
        rec_dir = config.get("recording_path", "/tmp/pilot_recordings")
        rec_sample_k = rec_cfg.get("sample_k", 1)
        rec_top_n = rec_cfg.get("top_n", 0)

        # One recorder per active buffer. Each keeps every k-th sequence
        # number, up to top_n records (0 = no limit). Override per buffer with
        # recording_config.<buffer_name>.{sample_k, top_n}
        rec_buffers: Dict[str, Any] = {
            "frame": frame_buf,
            "imu": imu_buf,
            "segmentation": seg_buf,
            "pose": pose_buf,
            "control": ctrl_buf,
        }
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

    # ZMQ telemetry for the sim visualiser
    zmq_publish_cfg = config.get("zmq_publish", {})
    if use_zmq and zmq_publish_cfg.get("enabled", False):
        _workers.append(
            ZMQPublishWorker(
                name="zmq_publish",
                port=zmq_telemetry_port,
                pose_buffer=pose_buf if zmq_publish_cfg.get("pose", False) else None,
                seg_buffer=seg_buf if zmq_publish_cfg.get("segmentation", False) else None,
                boundary_buffer=boundary_buf if zmq_publish_cfg.get("boundary", False) else None,
                pointcloud_buffer=pc_buf if zmq_publish_cfg.get("pointcloud", False) else None,
                publish=zmq_publish_cfg,
            )
        )
        logger.info(f"ZMQ telemetry publisher enabled on port {zmq_telemetry_port}")

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _shutdown)

    # spawn
    _restart_counts: Dict[str, int] = {w.name: 0 for w in _workers}
    for w in _workers:
        p = mp.Process(target=w.run, name=w.name, daemon=False)
        p.start()
        _processes.append(p)
        logger.info(f"Spawned worker: {w.name} (pid={p.pid})")

    logger.info(f"All {len(_workers)} workers running.")

    # operator keys + health monitor
    global _keyboard
    _keyboard = KeyboardControl()
    logger.info("=" * 60)
    logger.info("Keys:  [a] ARM   [SPACE] DISARM (stop)   [l] loop closure   [q] quit")
    logger.info("Outputs are neutral until armed.")
    logger.info("=" * 60)

    try:
        _mapping_announced = False
        _next_health_check = 0.0
        while not _shutting_down:
            for key in _keyboard.poll():
                if key == "a":
                    _armed_event.set()
                    logger.warning("ARMED by operator")
                elif key == "l" and enable_slam and loop_event is not None:
                    if not loop_event.is_set():
                        loop_event.set()
                        logger.info("Manual loop closure — extracting boundaries …")
                elif key == "q":
                    logger.info("Quit requested")
                    _shutdown()
                else:
                    # anything else disarms (SPACE is the documented one)
                    if _armed_event.is_set():
                        logger.warning("DISARMED by operator")
                    _armed_event.clear()

            # mapping done, switch SLAM over to localization only
            if (
                enable_slam
                and mapping_done_event is not None
                and mapping_done_event.is_set()
                and not _mapping_announced
            ):
                if localize_event is not None:
                    localize_event.set()
                _mapping_announced = True
                logger.info("=" * 60)
                logger.info("Track mapping complete — press [a] to arm and drive")
                logger.info("=" * 60)

            if time.monotonic() < _next_health_check:
                time.sleep(0.05)
                continue
            _next_health_check = time.monotonic() + 1.0

            for i, (w, p) in enumerate(zip(_workers, _processes)):
                if p.is_alive() or _shutting_down or _restart_counts.get(w.name, 0) > MAX_RESTARTS:
                    continue
                exit_code = p.exitcode
                if w.name in CRITICAL_WORKERS and _armed_event.is_set():
                    _armed_event.clear()
                    logger.error(f"Worker {w.name} died — DISARMED. Press [a] to re-arm once it is back")
                if w.name == "slam" and mapping_done_event is not None and mapping_done_event.is_set():
                    # a new SLAM process starts an empty map; localization is gone
                    _restart_counts[w.name] = MAX_RESTARTS + 1
                    logger.error("SLAM died after mapping — the map is lost. Restart the pipeline to re-map")
                    continue
                if w.name == "mapping" and mapping_done_event is not None and mapping_done_event.is_set():
                    _restart_counts[w.name] = MAX_RESTARTS + 1
                    continue
                _restart_counts[w.name] = _restart_counts.get(w.name, 0) + 1
                if _restart_counts[w.name] > MAX_RESTARTS:
                    logger.error(f"Worker {w.name} exceeded {MAX_RESTARTS} restarts — disabling")
                    if w.name == "actuator":
                        logger.error("No actuator output — shutting down")
                        _shutdown()
                    continue
                logger.warning(
                    f"Worker {w.name} (pid={p.pid}) exited with code {exit_code} — restarting "
                    f"({_restart_counts[w.name]}/{MAX_RESTARTS}) …"
                )
                w._stop_event = mp.Event()
                new_p = mp.Process(target=w.run, name=w.name, daemon=False)
                new_p.start()
                _processes[i] = new_p
                logger.info(f"Restarted worker: {w.name} (new pid={new_p.pid})")
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
