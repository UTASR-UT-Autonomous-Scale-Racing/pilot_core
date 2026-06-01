#!/usr/bin/env python3
"""
pilot_core entry point: creates the shared buffers, spawns one process per worker,
restarts crashed workers and cleans up on SIGINT / SIGTERM / SIGHUP.

Usage: python start.py --config config.yaml [--enable-slam] [--recording] [--debug]
       python start.py --config config_autodrive.yaml      # AutoDRIVE Unity sim, direct link
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import signal
import sys
import time
from dataclasses import dataclass
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

from shared_buffer.segment_buffer import max_payload_for as seg_max_payload_for
from workers import (
    ActuatorWorker,
    AutoDRIVEWorker,
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
)
from workers.control_worker import TRACK_MODES

def load_config(path: str) -> Dict[str, Any]:
    from control.types import resolve_path

    with open(resolve_path(path), "r") as f:
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
        "--control-mode",
        help="override control_mode from the config (e.g. pure_pursuit, depth_ftg)",
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
CRITICAL_WORKERS = {"actuator", "control", "state", "camera", "slam", "autodrive"}


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
    """Non-blocking single-key operator input (cbreak on a TTY, one key per line
    when piped). EOF is ignored: the car stays in its current state."""

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


@dataclass
class Pipeline:
    workers: List[BaseWorker]
    buffers: List[Any]
    armed_event: Any
    loop_event: Any
    mapping_done_event: Any
    localize_event: Any
    enable_slam: bool
    reset_event: Any = None          # sim reset (AutoDRIVE)
    counters: Dict[str, Any] = None  # sim lap / collision / checkpoint mp.Values
    recording_dir: str | None = None


def build_pipeline(config: Dict[str, Any]) -> Pipeline:
    """Create the shared buffers, events and workers for a config. Nothing is
    spawned; the buffers are unlinked again if anything fails."""
    buffers: List[Any] = []
    try:
        return _build(config, buffers)
    except BaseException:
        for buf in buffers:
            try:
                buf.unlink()
            except Exception:
                pass
        raise


def _build(config: Dict[str, Any], buffers: List[Any]) -> Pipeline:
    logger = logging.getLogger("start")
    enable_slam = config.get("enable_slam", False)
    use_saved_map = config.get("use_saved_map", False)
    from control.types import resolve_path

    saved_map_path = resolve_path(config.get("saved_map_path", "rl/boundaries"))
    control_mode = config.get("control_mode", "pure_pursuit")
    # AutoDRIVE sim over its own Socket.IO link: sensors in, supervised commands out
    autodrive = config.get("sim") == "autodrive"
    removed = sorted(k for k in config if k.startswith(("use_zmq", "zmq_")))
    if removed:
        raise SystemExit(f"{', '.join(removed)}: the ZMQ bridge was replaced by the direct sim link "
                         "(sim: autodrive, config_autodrive.yaml)")
    sim_pose = autodrive and config.get("sim_pose", True)   # ground-truth pose
    control_mode_early = config.get("control_mode", "pure_pursuit")
    if (sim_pose and control_mode_early in TRACK_MODES and not enable_slam and not use_saved_map):
        # nothing would ever fill boundary_buf: the controller would wait (or crash) forever
        raise SystemExit(
            f"control_mode {control_mode_early!r} needs track boundaries: with the sim's ground-truth "
            "pose, map a lap (--enable-slam, LiDAR mapping) or set use_saved_map")

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
    seg_buf = SegmentationBuffer(
        slots=buf_cfg.get("segmentation", {}).get("slots", 4),
        name="seg_buf",
        create=True,
        max_payload=seg_max_payload_for(height, width),
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

    buffers.extend([frame_buf, imu_buf, seg_buf, pose_buf, ctrl_buf])

    pc_buf = None
    kf_buf = None
    boundary_buf = None
    lidar_buf = None
    if enable_slam or use_saved_map:
        if enable_slam:
            pc_buf = PointCloudBuffer(
                slots=buf_cfg.get("pointcloud", {}).get("slots", 4),
                name="pc_buf",
                create=True,
            )
            # ORB-SLAM3 keyframe positions (loop-corrected trajectory for mapping)
            kf_buf = PointCloudBuffer(slots=2, name="kf_buf", max_points=5000, create=True)
            buffers.append(kf_buf)
        boundary_buf = TrackBoundaryBuffer(
            slots=buf_cfg.get("boundary", {}).get("slots", 2),
            name="boundary_buf",
            create=True,
        )
        buffers.extend([b for b in [pc_buf, boundary_buf] if b is not None])
    # the AutoDRIVE link always fills it (recordings, the mapping drive)
    needs_lidar = autodrive
    if needs_lidar:
        lidar_buf = LidarBuffer(
            slots=buf_cfg.get("lidar", {}).get("slots", 16),
            name="lidar_buf",
            create=True,
        )
        buffers.append(lidar_buf)

    # state estimate only makes sense if something is producing poses
    has_pose_source = enable_slam or sim_pose
    state_buf = None
    if has_pose_source:
        state_buf = StateBuffer(
            slots=buf_cfg.get("state", {}).get("slots", 8),
            name="state_buf",
            create=True,
        )
        buffers.append(state_buf)

    logger.info("Shared buffers created")

    loop_event = mp.Event() if enable_slam else None
    mapping_done_event = mp.Event() if (enable_slam or use_saved_map) else None
    localize_event = mp.Event() if enable_slam else None
    map_reset_event = mp.Event() if enable_slam else None   # SLAM started a new map

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
    workers: List[BaseWorker] = []

    rec_dir = None
    if config.get("recording", False):
        rec_dir = os.path.join(config.get("recording_path", "/tmp/pilot_recordings"), time.strftime("%Y%m%d_%H%M%S"))

    # sim reset (key r / TrackEnv) and the sim's race counters
    reset_event = mp.Event() if autodrive else None
    counters = ({"lap": mp.Value("i", 0), "collision": mp.Value("i", 0), "checkpoint": mp.Value("i", 0)}
                if autodrive else {})
    # supervised command (ActuatorWorker output) that AutoDRIVEWorker sends to the sim
    act_buf = None
    if autodrive:
        act_buf = ControlBuffer(slots=buf_cfg.get("control", {}).get("slots", 8), name="act_buf", create=True)
        buffers.append(act_buf)

    # sensors
    if autodrive:
        sim_host = config.get("sim_host", "0.0.0.0")
        sim_port = config.get("sim_port", 4567)
        workers.append(
            AutoDRIVEWorker(
                name="autodrive",
                host=sim_host,
                port=sim_port,
                frame_buffer=frame_buf if config.get("sim_camera", True) else None,
                imu_buffer=imu_buf,
                pose_buffer=pose_buf if sim_pose else None,
                lidar_buffer=lidar_buf,
                depth_max_meters=config.get("depth_max_meters", 15.0),
                depth_raw_scale=config.get("depth_raw_scale"),
                command_buffer=act_buf,
                command_timeout=config.get("safety", {}).get("command_timeout", 0.150),
                reset_event=reset_event,
                lap_counter=counters["lap"],
                collision_counter=counters["collision"],
                checkpoint_counter=counters["checkpoint"],
                log_dir=rec_dir,
                sim_depth=config.get("sim_depth"),
            )
        )
        logger.info(f"AutoDRIVE sim link on {sim_host}:{sim_port}")
    else:
        # only one process can open the OAK-D, so the camera worker streams its IMU
        oakd_imu_via_camera = use_oakd_imu and not use_synthetic
        workers.append(
            CameraWorker(
                name="camera",
                output_buffer=frame_buf,
                fps=fps,
                resolution=resolution,
                synthetic=use_synthetic,
                imu_buffer=imu_buf if oakd_imu_via_camera else None,
                settings_path=slam_cfg.get("settings", "orbslam_oakd_rgbd.yaml"),
                synthetic_scene=config.get("synthetic_scene", "corridor"),
                synthetic_track=config.get("synthetic_track", {}),
                **{k: v for k, v in config.get("camera", {}).items()
                   if k in ("max_exposure_us", "extended_disparity", "frame_timeout_s")},
            )
        )
        if not oakd_imu_via_camera:
            workers.append(
                IMUWorker(
                    name="imu",
                    output_buffer=imu_buf,
                    i2c_address=i2c_address,
                    i2c_bus=i2c_bus,
                    synthetic=use_synthetic,
                    poll_hz=config.get("imu_poll_hz", 100.0),
                )
            )

    # segmentation: nothing consumes seg_buf yet, so off unless vision.enabled
    if vision_cfg.get("enabled", False):
        workers.append(
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

    # throttle bound defaults to the speed controller's drivetrain gain
    safety_cfg = dict(config.get("safety", {}))
    safety_cfg.setdefault("speed_per_throttle", speed_per_throttle)

    # control: track modes need a pose source; depth_ftg / vae / bc only frame_buf
    armed_event = mp.Event()
    if control_mode in TRACK_MODES and has_pose_source:
        workers.append(
            ControlWorker(
                name="control",
                mode=control_mode,
                output_buffer=ctrl_buf,
                armed_event=armed_event,
                state_buffer=state_buf,
                boundary_buffer=boundary_buf,
                mapping_done_event=mapping_done_event,
                track_config=config.get("track_control", {}),
                safety=safety_cfg,
                vehicle_params=config.get("vehicle_params", "vehicle_params.yaml"),
                frame_buffer=frame_buf,
                camera_settings=slam_cfg.get("settings", "orbslam_oakd_rgbd.yaml"),
                depth_ftg_config=depth_ftg_cfg,
                lidar_buffer=lidar_buf,
            )
        )
    elif control_mode in ("depth_ftg", "vae", "bc"):
        workers.append(
            ControlWorker(
                name="control",
                mode=control_mode,
                output_buffer=ctrl_buf,
                armed_event=armed_event,
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
            " — it needs a pose source: run with --enable-slam (or the sim's ground-truth pose)"
            if needs_pose else
            " — valid modes: " + ", ".join((*TRACK_MODES, "depth_ftg", "vae", "bc"))
        )
        raise SystemExit(f"control_mode {control_mode!r} cannot run with this configuration{hint}")

    if has_pose_source:
        workers.append(
            StateWorker(
                name="state",
                pose_buffer=pose_buf,
                output_buffer=state_buf,
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
                use_pose_speed=sim_pose,
                estimator=config.get("estimator", {}),
                rate_hz=config.get("control_rate_hz", 50.0),
            )
        )

    # actuator stays neutral until the operator arms from the terminal
    workers.append(
        ActuatorWorker(
            name="actuator",
            input_buffer=ctrl_buf,
            i2c_address=i2c_address,
            i2c_bus=i2c_bus,
            armed_event=armed_event,
            state_buffer=state_buf,
            safety=safety_cfg,
            pwm=config.get("pwm", {}),
            output_buffer=act_buf,
            rate_hz=config.get("control_rate_hz", 50.0),
        )
    )

    # SLAM + mapping
    if enable_slam and sim_pose:
        # ground-truth pose mode: 2D LIDAR scan -> world-frame 3D points
        lidar_cfg = config.get("lidar", {})
        workers.append(
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
        logger.info("Using the sim's ground-truth pose + LIDAR projection for the point cloud")
    elif enable_slam:
        workers.append(
            SLAMWorker(
                name="slam",
                input_buffer=frame_buf,
                output_buffer=pose_buf,
                pointcloud_buffer=pc_buf,
                keyframe_buffer=kf_buf,
                map_reset_event=map_reset_event,
                synthetic=use_synthetic,
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
        workers.append(
            MappingWorker(
                name="mapping",
                pose_buffer=pose_buf,
                pointcloud_buffer=pc_buf,
                keyframe_buffer=kf_buf,
                map_reset_event=map_reset_event,
                boundary_buffer=boundary_buf,
                loop_event=loop_event,
                mapping_done_event=mapping_done_event,
                boundary_min_points=mapping_cfg.get("boundary_min_points", 50),
                bev_axes=mapping_cfg.get("bev_axes", [0, 2]),
                forward_axis=mapping_cfg.get("forward_axis", 2),
                boundary_save_path=mapping_cfg.get("boundary_save_path"),
                # one measured camera height for the whole config
                camera_height=mapping_cfg.get("camera_height", depth_ftg_cfg.get("camera_height", 0.15)),
                **{k: v for k, v in mapping_cfg.items()
                   if k in ("wall_height_band", "max_track_half_width", "boundary_bin_size",
                            "boundary_smooth_window", "lap_close_radius", "up_sign", "min_lap_length",
                            "auto_loop_close", "auto_loop_min_length", "auto_loop_overrun",
                            "jump_threshold", "min_track_half_width")},
            )
        )

    if config.get("recording", False):
        # recording_config.<buffer>.{sample_k, top_n, on_full, keep_files} override per buffer
        rec_cfg = config.get("recording_config", {})
        candidates = {
            "frame": frame_buf, "imu": imu_buf, "segmentation": seg_buf if vision_cfg.get("enabled", False) else None,
            "pose": pose_buf, "control": ctrl_buf, "actuation": act_buf, "pointcloud": pc_buf, "keyframes": kf_buf,
            "boundary": boundary_buf,
            "state": state_buf, "lidar": lidar_buf,
        }
        per_buffer_keys = ("sample_k", "top_n", "on_full", "keep_files")
        rec_buffers = {
            name: {"buffer": buf, **{k: v for k, v in (rec_cfg.get(name) or {}).items() if k in per_buffer_keys}}
            for name, buf in candidates.items() if buf is not None
        }
        workers.append(
            BufferRecorderWorker(
                name="recorder",
                buffers=rec_buffers,
                output_dir=rec_dir,
                sample_k=rec_cfg.get("sample_k", 1),
                top_n=rec_cfg.get("top_n", 0),
                on_full=rec_cfg.get("on_full", "rotate"),
                keep_files=rec_cfg.get("keep_files", 2),
                flush_interval=rec_cfg.get("flush_interval", 1.0),
            )
        )
        logger.info(f"Recording {', '.join(rec_buffers)} to {rec_dir}")

    return Pipeline(workers, buffers, armed_event, loop_event, mapping_done_event, localize_event, enable_slam,
                    reset_event=reset_event, counters=counters, recording_dir=rec_dir)


def main() -> None:
    args = parse_args()

    # spawn everywhere so Event/Value/Process share one context with the children
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

    # cap BLAS/OpenMP threads per worker process (inherited by the children)
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
    if args.control_mode:
        config["control_mode"] = args.control_mode

    global _workers, _buffers, _armed_event
    pipeline = build_pipeline(config)
    _workers, _buffers, _armed_event = pipeline.workers, pipeline.buffers, pipeline.armed_event
    enable_slam = pipeline.enable_slam
    loop_event = pipeline.loop_event
    mapping_done_event = pipeline.mapping_done_event
    localize_event = pipeline.localize_event
    reset_event = pipeline.reset_event

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
    logger.info("Keys:  [a] ARM   [SPACE] DISARM (stop)   [l] loop closure   [q] quit"
                + ("   [r] reset the sim" if reset_event is not None else ""))
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
                elif key == "r" and reset_event is not None:
                    reset_event.set()
                    logger.info("Sim reset requested")
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
