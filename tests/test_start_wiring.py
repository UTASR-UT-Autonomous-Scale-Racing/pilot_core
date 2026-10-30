"""start.build_pipeline for every shipped config: the right workers are
created with the buffers they need, and nothing is spawned."""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

import start
from tests.conftest import REPO_ROOT

CONFIGS = ["config.yaml", "config_cpu.yaml", "config_autodrive.yaml"]


@pytest.fixture(autouse=True)
def _unique_buffer_names(monkeypatch):
    # every buffer class takes name=...; give each test run its own names
    suffix = uuid.uuid4().hex[:6]
    for cls_name in ("FrameBuffer", "IMUBuffer", "SegmentationBuffer", "PoseBuffer", "ControlBuffer",
                     "PointCloudBuffer", "TrackBoundaryBuffer", "LidarBuffer", "StateBuffer", "RawIMUBuffer"):
        cls = getattr(start, cls_name)

        def make(*args, _cls=cls, **kwargs):
            kwargs["name"] = f"{kwargs['name']}_{suffix}"
            return _cls(*args, **kwargs)

        monkeypatch.setattr(start, cls_name, make)


def _build(config_name, **overrides):
    config = start.load_config(os.path.join(REPO_ROOT, config_name))
    config.update(overrides)
    return start.build_pipeline(config)


@pytest.mark.parametrize("config_name", CONFIGS)
@pytest.mark.parametrize("recording", [False, True])
def test_every_config_builds(config_name, recording):
    p = _build(config_name, recording=recording)
    try:
        names = [w.name for w in p.workers]
        assert "actuator" in names and "control" in names
        assert len(names) == len(set(names))
        assert ("recorder" in names) == recording
        actuator = next(w for w in p.workers if w.name == "actuator")
        control = next(w for w in p.workers if w.name == "control")
        assert actuator.config["armed_event"] is p.armed_event
        assert control.config["armed_event"] is p.armed_event
        if p.enable_slam:
            assert {"mapping", "state"} <= set(names)
    finally:
        for b in p.buffers:
            b.unlink()


def test_workstation_configs_never_write_i2c():
    for name in ("config_cpu.yaml", "config_autodrive.yaml"):
        config = start.load_config(os.path.join(REPO_ROOT, name))
        assert config.get("i2c_address") is None, name


def test_track_mode_without_pose_source_is_rejected():
    with pytest.raises(SystemExit):
        _build("config_cpu.yaml", control_mode="pure_pursuit", enable_slam=False)


def test_sim_pose_without_a_map_is_rejected():
    # ground-truth pose, track mode, but nothing would ever fill boundary_buf
    with pytest.raises(SystemExit, match="track boundaries"):
        _build("config_autodrive.yaml", enable_slam=False, use_saved_map=False)


def test_zmq_bridge_keys_are_rejected():
    with pytest.raises(SystemExit, match="direct sim link"):
        _build("config_cpu.yaml", use_zmq=True, zmq_sensor_port=5555)


def test_autodrive_wiring():
    p = _build("config_autodrive.yaml", recording=True)
    try:
        by = {w.name: w for w in p.workers}
        sim, act, ctrl = by["autodrive"], by["actuator"], by["control"]
        # the supervised command is what goes to the sim
        assert act.config["output_buffer"] is sim.config["command_buffer"] is not None
        assert sim.config["reset_event"] is p.reset_event is not None
        assert sim.config["lap_counter"] is p.counters["lap"]
        assert sim.config["log_dir"] == p.recording_dir
        # no LiDAR: the mapping lap's walls from the segmentation-edge scan on the ground-truth
        # pose; the gap follower drives that lap on the same kind of scan
        scan = by["seg_scan"].config["output_buffer"]
        assert sim.config["lidar_buffer"] is None and scan is not None
        assert by["seg_proj"].config["lidar_buffer"] is scan and ctrl.config["lidar_buffer"] is scan
        assert by["seg_proj"].config["range_max"] == 2.0 and by["seg_scan"].config["memory_s"] == 1.5
        assert by["mapping"].config["auto_loop_close"] is True
        assert by["mapping"].config["jump_threshold"] == 0.4      # sim respawns; off for SLAM
        assert "actuation" in by["recorder"].config["buffers"]
    finally:
        for b in p.buffers:
            b.unlink()
    # racing on a saved map: no mapping workers, no scan
    p = _build("config_autodrive.yaml", recording=True, enable_slam=False, use_saved_map=True)
    try:
        by = {w.name: w for w in p.workers}
        assert "mapping" not in by and "seg_proj" not in by and "lidar_proj" not in by
        assert "lidar" not in by["recorder"].config["buffers"]
    finally:
        for b in p.buffers:
            b.unlink()
    # asking for the LiDAR, or depth from it, is an error: pilot_core never uses one
    with pytest.raises(SystemExit):
        _build("config_autodrive.yaml", lidar={"enabled": True})
    with pytest.raises(SystemExit):
        _build("config_autodrive.yaml", sim_depth={"camera": "sim_camera/camera.yaml", "scale": 2, "depth": "lidar"})


def test_removed_modes_are_rejected():
    for mode in ("pid", "ftg", "lidar_pid"):
        with pytest.raises(SystemExit):
            _build("config_cpu.yaml", control_mode=mode)


def test_shipped_files_resolve_from_any_directory(tmp_path, monkeypatch):
    from control.depth_ftg import Intrinsics
    from control.types import VehicleParams, resolve_path

    monkeypatch.chdir(tmp_path)
    assert os.path.isfile(resolve_path("orbslam_oakd_rgbd.yaml"))
    Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
    VehicleParams.load()
    assert "control_mode" in start.load_config("config.yaml")
    (tmp_path / "config.yaml").write_text("control_mode: local\n")      # a local file wins
    assert start.load_config("config.yaml")["control_mode"] == "local"


def test_sim_depth_frames_fit_and_vision_uses_the_sim_camera():
    # sim_depth upscales the 192x108 sim camera (x2 here): the frame buffer must hold
    # that, and VisionWorker's intrinsics are the sim camera's, not slam.settings
    p = _build("config_autodrive.yaml", sim_depth={"camera": "sim_camera/camera.yaml", "scale": 2},
               vision={"enabled": True, "model": "ground_plane",
                       "camera_settings": "sim_camera/orbslam_settings.yaml"})
    try:
        by = {w.name: w for w in p.workers}
        by["autodrive"].config["frame_buffer"].write(np.zeros((216, 384, 3), np.uint8),
                                                      np.zeros((216, 384), np.float32))
        assert by["vision"].config["camera_settings"] == "sim_camera/orbslam_settings.yaml"
    finally:
        for b in p.buffers:
            b.unlink()


def test_rear_imu_feeds_the_imu_worker_and_the_state_worker():
    rear = {"enabled": True, "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "lever_m": [-0.25, 0.0]}
    p = _build("config.yaml", rear_imu=rear, use_oakd_imu=False, recording=True)
    try:
        imu = next(w for w in p.workers if w.name == "imu")
        buf = imu.config["raw_output_buffer"]
        assert buf is not None
        state = [w for w in p.workers if w.name == "state"]
        if state:
            assert state[0].config["rear_imu_buffer"] is buf
            assert state[0].config["rear_imu"]["lever_m"] == [-0.25, 0.0]
        rec = next(w for w in p.workers if w.name == "recorder")
        assert rec.config["buffers"]["rear_imu"]["buffer"] is buf
    finally:
        for b in p.buffers:
            b.unlink()


def test_rear_imu_needs_the_arduino_imu():
    with pytest.raises(SystemExit):
        _build("config.yaml", rear_imu={"enabled": True}, use_oakd_imu=True)


def test_sim_maps_from_the_segmentation_without_a_lidar():
    """The sim configs use no LiDAR (as the car): no lidar buffer; the mapping lap's walls come
    from the segmentation-edge scan (VisionWorker -> SegScanWorker -> projection)."""
    p = _build("config_autodrive.yaml")
    try:
        names = [w.name for w in p.workers]
        assert "vision" in names and "seg_scan" in names and "seg_proj" in names and "lidar_proj" not in names
        assert not any(getattr(b, "name", "").startswith("lidar_buf") for b in p.buffers)
        sim = next(w for w in p.workers if w.name == "autodrive")
        assert sim.config.get("lidar_buffer") is None and sim.config["sim_depth"]["depth"] == "none"
        proj = next(w for w in p.workers if w.name == "seg_proj")
        assert proj.config["scan_offset"] == [0.2733, 0.0, 0.10]
        ctrl = next(w for w in p.workers if w.name == "control")
        assert ctrl.config.get("lidar_buffer") is next(w for w in p.workers if w.name == "seg_scan").config["output_buffer"]
    finally:
        for b in p.buffers:
            b.unlink()



def test_sim_race_uses_no_lidar_no_ground_truth_speed_and_no_idle_segmentation():
    """The race config: no LiDAR buffer, the sim's own velocity is not fused into the state, and
    the segmentation (only needed for a mapping lap) is not started."""
    p = _build("config_autodrive_compete.yaml")
    try:
        by = {w.name: w for w in p.workers}
        assert by["autodrive"].config.get("lidar_buffer") is None
        assert by["state"].config["use_pose_speed"] is False
        assert "vision" not in by and "seg_scan" not in by
        assert p.mapping_done_event.is_set()          # racing on the saved map
    finally:
        for b in p.buffers:
            b.unlink()
