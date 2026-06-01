"""start.build_pipeline for every shipped config: the right workers are
created with the buffers they need, and nothing is spawned."""

from __future__ import annotations

import os
import uuid

import pytest

import start
from tests.conftest import REPO_ROOT

CONFIGS = ["config.yaml", "config_cpu.yaml", "config_dev_local.yaml", "config_zmq.yaml", "config_autodrive.yaml"]


@pytest.fixture(autouse=True)
def _unique_buffer_names(monkeypatch):
    # every buffer class takes name=...; give each test run its own names
    suffix = uuid.uuid4().hex[:6]
    for cls_name in ("FrameBuffer", "IMUBuffer", "SegmentationBuffer", "PoseBuffer", "ControlBuffer",
                     "PointCloudBuffer", "TrackBoundaryBuffer", "LidarBuffer", "StateBuffer"):
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
    for name in ("config_cpu.yaml", "config_dev_local.yaml", "config_zmq.yaml", "config_autodrive.yaml"):
        config = start.load_config(os.path.join(REPO_ROOT, name))
        assert config.get("i2c_address") is None, name


def test_track_mode_without_pose_source_is_rejected():
    with pytest.raises(SystemExit):
        _build("config_cpu.yaml", control_mode="pure_pursuit", enable_slam=False)


@pytest.mark.parametrize("config_name,overrides", [
    ("config_zmq.yaml", {"use_zmq_pose": True}),
    ("config_autodrive.yaml", {}),
])
def test_sim_pose_without_a_map_is_rejected(config_name, overrides):
    # ground-truth pose, track mode, but nothing would ever fill boundary_buf
    with pytest.raises(SystemExit, match="track boundaries"):
        _build(config_name, enable_slam=False, use_saved_map=False, **overrides)


def test_autodrive_wiring():
    p = _build("config_autodrive.yaml", recording=True)
    try:
        by = {w.name: w for w in p.workers}
        assert "zmq_sensor" not in by and "zmq_publish" not in by
        sim, act, ctrl = by["autodrive"], by["actuator"], by["control"]
        # the supervised command is what goes to the sim, nothing over ZMQ
        assert act.config["output_buffer"] is sim.config["command_buffer"] is not None
        assert not act.config["zmq_enabled"]
        assert sim.config["reset_event"] is p.reset_event is not None
        assert sim.config["lap_counter"] is p.counters["lap"]
        assert sim.config["log_dir"] == p.recording_dir
        # LiDAR mapping on the ground-truth pose; the gap follower drives that lap
        assert sim.config["lidar_buffer"] is by["lidar_proj"].config["lidar_buffer"] is not None
        assert ctrl.config["lidar_buffer"] is sim.config["lidar_buffer"]
        assert by["mapping"].config["auto_loop_close"] is True
        assert "actuation" in by["recorder"].config["buffers"]
    finally:
        for b in p.buffers:
            b.unlink()


def test_removed_modes_are_rejected():
    for mode in ("pid", "ftg", "lidar_pid"):
        with pytest.raises(SystemExit):
            _build("config_dev_local.yaml", control_mode=mode)


def test_shipped_files_resolve_from_any_directory(tmp_path, monkeypatch):
    from control.depth_ftg import Intrinsics
    from control.types import VehicleParams, resolve_path

    monkeypatch.chdir(tmp_path)
    assert os.path.isfile(resolve_path("orbslam_oakd_rgbd.yaml"))
    Intrinsics.from_orbslam_settings("synthetic_slam_settings.yaml")
    VehicleParams.load()
    assert "control_mode" in start.load_config("config.yaml")
    (tmp_path / "config.yaml").write_text("control_mode: local\n")      # a local file wins
    assert start.load_config("config.yaml")["control_mode"] == "local"
