"""start.build_pipeline for every shipped config: the right workers are
created with the buffers they need, and nothing is spawned."""

from __future__ import annotations

import os
import uuid

import pytest

import start
from tests.conftest import REPO_ROOT

CONFIGS = ["config.yaml", "config_cpu.yaml", "config_dev_local.yaml", "config_zmq.yaml"]


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
    for name in ("config_cpu.yaml", "config_dev_local.yaml", "config_zmq.yaml"):
        config = start.load_config(os.path.join(REPO_ROOT, name))
        assert config.get("i2c_address") is None, name


def test_track_mode_without_pose_source_is_rejected():
    with pytest.raises(SystemExit):
        _build("config_cpu.yaml", control_mode="pure_pursuit", enable_slam=False)


def test_removed_modes_are_rejected():
    for mode in ("pid", "ftg", "lidar_pid"):
        with pytest.raises(SystemExit):
            _build("config_dev_local.yaml", control_mode=mode)
