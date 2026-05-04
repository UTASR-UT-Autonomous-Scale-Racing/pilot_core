"""scripts/recording_to_dataset.py on a synthetic recording made by the real recorder."""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from control.pwm import PWMConfig
from scripts.recording_to_dataset import (
    build_bc_dataset,
    build_vision_dataset,
    main,
    nearest_indices,
    pwm_to_signed_steering,
)
from shared_buffer import ControlBuffer, FrameBuffer, IMUBuffer
from workers.buffer_recorder_worker import BufferRecorderWorker

H, W = 100, 160
FX, FY, CX, CY = 396.0 / 4, 395.8 / 4, 327.0 / 4, 189.0 / 4   # orbslam_oakd_rgbd.yaml scaled to 160x100
T0 = 5000.0
N_FRAMES = 6
PWM = PWMConfig()


def _floor_depth(cam_height=0.12):
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    ry = (v - CY) / FY
    with np.errstate(divide="ignore"):
        depth = np.where(ry > 1e-6, cam_height / ry, 0.0)
    return depth.astype(np.float32)


def _steer_pw(i):
    # frame i: full left, straight-ish, full right, ...
    return [PWM.steer_min, (PWM.steer_min + PWM.steer_max) / 2, PWM.steer_max][i % 3]


@pytest.fixture
def recording(tmp_path):
    u = uuid.uuid4().hex[:8]
    frame = FrameBuffer(slots=8, name=f"ds_f_{u}")
    imu = IMUBuffer(slots=64, name=f"ds_i_{u}")
    ctrl = ControlBuffer(slots=64, name=f"ds_c_{u}")
    out = tmp_path / "rec"
    try:
        w = BufferRecorderWorker(name="recorder", output_dir=str(out),
                                 buffers={"frame": frame, "imu": imu, "control": ctrl})
        w._logger = MagicMock()
        w.setup()
        depth = _floor_depth()
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[..., 0] = 200
        for i in range(N_FRAMES):
            t = T0 + i * 0.1
            frame.write(rgb, depth, ts=t)
            # labels at 100 Hz around each frame; the last frame gets none
            if i < N_FRAMES - 1:
                for k in range(-2, 3):
                    vec = np.zeros(9, np.float32)
                    vec[8] = _steer_pw(i)
                    imu.write(vec, ts=t + k * 0.01)
                    ctrl.write(i / 10, 0.5, ts=t + k * 0.01)
            w.step()
        # an invalid PWM reading (0 = no feedback) must be ignored
        imu.write(np.zeros(9, np.float32), ts=T0 + (N_FRAMES - 1) * 0.1)
        w.step()
        w.teardown()
        yield str(out)
    finally:
        for b in (frame, imu, ctrl):
            b.unlink()


def test_nearest_indices():
    ref = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(nearest_indices([0.4, 0.6, 2.04, 3.0], ref, 0.05), [-1, -1, 2, -1])
    np.testing.assert_array_equal(nearest_indices([0.4, 0.6, 1.9], ref, 0.5), [0, 1, 2])
    assert nearest_indices([1.0], np.array([]), 1.0)[0] == -1


def test_pwm_sign_convention():
    y = pwm_to_signed_steering([PWM.steer_min, PWM.steer_max, (PWM.steer_min + PWM.steer_max) / 2], PWM)
    np.testing.assert_allclose(y, [1.0, -1.0, 0.0])     # full-left pulse -> +1 (LEFT)


def test_bc_from_imu(recording, tmp_path):
    out = str(tmp_path / "bc.npz")
    stats = build_bc_dataset(recording, out, label_source="imu", tolerance=0.02)
    assert stats["frames"] == N_FRAMES and stats["paired"] == N_FRAMES - 1
    d = np.load(out)
    x, y = d["x"], d["y"]
    assert x.shape == (2 * (N_FRAMES - 1), 1, 40, 64) and x.dtype == np.float32
    assert y.shape == (2 * (N_FRAMES - 1), 1) and y.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    expected = [pwm_to_signed_steering(_steer_pw(i), PWM) for i in range(N_FRAMES - 1)]
    np.testing.assert_allclose(y[: N_FRAMES - 1, 0], expected, atol=1e-6)
    np.testing.assert_allclose(y[N_FRAMES - 1:, 0], -np.array(expected), atol=1e-6)   # mirrored half
    np.testing.assert_array_equal(x[N_FRAMES - 1], x[0][..., ::-1])


def test_bc_from_control_no_mirror(recording, tmp_path):
    out = str(tmp_path / "bc_ctrl.npz")
    stats = build_bc_dataset(recording, out, label_source="control", tolerance=0.02, mirror=False)
    d = np.load(out)
    assert stats["samples"] == N_FRAMES - 1 == len(d["x"])
    # ctrl steering i/10 in [0, 1] -> signed 1 - 2 * i/10
    np.testing.assert_allclose(d["y"][:, 0], [1 - 2 * i / 10 for i in range(N_FRAMES - 1)], atol=1e-6)


def test_bc_trainer_loads_it(recording, tmp_path):
    """The .npz is what train_bc_policy.load_or_collect reads."""
    import argparse
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    from train_bc_policy import load_or_collect

    out = str(tmp_path / "bc.npz")
    build_bc_dataset(recording, out)
    x, y = load_or_collect(argparse.Namespace(data=out), None, None)
    assert x.shape[1:] == (1, 40, 64) and y.shape[1:] == (1,)


def test_vision_with_bootstrap_masks(recording, tmp_path):
    from vision.data import find_pairs, load_pair

    out = str(tmp_path / "vis")
    stats = build_vision_dataset(recording, out, stride=2, bootstrap_masks=True,
                                 camera_settings="orbslam_oakd_rgbd.yaml")
    assert stats["images"] == N_FRAMES // 2
    assert stats["auto_masks"] == N_FRAMES // 2
    pairs = find_pairs(out)
    assert len(pairs) == N_FRAMES // 2
    rgb, mask = load_pair(*pairs[0], size=(H, W))
    assert rgb[..., 0].mean() == pytest.approx(200, abs=1)      # RGB order preserved
    assert mask[H - 10:].mean() > 0.8 and mask[: int(CY) - 5].sum() == 0
    with open(os.path.join(out, "auto_labels.json")) as f:
        meta = json.load(f)
    assert len(meta["auto_masks"]) == N_FRAMES // 2 and "review" in meta["note"]


def test_vision_images_only_and_cli(recording, tmp_path, capsys):
    out = str(tmp_path / "vis2")
    assert main([recording, "--vision", out, "--bc", str(tmp_path / "c.npz"), "--label-source", "imu",
                 "--tolerance", "0.02", "--config", "config.yaml"]) == 0
    assert len(os.listdir(os.path.join(out, "images"))) == N_FRAMES
    assert not os.path.exists(os.path.join(out, "masks"))
    assert "paired" in capsys.readouterr().out
