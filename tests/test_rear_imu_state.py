"""The rear IMU on the car: StateWorker on control.ImuEKF (axes, lever arm, buffers) and the
mounting / delay logic of scripts/check_rear_imu.py."""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

import workers.state_worker as sw_mod
from shared_buffer import PoseBuffer, RawIMUBuffer, StateBuffer
from workers.state_worker import StateWorker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import check_rear_imu as chk  # noqa: E402


def _cam_pose(x, y, yaw):
    """ORB-SLAM3-style camera-to-world (world Y down, camera Z forward): BEV (x, y) = (X, Z)."""
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0], T[:3, 1], T[:3, 2] = (s, 0, -c), (0, 1, 0), (c, 0, s)
    T[0, 3], T[2, 3] = x, y
    return T


def test_state_worker_runs_the_imu_ekf_on_the_rear_imu(monkeypatch):
    """Car on a circle (3 m/s, 1 rad/s); the IMU 0.25 m behind the camera and mounted with its
    x forward and its y down; poses are the camera's. The published state is the camera's
    point, with the gyro's yaw rate."""
    v, r, lever = 3.0, 1.0, np.array([-0.25, 0.0])
    R_cfg = chk.rotation_from((1, -1), (0, 1))          # car = R_cfg @ sensor
    clock = {"t": 0.0}
    monkeypatch.setattr(sw_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(sw_mod.time, "sleep", lambda s: None)
    u = uuid.uuid4().hex[:8]
    pb, ib, sb = PoseBuffer(slots=8, name=f"rs_p_{u}"), RawIMUBuffer(slots=64, name=f"rs_i_{u}"), StateBuffer(slots=8, name=f"rs_s_{u}")
    w = StateWorker(name="state", pose_buffer=pb, output_buffer=sb, rear_imu_buffer=ib,
                    rear_imu={"rotation": R_cfg.tolist(), "lever_m": lever.tolist()}, rate_hz=50)
    w._logger = MagicMock()
    w.setup()
    try:
        def imu_point(t):
            yaw = r * t
            return (v / r) * np.sin(yaw), (v / r) * (1 - np.cos(yaw)), yaw

        for k in range(400):                       # 4 s: IMU at 100 Hz, poses at 30 Hz, 30 ms late
            t = k * 0.01
            car_acc, car_gyro = np.array([0.0, v * r, 9.81]), np.array([0.0, 0.0, r])
            ib.write(np.r_[R_cfg.T @ car_acc, R_cfg.T @ car_gyro].astype(np.float32), ts=t)
            if k % 3 == 0 and t >= 0.03:
                tp = t - 0.03
                x, y, yaw = imu_point(tp)
                cx, cy = x - (np.cos(yaw) * lever[0] - np.sin(yaw) * lever[1]), y - (np.sin(yaw) * lever[0] + np.cos(yaw) * lever[1])
                pb.write(_cam_pose(cx, cy, yaw), ts=tp)
            clock["t"] = t
            w.step()
        _, st = sb.read_latest()
        x, y, yaw = imu_point(st.t)
        cx, cy = x - np.cos(yaw) * lever[0], y - np.sin(yaw) * lever[0]
        assert np.hypot(st.x - cx, st.y - cy) < 0.05
        assert st.yaw_rate == pytest.approx(r, abs=0.02)
        assert st.v == pytest.approx(v, abs=0.1)
    finally:
        for b in (pb, ib, sb):
            b.unlink()


def test_state_worker_rejects_a_non_rotation():
    u = uuid.uuid4().hex[:8]
    pb, ib, sb = PoseBuffer(slots=4, name=f"rn_p_{u}"), RawIMUBuffer(slots=4, name=f"rn_i_{u}"), StateBuffer(slots=4, name=f"rn_s_{u}")
    w = StateWorker(name="state", pose_buffer=pb, output_buffer=sb, rear_imu_buffer=ib,
                    rear_imu={"rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 0]]})
    w._logger = MagicMock()
    try:
        with pytest.raises(ValueError):
            w.setup()
    finally:
        for b in (pb, ib, sb):
            b.unlink()


def test_check_finds_the_mounting_from_a_turn_and_a_push():
    # sensor y points down (car up = -y), sensor x points backward (forward = -x)
    t = np.linspace(0, 5, 501)
    gyro = np.zeros((501, 3))
    gyro[:, 1] = -np.radians(18.0)                   # 90 deg left turn over 5 s, about -y
    up = chk.axis_from_turn(t, gyro)
    assert up[:2] == (1, -1) and up[2] == pytest.approx(90.0, abs=0.5)
    acc = np.tile([0.0, -9.81, 0.0], (400, 1))
    acc[100:130, 0] = -1.5                           # pushed forward: -x
    acc[200:230, 0] = +1.5                           # stopped
    fwd = chk.axis_from_push(acc, up[0])
    assert fwd == (0, -1)
    R = chk.rotation_from(up[:2], fwd)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R @ [-1, 0, 0], [1, 0, 0])      # sensor -x = car forward
    np.testing.assert_allclose(R @ [0, -1, 0], [0, 0, 1])      # sensor -y = car up
    block = yaml.safe_load(chk.config_block(R, 0.012))["rear_imu"]
    np.testing.assert_allclose(block["rotation"], R)
    assert block["enabled"] and block["delay_s"] == pytest.approx(0.012)


def test_check_measures_the_rear_imu_delay():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 0.005)
    yaw_rate = np.convolve(rng.normal(0, 1, len(t)), np.ones(40) / 40, mode="same")
    cam = np.column_stack([np.zeros_like(t), yaw_rate, np.zeros_like(t)])       # OAK-D axes differ
    t_rear = np.arange(0, 20, 0.01)
    rear = np.column_stack([np.zeros_like(t_rear), np.zeros_like(t_rear), np.interp(t_rear - 0.012, t, yaw_rate)])
    assert chk.time_offset(t, cam, t_rear, rear) == pytest.approx(0.012, abs=0.001)
