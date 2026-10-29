"""SLAMWorker's RGB-D-inertial mode: IMU samples handed to ORB-SLAM3 per frame, and poses
kept in the origin keyframe's frame across ORB-SLAM3's gravity re-rotations."""

from __future__ import annotations

import sys
import types
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared_buffer import FrameBuffer, PoseBuffer, RawIMUBuffer
from workers.slam_worker import SLAMWorker


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _rigid(axis: str, deg: float, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_euler(axis, deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


@pytest.fixture
def bufs():
    fb = FrameBuffer(slots=2, name=f"si_f_{_uid()}", create=True)
    pb = PoseBuffer(slots=4, name=f"si_p_{_uid()}", create=True)
    ib = RawIMUBuffer(slots=64, name=f"si_i_{_uid()}", create=True)
    yield fb, pb, ib
    for b in (fb, pb, ib):
        b.unlink()


def _inertial_worker(fb, pb, ib, Twc=np.eye(4, dtype=np.float32)):
    w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb, inertial=True,
                   raw_imu_buffer=ib, imu_wait_s=0.0, warmup_frames=0)
    w._logger = MagicMock()
    w._frame_buf, w._pose_buf, w._imu_buf = fb, pb, ib
    w._warmup_frames = 0
    w._slam = MagicMock()
    w._slam.track_rgbd.return_value = np.linalg.inv(Twc).astype(np.float32)
    w._slam.get_tracking_state.return_value = 2
    w._slam.map_changed.return_value = False
    w._slam.map_info.return_value = (0, 0, 10)
    w._slam.origin_pose.return_value = np.eye(4, dtype=np.float32)
    w._slam.imu_initialized.return_value = False
    return w


def _frame(fb, ts):
    fb.write(np.zeros((40, 64, 3), np.uint8), np.ones((40, 64), np.float32), ts=ts)


def test_raw_imu_buffer_roundtrip(bufs):
    _, _, ib = bufs
    ib.write(np.arange(6, dtype=np.float32), ts=5.5)
    header, sample = ib.read_latest()
    assert header.produced_ts == 5.5
    np.testing.assert_array_equal(sample, np.arange(6, dtype=np.float32))
    with pytest.raises(ValueError):
        ib.write(np.zeros(9, np.float32))


def test_imu_samples_are_split_at_frame_times(bufs):
    fb, pb, ib = bufs
    w = _inertial_worker(fb, pb, ib)
    for k, t in enumerate(np.arange(0.90, 1.21, 0.05)):
        ib.write(np.array([0, 0, 9.81, 0, 0, k], np.float32), ts=float(t))
    _frame(fb, 1.0)
    w.step()
    imu = w._slam.track_rgbd.call_args[0][3]
    assert imu.dtype == np.float64 and imu.shape[1] == 7
    np.testing.assert_allclose(imu[:, 0], [0.90, 0.95, 1.00], atol=1e-9)
    np.testing.assert_allclose(imu[:, 6], [0, 1, 2])          # gyro z made it through
    _frame(fb, 1.1)
    w.step()
    imu = w._slam.track_rgbd.call_args[0][3]
    np.testing.assert_allclose(imu[:, 0], [1.05, 1.10], atol=1e-9)   # none twice, none lost


def test_warmup_frames_consume_their_imu(bufs):
    fb, pb, ib = bufs
    w = _inertial_worker(fb, pb, ib)
    w._warmup_frames = 1
    for t in (0.95, 1.0, 1.05, 1.1):
        ib.write(np.zeros(6, np.float32), ts=t)
    _frame(fb, 1.0)
    w.step()                                     # warm-up: not tracked
    assert w._slam.track_rgbd.call_count == 0
    _frame(fb, 1.1)
    w.step()
    np.testing.assert_allclose(w._slam.track_rgbd.call_args[0][3][:, 0], [1.05, 1.1], atol=1e-9)


def test_poses_stay_put_when_the_map_is_rotated_to_gravity(bufs):
    fb, pb, ib = bufs
    Twc = _rigid("y", 30.0, [1.0, 0.0, 2.0])
    w = _inertial_worker(fb, pb, ib, Twc)
    _frame(fb, 1.0)
    w.step()
    _, pose, _, _ = pb.read_latest()
    np.testing.assert_allclose(pose[:3, 3], [1.0, 0.0, 2.0], atol=1e-5)

    # IMU init: ORB-SLAM3 rotates everything, origin keyframe and camera alike
    G = _rigid("x", 12.0, [0.0, 0.0, 0.0])
    w._slam.origin_pose.return_value = G
    w._slam.track_rgbd.return_value = np.linalg.inv(G @ Twc).astype(np.float32)
    w._slam.imu_initialized.return_value = True
    _frame(fb, 1.1)
    w.step()
    assert pb.latest_seq() == 1                  # the rotation frame is withheld
    _frame(fb, 1.2)
    w.step()
    assert pb.latest_seq() == 2
    _, pose, _, _ = pb.read_latest()
    np.testing.assert_allclose(pose, Twc, atol=1e-5)   # same pose as before the rotation


def test_setup_picks_imu_rgbd_and_needs_the_raw_imu_buffer(bufs, monkeypatch):
    fb, pb, ib = bufs
    fake = types.ModuleType("orbslam3_python")
    fake.RGBD, fake.IMU_RGBD = 2, 5
    fake.System = MagicMock()
    monkeypatch.setitem(sys.modules, "orbslam3_python", fake)

    w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb, inertial=True,
                   raw_imu_buffer=ib, level_world=False)
    w._logger = MagicMock()
    w.setup()
    assert fake.System.call_args[0][2] == 5

    w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb, inertial=True, level_world=False)
    w._logger = MagicMock()
    with pytest.raises(RuntimeError, match="raw_imu_buffer"):
        w.setup()
