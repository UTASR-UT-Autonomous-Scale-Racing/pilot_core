"""
test_slam_pose_contract.py — SLAMWorker publishes camera-to-world poses
(Twc) and withholds poses while ORB-SLAM3 tracking is not OK.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
from scipy.spatial.transform import Rotation

from shared_buffer import FrameBuffer, PoseBuffer
from workers.slam_worker import SLAMWorker, invert_se3


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _rigid(yaw_deg: float, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rotation.from_euler("y", yaw_deg, degrees=True).as_matrix()
    T[:3, 3] = t
    return T


def test_invert_se3_matches_linalg():
    T = _rigid(37.0, [1.0, -0.2, 3.0])
    np.testing.assert_allclose(invert_se3(T), np.linalg.inv(T), atol=1e-6)


def _worker_with_fake_slam(Tcw: np.ndarray, state: int):
    fb = FrameBuffer(slots=2, name=f"sp_f_{_uid()}", create=True)
    pb = PoseBuffer(slots=2, name=f"sp_p_{_uid()}", create=True)
    w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
    w._logger = MagicMock()
    w._frame_buf, w._pose_buf = fb, pb
    w._slam = MagicMock()
    w._slam.track_rgbd.return_value = Tcw
    w._slam.get_tracking_state.return_value = state
    w._slam.map_changed.return_value = False
    fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))
    return w, fb, pb


def test_publishes_camera_position_in_world():
    # Camera at world (2, 0, 5), yawed 90° — Tcw translation is NOT the position
    Twc = _rigid(90.0, [2.0, 0.0, 5.0])
    w, fb, pb = _worker_with_fake_slam(np.linalg.inv(Twc).astype(np.float32), state=2)
    try:
        w.step()
        _, pose, _, _ = pb.read_latest()
        np.testing.assert_allclose(pose[:3, 3], [2.0, 0.0, 5.0], atol=1e-5)
    finally:
        fb.unlink(); pb.unlink()


def test_lost_tracking_publishes_nothing():
    w, fb, pb = _worker_with_fake_slam(np.eye(4, dtype=np.float32), state=3)
    try:
        w.step()
        assert pb.read_latest() is None
    finally:
        fb.unlink(); pb.unlink()


def test_warmup_frames_are_skipped():
    w, fb, pb = _worker_with_fake_slam(np.eye(4, dtype=np.float32), state=2)
    try:
        w._warmup_frames = 2
        for i in range(3):
            fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))
            w.step()
        assert w._slam.track_rgbd.call_count == 1   # frames 1-2 skipped, frame 3 tracked
    finally:
        fb.unlink(); pb.unlink()
