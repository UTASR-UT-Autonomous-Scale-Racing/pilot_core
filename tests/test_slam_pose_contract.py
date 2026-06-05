"""SLAMWorker must publish Twc (camera-to-world), and nothing while ORB-SLAM3 tracking isn't OK."""

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
    w._slam.map_info.return_value = (0, 0, 10)
    fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32), ts=123.25)
    return w, fb, pb


def test_publishes_camera_position_in_world():
    # camera at world (2, 0, 5), yawed 90 deg. Tcw's translation is not the position
    Twc = _rigid(90.0, [2.0, 0.0, 5.0])
    w, fb, pb = _worker_with_fake_slam(np.linalg.inv(Twc).astype(np.float32), state=2)
    try:
        w.step()
        header, pose, _, _ = pb.read_latest()
        np.testing.assert_allclose(pose[:3, 3], [2.0, 0.0, 5.0], atol=1e-5)
        # the frame's capture time goes to ORB-SLAM3 and onto the pose
        assert w._slam.track_rgbd.call_args[0][2] == 123.25
        assert header.produced_ts == 123.25
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


def test_map_reset_withholds_in_localization_and_signals():
    import multiprocessing as mp
    w, fb, pb = _worker_with_fake_slam(np.eye(4, dtype=np.float32), state=2)
    try:
        w._map_reset_event = mp.Event()
        w.step()                                   # map (0, 0): reference map
        assert pb.latest_seq() == 1
        w._localization_activated = True
        w._slam.map_info.return_value = (1, 57, 3)  # new map after tracking loss
        fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))
        w.step()
        assert w._map_reset_event.is_set()
        assert pb.latest_seq() == 1                # withheld: not the mapped track's frame
        w._slam.map_info.return_value = (0, 0, 12)  # merged back into the mapped map
        fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))
        w.step()
        assert pb.latest_seq() == 2
    finally:
        fb.unlink(); pb.unlink()


def test_map_reset_while_mapping_adopts_new_map():
    import multiprocessing as mp
    w, fb, pb = _worker_with_fake_slam(np.eye(4, dtype=np.float32), state=2)
    try:
        w._map_reset_event = mp.Event()
        w.step()
        w._slam.map_info.return_value = (0, 40, 2)   # active map re-initialised
        fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))
        w.step()
        assert w._map_reset_event.is_set()
        assert pb.latest_seq() == 2
    finally:
        fb.unlink(); pb.unlink()


def test_missing_bindings_is_an_error(monkeypatch):
    import builtins
    import pytest

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "orbslam3_python":
            raise ImportError("libORB_SLAM3.so: cannot open shared object file")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    fb = FrameBuffer(slots=2, name=f"sp_f_{_uid()}", create=True)
    pb = PoseBuffer(slots=2, name=f"sp_p_{_uid()}", create=True)
    try:
        w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
        w._logger = MagicMock()
        with pytest.raises(RuntimeError, match="LD_LIBRARY_PATH"):
            w.setup()
    finally:
        fb.unlink(); pb.unlink()


def test_loop_is_signalled_only_after_global_ba_settles():
    import time

    w, fb, pb = _worker_with_fake_slam(np.eye(4, dtype=np.float32), state=2)
    w._loop_event = MagicMock()
    w._R_level = np.eye(4, dtype=np.float32)
    w._slam.get_all_map_points.return_value = np.zeros((5, 3), np.float32)
    w._pc_buf = MagicMock()

    def new_frame():
        fb.write(np.zeros((400, 640, 3), np.uint8), np.ones((400, 640), np.float32))
        w.step()

    try:
        w._slam.map_changed.return_value = True            # loop detected, GBA starts
        w._slam.is_running_gba.return_value = True
        new_frame()
        w._slam.map_changed.return_value = False
        new_frame()
        w._loop_event.set.assert_not_called()               # still optimising
        w._slam.is_running_gba.return_value = False
        new_frame()
        w._loop_event.set.assert_not_called()               # finished, but not settled yet
        w._last_map_change = time.monotonic() - w._LOOP_SETTLE_S - 0.1
        new_frame()
        w._loop_event.set.assert_called_once()
        assert w._pc_buf.write.called                       # the optimised map went out
    finally:
        fb.unlink(); pb.unlink()
