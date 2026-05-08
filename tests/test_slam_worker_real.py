"""SLAMWorker with the real ORB-SLAM3 wrapper on one synthetic lap: poses carry
the capture timestamps and come out in the level (floor-aligned) frame, and
map points / keyframe positions are published.

Skipped unless orbslam3_python imports (see docs/slam.md)."""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("orbslam3_python")

from shared_buffer import FrameBuffer, PointCloudBuffer, PoseBuffer
from synthetic.closed_loop_generator import ClosedLoopFrameGenerator
from workers.slam_worker import SLAMWorker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_slam_worker_level_poses_on_synthetic_lap():
    gen = ClosedLoopFrameGenerator(n_frames=200, straight_len=2.0, turn_radius=1.0, seed=3)
    frames = gen.generate_all(n_loops=1)
    uid = uuid.uuid4().hex[:8]
    fb = FrameBuffer(slots=4, name=f"swr_f_{uid}", create=True)
    pb = PoseBuffer(slots=256, name=f"swr_p_{uid}", create=True)
    pcb = PointCloudBuffer(slots=2, name=f"swr_pc_{uid}", create=True)
    kfb = PointCloudBuffer(slots=2, name=f"swr_kf_{uid}", create=True, max_points=5000)
    w = SLAMWorker(
        name="slam", input_buffer=fb, output_buffer=pb, pointcloud_buffer=pcb,
        keyframe_buffer=kfb,
        vocab_path=os.path.join(ROOT, "ORB_SLAM3", "Vocabulary", "ORBvoc.txt"),
        settings_path=os.path.join(ROOT, "synthetic_slam_settings.yaml"),
        warmup_frames=3,
    )
    w._logger = MagicMock()
    try:
        w.setup()
        heights, n_pub = [], 0
        t0 = 1000.0
        for i, (rgb, depth, _, ts, _) in enumerate(frames):
            fb.write(rgb, depth, ts=t0 + ts)
            w.step()
            if pb.latest_seq() > n_pub:
                n_pub = pb.latest_seq()
                header, pose, _, _ = pb.read_latest()
                assert header.produced_ts == pytest.approx(t0 + ts)
                heights.append(pose[1, 3])
                # the camera looks down by its mounting pitch against the floor,
                # not against the (pitched) first camera frame
                assert pose[1, 2] == pytest.approx(np.sin(gen.pitch), abs=0.03)
        assert w._R_level is not None
        assert n_pub > 0.5 * len(frames)
        # the camera drives at a constant height: level Y is constant
        assert np.ptp(heights) < 0.05
        assert pcb.latest_seq() > 0 and kfb.latest_seq() > 0
        _, kf = kfb.read_latest()
        assert np.ptp(kf[:, 1]) < 0.05
    finally:
        w.teardown()
        for b in (fb, pb, pcb, kfb):
            b.unlink()
