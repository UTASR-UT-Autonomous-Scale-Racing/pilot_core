"""
Per-worker unit tests, stepped in-process over real shm buffers.

Camera capture tests need OAKD_AVAILABLE=1. The real-SLAM test also needs the
ORB-SLAM3 binding and vocabulary; everything else runs SLAM in stub mode.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    IMUBuffer,
    PointCloudBuffer,
    PoseBuffer,
    SegmentationBuffer,
    TrackBoundaryBuffer,
)
from workers.base_worker import BaseWorker


def _uid():
    return uuid.uuid4().hex[:8]


class TestBaseWorker:

    def test_smoke_init(self):
        w = BaseWorker(name="test_worker", foo=42, bar="baz")
        assert w.name == "test_worker"
        assert w.config["foo"] == 42
        assert w.config["bar"] == "baz"

    def test_stop_sets_event(self):
        w = BaseWorker(name="test_worker")
        assert not w._stop_event.is_set()
        w.stop()
        assert w._stop_event.is_set()

    def test_setup_not_implemented(self):
        w = BaseWorker(name="test_worker")
        with pytest.raises(NotImplementedError):
            w.setup()

    def test_step_not_implemented(self):
        w = BaseWorker(name="test_worker")
        with pytest.raises(NotImplementedError):
            w.step()

    def test_teardown_is_noop(self):
        w = BaseWorker(name="test_worker")
        w.teardown()


# CameraWorker (capture needs the OAK-D)

class TestCameraWorker:

    def test_smoke_init(self):
        from workers import CameraWorker
        name = f"cam_{_uid()}"
        fb = FrameBuffer(slots=4, name=name, create=True)
        try:
            w = CameraWorker(name="camera", output_buffer=fb, fps=30, resolution=[640, 400])
            assert w.name == "camera"
            assert w.config["fps"] == 30
        finally:
            fb.unlink()

    @pytest.mark.skipif(
        not os.environ.get("OAKD_AVAILABLE", ""),
        reason="OAK-D camera not flagged as available (set OAKD_AVAILABLE=1)"
    )
    def test_setup_and_capture(self):
        from workers import CameraWorker
        name = f"cam_{_uid()}"
        fb = FrameBuffer(slots=4, name=name, create=True)
        try:
            w = CameraWorker(name="camera", output_buffer=fb, fps=30, resolution=[640, 400])
            w._logger = MagicMock()
            w._logger.info = print
            w.setup()

            # up to ~5 s for the first frame
            result = None
            for _ in range(100):
                w.step()
                result = fb.read_latest()
                if result is not None:
                    break
                time.sleep(0.05)

            assert result is not None, "No frame captured from OAK-D within 5s"
            header, rgb, depth = result
            assert rgb.shape == (400, 640, 3)
            assert depth.shape == (400, 640)
            assert rgb.dtype == np.uint8
            assert depth.dtype == np.float32

            w.teardown()
        finally:
            fb.unlink()

    @pytest.mark.skipif(
        not os.environ.get("OAKD_AVAILABLE", ""),
        reason="OAK-D camera not flagged as available"
    )
    def test_multiple_frames_sequential(self):
        from workers import CameraWorker
        name = f"cam_{_uid()}"
        fb = FrameBuffer(slots=8, name=name, create=True)
        try:
            w = CameraWorker(name="camera", output_buffer=fb, fps=30, resolution=[640, 400])
            w._logger = MagicMock()
            w._logger.info = print
            w.setup()

            prev_seq = 0
            for _ in range(200):
                w.step()
                result = fb.read_latest()
                if result is not None:
                    header, _, _ = result
                    if header.seq_no > prev_seq:
                        prev_seq = header.seq_no
                if prev_seq >= 10:
                    break
                time.sleep(0.02)

            assert prev_seq >= 10, f"Only captured {prev_seq} frames"
            w.teardown()
        finally:
            fb.unlink()


class TestIMUWorker:

    def test_smoke_init(self):
        from workers import IMUWorker
        name = f"imu_{_uid()}"
        ib = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = IMUWorker(name="imu", output_buffer=ib)
            assert w.name == "imu"
        finally:
            ib.unlink()

    def test_init_with_config(self):
        from workers import IMUWorker
        name = f"imu_{_uid()}"
        ib = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = IMUWorker(
                name="imu",
                output_buffer=ib,
                i2c_address=0x08,
                i2c_bus=1,
                use_oakd_imu=False,
            )
            assert w.config["i2c_address"] == 0x08
            assert w.config["i2c_bus"] == 1
            assert w.config["use_oakd_imu"] is False
        finally:
            ib.unlink()


# VisionWorker, with a mocked model

# SLAMWorker in stub mode (no orbslam3_python needed)

class TestSLAMWorker:

    def test_smoke_init(self):
        from workers import SLAMWorker
        fn = f"slam_f_{_uid()}"
        pn = f"slam_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = SLAMWorker(
                name="slam",
                input_buffer=fb,
                output_buffer=pb,
                vocab_path="nonexistent.txt",
                settings_path="nonexistent.yaml",
                synthetic=True,          # the stub is only allowed in synthetic runs
            )
            assert w.name == "slam"
        finally:
            fb.unlink()
            pb.unlink()

    def test_setup_stub_mode(self):
        from workers import SLAMWorker
        fn = f"slam_f_{_uid()}"
        pn = f"slam_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = SLAMWorker(
                name="slam",
                input_buffer=fb,
                output_buffer=pb,
                vocab_path="nonexistent.txt",
                settings_path="nonexistent.yaml",
                synthetic=True,          # the stub is only allowed in synthetic runs
            )
            w._logger = MagicMock()
            # make `import orbslam3_python` fail
            with patch.dict("sys.modules", {"orbslam3_python": None}):
                w.setup()
            assert w._slam is None
        finally:
            fb.unlink()
            pb.unlink()

    def test_step_stub_writes_identity(self, rgb_frame, depth_frame):
        from workers import SLAMWorker
        fn = f"slam_f_{_uid()}"
        pn = f"slam_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = SLAMWorker(
                name="slam",
                input_buffer=fb,
                output_buffer=pb,
            )
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._slam = None
            w._last_seq = 0

            fb.write(rgb_frame, depth_frame)

            w.step()

            result = pb.read_latest()
            assert result is not None
            _, pose, _, _ = result
            np.testing.assert_allclose(pose, np.eye(4, dtype=np.float32), atol=1e-7)
        finally:
            fb.unlink()
            pb.unlink()

    def test_step_no_frame_doesnt_crash(self):
        from workers import SLAMWorker
        fn = f"slam_f_{_uid()}"
        pn = f"slam_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._slam = None
            w._last_seq = 0

            w.step()
            assert pb.read_latest() is None
        finally:
            fb.unlink()
            pb.unlink()

    def test_slam_with_events(self, rgb_frame, depth_frame):
        """loop_event / localize_event kwargs are accepted, and localize sets the flag."""
        from workers import SLAMWorker
        fn = f"slam_f_{_uid()}"
        pn = f"slam_p_{_uid()}"
        pcn = f"slam_pc_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        pcb = PointCloudBuffer(slots=2, name=pcn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        loc_ev = ctx.Event()
        try:
            w = SLAMWorker(
                name="slam",
                input_buffer=fb,
                output_buffer=pb,
                pointcloud_buffer=pcb,
                loop_event=loop_ev,
                localize_event=loc_ev,
            )
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._pc_buf = pcb
            w._loop_event = loop_ev
            w._localize_event = loc_ev
            w._slam = None
            w._last_seq = 0
            w._localization_activated = False

            loc_ev.set()
            fb.write(rgb_frame, depth_frame)
            w.step()

            # stub has nothing to switch but the flag still has to flip
            assert w._localization_activated
        finally:
            fb.unlink()
            pb.unlink()
            pcb.unlink()

    @pytest.mark.skipif(
        not os.environ.get("OAKD_AVAILABLE", ""),
        reason="OAK-D + ORB_SLAM3 not available"
    )
    def test_real_slam_tracks_live_frames(self):
        """Real ORB_SLAM3 on live OAK-D frames publishes a pose. Initialising
        needs enough texture with valid depth, so allow a few seconds."""
        pytest.importorskip("orbslam3_python")
        from workers import CameraWorker, SLAMWorker

        fb = FrameBuffer(slots=4, name=f"slam_f_{_uid()}", create=True)
        pb = PoseBuffer(slots=4, name=f"slam_p_{_uid()}", create=True)
        cam = w = None
        try:
            cam = CameraWorker(name="cam", output_buffer=fb, fps=30, resolution=[640, 400])
            cam._logger = MagicMock()
            cam.setup()
            w = SLAMWorker(
                name="slam",
                input_buffer=fb,
                output_buffer=pb,
                vocab_path="ORB_SLAM3/Vocabulary/ORBvoc.txt",
                settings_path="orbslam_oakd_rgbd.yaml",
                warmup_frames=15,
                level_world=False,       # the floor may not be in view on a desk
            )
            w._logger = MagicMock()
            w.setup()

            deadline = time.time() + 15.0
            result = None
            while time.time() < deadline and result is None:
                cam.step()
                w.step()
                result = pb.read_latest()
            assert result is not None, "no pose in 15 s: is the camera looking at a textured, lit scene?"
            header, pose, _, _ = result
            assert pose.shape == (4, 4)
            assert header.produced_ts <= time.monotonic()      # capture-time stamp
        finally:
            if cam is not None:
                cam.teardown()
            if w is not None:
                w.teardown()
            fb.unlink()
            pb.unlink()


class TestMappingWorker:

    def test_smoke_init(self):
        from workers import MappingWorker
        pn = f"map_p_{_uid()}"
        sn = f"map_s_{_uid()}"
        pcn = f"map_pc_{_uid()}"
        bn = f"map_b_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        try:
            w = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
            )
            assert w.name == "mapping"
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_setup(self):
        from workers import MappingWorker
        pn = f"map_p_{_uid()}"
        sn = f"map_s_{_uid()}"
        pcn = f"map_pc_{_uid()}"
        bn = f"map_b_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()
        try:
            w = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
                loop_event=loop_ev,
                mapping_done_event=done_ev,
                boundary_min_points=30,
            )
            w._logger = MagicMock()
            w.setup()
            assert w._min_points == 30
            # defaults to the SLAM camera frame: X/Z ground plane, Z forward
            assert w._bev_axes == (0, 2)
            assert w._forward_axis == 2
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_step_accumulates_data(self, pose_4x4, segmentation_mask, point_cloud):
        from workers import MappingWorker
        pn = f"map_p_{_uid()}"
        sn = f"map_s_{_uid()}"
        pcn = f"map_pc_{_uid()}"
        bn = f"map_b_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()
        try:
            w = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
                loop_event=loop_ev,
                mapping_done_event=done_ev,
            )
            w._logger = MagicMock()
            w.setup()

            pb.write(pose_4x4)
            sb.write(segmentation_mask)
            pcb.write(point_cloud)

            w.step()

            assert len(w._bev_trajectory) >= 1
            assert len(w._bev_wall_pts) >= 1
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_boundary_extraction_on_loop_event(self):
        from workers import MappingWorker
        pn = f"map_p_{_uid()}"
        sn = f"map_s_{_uid()}"
        pcn = f"map_pc_{_uid()}"
        bn = f"map_b_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()
        try:
            w = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
                loop_event=loop_ev,
                mapping_done_event=done_ev,
                boundary_min_points=10,
            )
            w._logger = MagicMock()
            w.setup()

            # 200 wall points on two ovals
            theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            outer_3d = np.column_stack([
                8 * np.cos(theta),
                np.zeros(100),
                5 * np.sin(theta),
            ]).astype(np.float32)
            inner_3d = np.column_stack([
                5 * np.cos(theta),
                np.zeros(100),
                3 * np.sin(theta),
            ]).astype(np.float32)
            all_pts = np.vstack([outer_3d, inner_3d])
            pcb.write(all_pts)

            # One CCW lap of the centre line (X-Z) with camera forward (Z) along
            # the tangent. The worker only reads the newest pose, so we have to
            # step after every write.
            for phi in np.linspace(0, 2 * np.pi, 40, endpoint=False):
                T = np.eye(4, dtype=np.float32)
                T[0, 3], T[2, 3] = 6.5 * np.cos(phi), 4.0 * np.sin(phi)
                tangent = np.array([-6.5 * np.sin(phi), 0.0, 4.0 * np.cos(phi)])
                T[:3, 2] = tangent / np.linalg.norm(tangent)
                pb.write(T)
                w.step()

            loop_ev.set()
            w.step()

            assert w._boundaries_extracted
            assert done_ev.is_set()

            result = bb.read_latest()
            assert result is not None
            _, outer_out, inner_out = result
            assert len(outer_out) > 0
            assert len(inner_out) > 0
            # CCW lap, so the inner wall is on the left
            r_outer = np.linalg.norm(outer_out / [8.0, 5.0], axis=1)
            r_inner = np.linalg.norm(inner_out / [8.0, 5.0], axis=1)
            assert np.median(r_outer) > np.median(r_inner)
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_step_empty_buffers_no_crash(self):
        from workers import MappingWorker
        pn = f"map_p_{_uid()}"
        sn = f"map_s_{_uid()}"
        pcn = f"map_pc_{_uid()}"
        bn = f"map_b_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        try:
            w = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
            )
            w._logger = MagicMock()
            w.setup()
            w.step()
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()


class TestActuatorWorker:

    def test_smoke_init(self):
        from workers import ActuatorWorker
        cn = f"act_{_uid()}"
        cb = ControlBuffer(slots=4, name=cn, create=True)
        try:
            w = ActuatorWorker(name="actuator", input_buffer=cb)
            assert w.name == "actuator"
        finally:
            cb.unlink()
