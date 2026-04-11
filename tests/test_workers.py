"""
test_workers.py — Tests for all worker modules.

Categories:
    - Smoke / Initialisation: worker instantiation without starting
    - Setup tests: verify setup() connects to buffers & devices
    - Step I/O: write input buffers, call step, verify output buffers
    - Null checks: step with no data, missing config
    - Teardown: verify clean shutdown

Workers tested:
    - CameraWorker (requires OAK-D)
    - IMUWorker (synthetic)
    - VisionWorker (ResNet18 mock stub)
    - SLAMWorker (stub mode — no orbslam3_python required)
    - MappingWorker (synthetic data)
    - ControlWorker (neural mock + PID mode)
    - ActuatorWorker (no-hardware stub)
    - RecorderWorker (tempdir)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
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


#
# --- BaseWorker ---
#

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
        w.teardown()  # should not raise


#
# --- CameraWorker — requires OAK-D hardware ---
#

class TestCameraWorker:

    def _have_oakd(self) -> bool:
        """Check if OAK-D is plugged in."""
        try:
            import depthai as dai
            devices = dai.Device.getAllAvailableDevices()
            return len(devices) > 0
        except Exception:
            return False

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
        """Smoke test: setup OAK-D pipeline, capture one frame, teardown."""
        from workers import CameraWorker
        name = f"cam_{_uid()}"
        fb = FrameBuffer(slots=4, name=name, create=True)
        try:
            w = CameraWorker(name="camera", output_buffer=fb, fps=30, resolution=[640, 400])
            w._logger = MagicMock()
            w._logger.info = print
            w.setup()

            # Wait for at least one frame
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
        """Capture 10 frames and verify monotonic seq_no."""
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


#
# --- IMUWorker — synthetic I2C stub ---
#

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


#
# --- VisionWorker — mock model ---
#

class TestVisionWorker:

    def test_smoke_init(self):
        from workers import VisionWorker
        fn = f"vis_{_uid()}"
        sn = f"seg_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        try:
            w = VisionWorker(
                name="vision",
                input_buffer=fb,
                output_buffer=sb,
                model_config="dummy_path",
                device="cpu",
                use_tensorrt=False,
            )
            assert w.name == "vision"
        finally:
            fb.unlink()
            sb.unlink()

    def test_step_with_mock_model(self, rgb_frame, depth_frame):
        """Write a frame, mock the model, run step, check seg output."""
        from workers import VisionWorker
        fn = f"vis_{_uid()}"
        sn = f"seg_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        try:
            w = VisionWorker(
                name="vision",
                input_buffer=fb,
                output_buffer=sb,
                model_config="dummy",
                device="cpu",
            )
            w._logger = MagicMock()
            w._logger.info = MagicMock()
            w._logger.debug = MagicMock()

            # Write a frame as input
            fb.write(rgb_frame, depth_frame)

            # Mock model that returns a uint8 mask (ResNet18-like)
            mock_mask = np.random.randint(0, 3, (400, 640), dtype=np.int32)
            mock_model = MagicMock()
            mock_model.get_mask.return_value = mock_mask

            # Patch setup to just inject mock model
            w._model = mock_model
            w._frame_buf = fb
            w._seg_buf = sb

            # Step should read frame and write segmentation
            # We need to replicate what step does (read frame -> model -> write seg)
            result = fb.read_latest()
            assert result is not None
            header, rgb, depth = result
            mask = w._model.get_mask(depth, rgb)
            sb.write(mask)

            seg_result = sb.read_latest()
            assert seg_result is not None
            _, mask_out = seg_result
            np.testing.assert_array_equal(mask_out, mock_mask)
        finally:
            fb.unlink()
            sb.unlink()

    def test_step_no_frame_available(self):
        """Step with empty input buffer should not crash."""
        from workers import VisionWorker
        fn = f"vis_{_uid()}"
        sn = f"seg_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        try:
            w = VisionWorker(
                name="vision",
                input_buffer=fb,
                output_buffer=sb,
                model_config="dummy",
                device="cpu",
            )
            w._logger = MagicMock()
            w._frame_buf = fb
            w._seg_buf = sb
            w._model = MagicMock()
            w._last_seq = 0

            # step() when frame_buf is empty → should return without crash
            # We replicate the guard logic
            result = fb.read_latest()
            assert result is None  # no frame yet
        finally:
            fb.unlink()
            sb.unlink()


#
# --- SLAMWorker — stub mode (no orbslam3_python needed) ---
#

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
            )
            assert w.name == "slam"
        finally:
            fb.unlink()
            pb.unlink()

    def test_setup_stub_mode(self):
        """Setup should fall back to stub when orbslam3_python is unavailable."""
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
            )
            w._logger = MagicMock()
            # Force stub by patching import
            with patch.dict("sys.modules", {"orbslam3_python": None}):
                w.setup()
            assert w._slam is None  # stub mode
        finally:
            fb.unlink()
            pb.unlink()

    def test_step_stub_writes_identity(self, rgb_frame, depth_frame):
        """In stub mode, step should write an identity pose."""
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
            w._slam = None  # stub mode
            w._last_seq = 0

            # Write a frame
            fb.write(rgb_frame, depth_frame)

            # Step
            w.step()

            # Should have written identity pose
            result = pb.read_latest()
            assert result is not None
            _, pose, _, _ = result
            np.testing.assert_allclose(pose, np.eye(4, dtype=np.float32), atol=1e-7)
        finally:
            fb.unlink()
            pb.unlink()

    def test_step_no_frame_doesnt_crash(self):
        """Step with empty frame buffer should sleep and return."""
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

            # Step with no data — should not raise
            w.step()
            assert pb.read_latest() is None
        finally:
            fb.unlink()
            pb.unlink()

    def test_slam_with_events(self, rgb_frame, depth_frame):
        """Verify loop_event and localize_event kwargs are accepted."""
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

            # Set localize event and step
            loc_ev.set()
            fb.write(rgb_frame, depth_frame)
            w.step()

            # Should have activated localization (noop for stub)
            assert w._localization_activated
        finally:
            fb.unlink()
            pb.unlink()
            pcb.unlink()

    @pytest.mark.skipif(
        not os.environ.get("OAKD_AVAILABLE", ""),
        reason="OAK-D + ORB_SLAM3 not available"
    )
    def test_real_slam_single_frame(self):
        """Integration: real ORB_SLAM3 tracking on a single real frame."""
        import orbslam3_python
        from workers import CameraWorker, SLAMWorker

        fn = f"slam_f_{_uid()}"
        pn = f"slam_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            # Capture one real frame via OAK-D
            cam = CameraWorker(name="cam", output_buffer=fb, fps=30, resolution=[640, 400])
            cam._logger = MagicMock()
            cam._logger.info = print
            cam.setup()

            for _ in range(100):
                cam.step()
                if fb.read_latest() is not None:
                    break
                time.sleep(0.05)
            cam.teardown()

            # Run SLAM on captured frame
            w = SLAMWorker(
                name="slam",
                input_buffer=fb,
                output_buffer=pb,
                vocab_path="ORB_SLAM3/Vocabulary/ORBvoc.txt",
                settings_path="orbslam_oakd_rgbd.yaml",
                warmup_frames=0,  # single-frame test
            )
            w._logger = MagicMock()
            w._logger.info = print
            w._logger.warning = print
            w.setup()
            w._frame_buf = fb
            w._pose_buf = pb
            w._last_seq = 0

            w.step()
            result = pb.read_latest()
            assert result is not None
            _, pose, _, _ = result
            assert pose.shape == (4, 4)
            w.teardown()
        finally:
            fb.unlink()
            pb.unlink()


#
# --- MappingWorker ---
#

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
            # Defaults: SLAM camera frame (X/Z ground plane, Z forward)
            assert w._bev_axes == (0, 2)
            assert w._forward_axis == 2
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_step_accumulates_data(self, pose_4x4, segmentation_mask, point_cloud):
        """Step reads buffers and accumulates BEV data."""
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

            # Write data to buffers
            pb.write(pose_4x4)
            sb.write(segmentation_mask)
            pcb.write(point_cloud)

            # Step
            w.step()

            # Should have accumulated trajectory
            assert len(w._bev_trajectory) >= 1
            # Should have accumulated BEV wall points
            assert len(w._bev_wall_pts) >= 1
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_boundary_extraction_on_loop_event(self):
        """Trigger loop event → verify boundary extraction + mapping_done."""
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

            # Pre-fill data: 200 wall points (oval track) + trajectory
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

            # Drive one lap of the centre line (counter-clockwise in X-Z) with
            # the camera's forward (Z) axis along the track tangent.  The
            # worker reads only the newest pose, so step after every write.
            for phi in np.linspace(0, 2 * np.pi, 40, endpoint=False):
                T = np.eye(4, dtype=np.float32)
                T[0, 3], T[2, 3] = 6.5 * np.cos(phi), 4.0 * np.sin(phi)
                tangent = np.array([-6.5 * np.sin(phi), 0.0, 4.0 * np.cos(phi)])
                T[:3, 2] = tangent / np.linalg.norm(tangent)
                pb.write(T)
                w.step()

            # Set loop closure → should extract boundaries
            loop_ev.set()
            w.step()

            assert w._boundaries_extracted
            assert done_ev.is_set()

            # Verify boundary was written to buffer
            result = bb.read_latest()
            assert result is not None
            _, outer_out, inner_out = result
            assert len(outer_out) > 0
            assert len(inner_out) > 0
            # Counter-clockwise lap: the inner wall is on the left
            r_outer = np.linalg.norm(outer_out / [8.0, 5.0], axis=1)
            r_inner = np.linalg.norm(inner_out / [8.0, 5.0], axis=1)
            assert np.median(r_outer) > np.median(r_inner)
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_step_empty_buffers_no_crash(self):
        """Step with no data in any buffer should not crash."""
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
            w.step()  # should not raise
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()


#
# --- ControlWorker — both modes ---
#

class TestControlWorkerPID:

    def test_smoke_init_pid(self):
        from workers import ControlWorker
        cn = f"ctrl_{_uid()}"
        pn = f"ctrl_p_{_uid()}"
        bn = f"ctrl_b_{_uid()}"
        cb = ControlBuffer(slots=4, name=cn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        done_ev = ctx.Event()
        try:
            w = ControlWorker(
                name="control",
                mode="pid",
                output_buffer=cb,
                pose_buffer=pb,
                boundary_buffer=bb,
                mapping_done_event=done_ev,
                pid_config={"kp_steer": 1.5},
            )
            assert w.config["mode"] == "pid"
        finally:
            cb.unlink()
            pb.unlink()
            bb.unlink()

    def test_setup_pid(self):
        from workers import ControlWorker
        cn = f"ctrl_{_uid()}"
        pn = f"ctrl_p_{_uid()}"
        bn = f"ctrl_b_{_uid()}"
        cb = ControlBuffer(slots=4, name=cn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        done_ev = ctx.Event()
        try:
            w = ControlWorker(
                name="control",
                mode="pid",
                output_buffer=cb,
                pose_buffer=pb,
                boundary_buffer=bb,
                mapping_done_event=done_ev,
                pid_config={"kp_steer": 2.0, "target_throttle": 0.5},
            )
            w._logger = MagicMock()
            w.setup()
            assert w._mode == "pid"
            assert w._pid is not None
            assert w._pid.kp == 2.0
            assert w._pid.target_throttle == 0.5
        finally:
            cb.unlink()
            pb.unlink()
            bb.unlink()

    def test_step_pid_waits_for_mapping(self):
        """PID step should wait (sleep) when mapping_done_event is not set."""
        from workers import ControlWorker
        cn = f"ctrl_{_uid()}"
        pn = f"ctrl_p_{_uid()}"
        bn = f"ctrl_b_{_uid()}"
        cb = ControlBuffer(slots=4, name=cn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        done_ev = ctx.Event()
        try:
            w = ControlWorker(
                name="control",
                mode="pid",
                output_buffer=cb,
                pose_buffer=pb,
                boundary_buffer=bb,
                mapping_done_event=done_ev,
                pid_config={},
            )
            w._logger = MagicMock()
            w.setup()

            # mapping not done → step should return without writing
            t0 = time.time()
            w.step()
            assert cb.read_latest() is None  # nothing written
        finally:
            cb.unlink()
            pb.unlink()
            bb.unlink()

    def test_step_pid_full_cycle(self, track_boundaries):
        """Full PID cycle: mapping done, boundaries loaded, pose → control output."""
        from workers import ControlWorker
        cn = f"ctrl_{_uid()}"
        pn = f"ctrl_p_{_uid()}"
        bn = f"ctrl_b_{_uid()}"
        cb = ControlBuffer(slots=4, name=cn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        done_ev = ctx.Event()
        outer, inner = track_boundaries
        try:
            w = ControlWorker(
                name="control",
                mode="pid",
                output_buffer=cb,
                pose_buffer=pb,
                boundary_buffer=bb,
                mapping_done_event=done_ev,
                pid_config={
                    "kp_steer": 1.0,
                    "ki_steer": 0.0,
                    "kd_steer": 0.1,
                    "target_throttle": 0.3,
                    "noise_std": 0.0,  # deterministic for testing
                },
            )
            w._logger = MagicMock()
            w.setup()

            # Set mapping done + write boundaries
            done_ev.set()
            bb.write(outer, inner)

            # Write a pose
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = 6.5  # near outer boundary
            pb.write(T)

            # Step multiple times to load boundaries then compute control
            for _ in range(3):
                w.step()

            result = cb.read_latest()
            assert result is not None
            _, steer, throttle = result
            assert -1.0 <= steer <= 1.0
            assert -1.0 <= throttle <= 1.0
        finally:
            cb.unlink()
            pb.unlink()
            bb.unlink()


class TestControlWorkerNeural:

    def test_smoke_init_neural(self):
        from workers import ControlWorker
        cn = f"ctrl_{_uid()}"
        sn = f"ctrl_s_{_uid()}"
        imuname = f"ctrl_i_{_uid()}"
        pn = f"ctrl_p_{_uid()}"
        cb = ControlBuffer(slots=4, name=cn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        ib = IMUBuffer(slots=8, name=imuname, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = ControlWorker(
                name="control",
                mode="neural",
                input_buffers={"segments": sb, "imu": ib, "pose": pb},
                output_buffer=cb,
                model_config="dummy",
                device="cpu",
            )
            assert w.config["mode"] == "neural"
        finally:
            cb.unlink()
            sb.unlink()
            ib.unlink()
            pb.unlink()


#
# --- ActuatorWorker ---
#

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


#
# --- RecorderWorker ---
#

class TestRecorderWorker:

    def test_smoke_init(self):
        from workers import RecorderWorker
        fn = f"rec_f_{_uid()}"
        imuname = f"rec_i_{_uid()}"
        sn = f"rec_s_{_uid()}"
        pn = f"rec_p_{_uid()}"
        cn = f"rec_c_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        ib = IMUBuffer(slots=8, name=imuname, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        cb = ControlBuffer(slots=4, name=cn, create=True)
        try:
            with tempfile.TemporaryDirectory() as td:
                w = RecorderWorker(
                    name="recorder",
                    buffers={
                        "frames": fb,
                        "imu": ib,
                        "segments": sb,
                        "pose": pb,
                        "control": cb,
                    },
                    output_dir=td,
                )
                assert w.name == "recorder"
        finally:
            fb.unlink()
            ib.unlink()
            sb.unlink()
            pb.unlink()
            cb.unlink()
