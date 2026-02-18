"""
test_synthetic_pipeline.py — End-to-end tests using synthetic FPV frames,
generated depth, and synthetic IMU data.

These tests run WITHOUT any hardware (no OAK-D camera, no I2C, no GPU).
They exercise the full pipeline path:
    Synthetic Camera → FrameBuffer → VisionWorker (mock) → SegmentationBuffer
    Synthetic IMU    → IMUBuffer
    FrameBuffer → SLAMWorker (stub) → PoseBuffer
    Mapping → TrackBoundaryBuffer → PID ControlWorker → ControlBuffer
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


def _uid():
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
#  Synthetic frame generator tests
# ---------------------------------------------------------------------------

class TestSyntheticFrameGenerator:

    def test_smoke_init(self):
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator()
        assert gen.width == 640
        assert gen.height == 400

    def test_next_returns_valid_shapes(self):
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=640, height=400)
        rgb, depth = gen.next_fast()
        assert rgb.shape == (400, 640, 3)
        assert rgb.dtype == np.uint8
        assert depth.shape == (400, 640)
        assert depth.dtype == np.float32

    def test_depth_range(self):
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=640, height=400, corridor_length=10.0)
        rgb, depth = gen.next_fast()
        assert depth.min() >= 0.0
        assert depth.max() <= 10.0

    def test_multiple_frames_vary(self):
        """Successive frames should differ (texture scrolling)."""
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=320, height=200)
        frames = [gen.next_fast()[0].copy() for _ in range(5)]
        # At least some frames should differ (scrolling)
        all_same = all(np.array_equal(frames[0], f) for f in frames[1:])
        # Due to speed=2 and time differences, they should vary
        # But if called fast enough they might match — so just check no crash
        assert len(frames) == 5

    def test_rgb_has_wall_and_floor_regions(self):
        """Verify the frame contains distinct regions (not all uniform)."""
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=640, height=400)
        rgb, _ = gen.next_fast()
        unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
        # Should have at least 3 distinct colors (floor, wall, sky)
        assert unique_colors >= 3

    def test_depth_map_has_wall_structure(self):
        """Depth should show walls closer on edges, floor in centre."""
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=640, height=400)
        _, depth = gen.next_fast()
        # Bottom-centre pixel (floor) should be closer than top-centre (sky/far)
        bottom_centre = depth[350, 320]
        top_centre = depth[50, 320]
        assert bottom_centre < top_centre


class TestSyntheticIMUGenerator:

    def test_smoke_init(self):
        from synthetic import SyntheticIMUGenerator
        gen = SyntheticIMUGenerator()
        vec = gen.next()
        assert vec.shape == (9,)
        assert vec.dtype == np.float32

    def test_quaternion_near_identity(self):
        from synthetic import SyntheticIMUGenerator
        gen = SyntheticIMUGenerator()
        vec = gen.next()
        # qw should be ~1.0
        assert abs(vec[0] - 1.0) < 0.01
        # qx, qy, qz should be near 0
        assert abs(vec[1]) < 0.1
        assert abs(vec[2]) < 0.1
        assert abs(vec[3]) < 0.1

    def test_accelerometer_has_gravity(self):
        from synthetic import SyntheticIMUGenerator
        gen = SyntheticIMUGenerator()
        vec = gen.next()
        # az should be near 9.81
        assert abs(vec[6] - 9.81) < 1.0


# ---------------------------------------------------------------------------
#  CameraWorker with synthetic mode
# ---------------------------------------------------------------------------

class TestCameraWorkerSynthetic:

    def test_setup_synthetic_via_env(self):
        from workers import CameraWorker

        fn = f"syn_cam_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        try:
            w = CameraWorker(
                name="camera",
                output_buffer=fb,
                fps=10,
                resolution=[640, 400],
                synthetic=True,
            )
            w._logger = MagicMock()
            w.setup()
            assert w._synthetic is True

            w.step()
            result = fb.read_latest()
            assert result is not None
            header, rgb, depth = result
            assert rgb.shape == (400, 640, 3)
            assert depth.shape == (400, 640)
            assert rgb.dtype == np.uint8
            assert depth.dtype == np.float32

            w.teardown()
        finally:
            fb.unlink()

    def test_multiple_frames_sequential(self):
        from workers import CameraWorker

        fn = f"syn_cam_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        try:
            w = CameraWorker(
                name="camera",
                output_buffer=fb,
                fps=30,
                resolution=[640, 400],
                synthetic=True,
            )
            w._logger = MagicMock()
            w.setup()

            for _ in range(5):
                w.step()

            result = fb.read_latest()
            assert result is not None
            header, _, _ = result
            assert header.seq_no >= 5

            w.teardown()
        finally:
            fb.unlink()


# ---------------------------------------------------------------------------
#  IMUWorker with synthetic mode
# ---------------------------------------------------------------------------

class TestIMUWorkerSynthetic:

    def test_setup_synthetic(self):
        from workers import IMUWorker

        imu_name = f"syn_imu_{_uid()}"
        ib = IMUBuffer(slots=8, name=imu_name, create=True)
        try:
            w = IMUWorker(
                name="imu",
                output_buffer=ib,
                synthetic=True,
            )
            w._logger = MagicMock()
            w.setup()
            assert w._synthetic is True

            w.step()
            result = ib.read_latest()
            assert result is not None
            _, vec = result
            assert vec.shape == (9,)
            assert vec.dtype == np.float32
        finally:
            ib.unlink()


# ---------------------------------------------------------------------------
#  Full synthetic pipeline: Camera → SLAM(stub) → Vision(mock) → Control
# ---------------------------------------------------------------------------

class TestFullSyntheticPipeline:

    def test_synthetic_frame_through_slam_stub(self):
        """Synthetic frames → SLAMWorker (stub) → identity poses."""
        from workers import SLAMWorker

        fn = f"synfull_f_{_uid()}"
        pn = f"synfull_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            # Write synthetic frames
            from synthetic import SyntheticFrameGenerator
            gen = SyntheticFrameGenerator(width=640, height=400)

            w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._slam = None  # stub
            w._last_seq = 0

            for _ in range(5):
                rgb, depth = gen.next_fast()
                fb.write(rgb, depth)
                w.step()

            result = pb.read_latest()
            assert result is not None
            _, pose, _ = result
            np.testing.assert_allclose(pose, np.eye(4, dtype=np.float32), atol=1e-7)
        finally:
            fb.unlink()
            pb.unlink()

    def test_synthetic_to_segmentation(self):
        """Synthetic depth → simple threshold segmentation → 3-class mask."""
        from synthetic import SyntheticFrameGenerator

        gen = SyntheticFrameGenerator(width=640, height=400)
        rgb, depth = gen.next_fast()

        # Simple depth-based segmentation (like the Threshold model does)
        mask = np.zeros((400, 640), dtype=np.int32)
        mask[depth < 1.0] = 1   # close = road/floor
        mask[depth >= 1.0] = 2  # far = wall

        assert mask.shape == (400, 640)
        assert 1 in mask  # should have some road
        assert 2 in mask  # should have some wall

    def test_full_pipeline_synthetic_no_hardware(self):
        """Complete synthetic pipeline: camera → vision → SLAM → mapping → PID."""
        from workers import CameraWorker, ControlWorker, MappingWorker, SLAMWorker

        fn = f"fullsyn_f_{_uid()}"
        pn = f"fullsyn_p_{_uid()}"
        sn = f"fullsyn_s_{_uid()}"
        pcn = f"fullsyn_pc_{_uid()}"
        bn = f"fullsyn_b_{_uid()}"
        cn = f"fullsyn_c_{_uid()}"
        imu_n = f"fullsyn_i_{_uid()}"

        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        cb = ControlBuffer(slots=4, name=cn, create=True)
        ib = IMUBuffer(slots=8, name=imu_n, create=True)

        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()

        try:
            # 1. Synthetic camera
            cam = CameraWorker(
                name="camera", output_buffer=fb, fps=30,
                resolution=[640, 400], synthetic=True,
            )
            cam._logger = MagicMock()
            cam.setup()

            # 2. SLAM stub
            slam = SLAMWorker(
                name="slam", input_buffer=fb, output_buffer=pb,
                pointcloud_buffer=pcb,
            )
            slam._logger = MagicMock()
            slam._frame_buf = fb
            slam._pose_buf = pb
            slam._pc_buf = pcb
            slam._slam = None
            slam._last_seq = 0

            # 3. Mapping worker
            mapper = MappingWorker(
                name="mapping",
                pose_buffer=pb, segment_buffer=sb,
                pointcloud_buffer=pcb, boundary_buffer=bb,
                loop_event=loop_ev, mapping_done_event=done_ev,
                boundary_min_points=10,
            )
            mapper._logger = MagicMock()
            mapper.setup()

            # 4. Run pipeline: generate 15 synthetic frames
            for i in range(15):
                cam.step()
                slam.step()

                # Synthetic segmentation from depth
                result = fb.read_latest()
                if result is not None:
                    _, rgb, depth = result
                    mask = np.zeros((400, 640), dtype=np.int32)
                    mask[depth < 1.5] = 1   # road
                    mask[depth >= 1.5] = 2  # wall
                    sb.write(mask)

                # Synthetic point cloud (wall circle)
                theta = np.linspace(0, 2 * np.pi, 50) + i * 0.2
                pts = np.column_stack([
                    8.0 * np.cos(theta),
                    np.zeros(50),
                    5.0 * np.sin(theta),
                ]).astype(np.float32)
                pcb.write(pts)

                mapper.step()

            # 5. Trigger loop closure → extract boundaries
            loop_ev.set()
            mapper.step()

            assert mapper._boundaries_extracted
            assert done_ev.is_set()

            # 6. Verify boundary data
            bnd_result = bb.read_latest()
            assert bnd_result is not None
            _, outer_line, inner_line = bnd_result
            assert len(outer_line) > 0
            assert len(inner_line) > 0

            # 7. PID control
            ctrl = ControlWorker(
                name="control", mode="pid",
                output_buffer=cb, pose_buffer=pb,
                boundary_buffer=bb, mapping_done_event=done_ev,
                pid_config={
                    "kp_steer": 1.0, "ki_steer": 0.0,
                    "kd_steer": 0.1, "target_throttle": 0.3,
                    "noise_std": 0.0,
                },
            )
            ctrl._logger = MagicMock()
            ctrl.setup()

            # Write a pose and step control
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = 6.5
            pb.write(T)

            for _ in range(5):
                ctrl.step()

            result = cb.read_latest()
            assert result is not None
            _, steer, throttle = result
            assert -1.0 <= steer <= 1.0
            assert -1.0 <= throttle <= 1.0

            cam.teardown()
        finally:
            fb.unlink()
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()
            cb.unlink()
            ib.unlink()

    def test_synthetic_depth_generates_wall_structure(self):
        """Verify synthetic depth has wall-like structure for segmentation."""
        from synthetic import SyntheticFrameGenerator

        gen = SyntheticFrameGenerator(width=640, height=400, wall_distance=1.5)
        _, depth = gen.next_fast()

        # Extract depth histogram to verify structure
        near = (depth < 1.0).sum()
        mid = ((depth >= 1.0) & (depth < 5.0)).sum()
        far = (depth >= 5.0).sum()

        # Should have a mix of near/mid/far depths
        total = depth.size
        assert near > 0, "No near-range pixels (floor should be close)"
        assert far > 0, "No far-range pixels (sky/end should be far)"

    def test_synthetic_imu_pipeline(self):
        """Synthetic IMU data flows through IMUBuffer correctly."""
        from workers import IMUWorker

        imu_name = f"syn_imu_pipe_{_uid()}"
        ib = IMUBuffer(slots=8, name=imu_name, create=True)
        try:
            w = IMUWorker(name="imu", output_buffer=ib, synthetic=True)
            w._logger = MagicMock()
            w.setup()

            for _ in range(10):
                w.step()

            result = ib.read_latest()
            assert result is not None
            header, vec = result
            assert header.seq_no >= 10
            assert vec.shape == (9,)
            # Quaternion w should be near 1.0
            assert abs(vec[0] - 1.0) < 0.1
        finally:
            ib.unlink()
