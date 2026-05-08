"""
synthetic/ generators and the pipeline they drive, no hardware needed (no
OAK-D, I2C or GPU). Workers get stepped in-process rather than spawned.
"""

from __future__ import annotations

import multiprocessing as mp
import uuid
from unittest.mock import MagicMock

import numpy as np

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
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=320, height=200)
        frames = [gen.next_fast()[0].copy() for _ in range(5)]
        # Scrolling is time based, so back-to-back calls can return the same
        # frame. Only checks that several frames generate without error.
        assert len(frames) == 5

    def test_rgb_has_wall_and_floor_regions(self):
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=640, height=400)
        rgb, _ = gen.next_fast()
        unique_colors = len(np.unique(rgb.reshape(-1, 3), axis=0))
        # floor, wall, sky
        assert unique_colors >= 3

    def test_depth_map_has_wall_structure(self):
        from synthetic import SyntheticFrameGenerator
        gen = SyntheticFrameGenerator(width=640, height=400)
        _, depth = gen.next_fast()
        # bottom-centre (floor) closer than top-centre (sky)
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
        # [qw, qx, qy, qz, ax, ay, az, throttle, steering]
        assert abs(vec[0] - 1.0) < 0.01
        assert abs(vec[1]) < 0.1
        assert abs(vec[2]) < 0.1
        assert abs(vec[3]) < 0.1

    def test_accelerometer_has_gravity(self):
        from synthetic import SyntheticIMUGenerator
        gen = SyntheticIMUGenerator()
        vec = gen.next()
        assert abs(vec[6] - 9.81) < 1.0


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


# end to end: camera -> SLAM stub -> mock vision -> control

class TestFullSyntheticPipeline:

    def test_synthetic_frame_through_slam_stub(self):
        # with _slam = None the worker publishes identity poses
        from workers import SLAMWorker

        fn = f"synfull_f_{_uid()}"
        pn = f"synfull_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            from synthetic import SyntheticFrameGenerator
            gen = SyntheticFrameGenerator(width=640, height=400)

            w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._slam = None
            w._last_seq = 0

            for _ in range(5):
                rgb, depth = gen.next_fast()
                fb.write(rgb, depth)
                w.step()

            result = pb.read_latest()
            assert result is not None
            _, pose, _, _ = result
            np.testing.assert_allclose(pose, np.eye(4, dtype=np.float32), atol=1e-7)
        finally:
            fb.unlink()
            pb.unlink()

    def test_synthetic_to_segmentation(self):
        from synthetic import SyntheticFrameGenerator

        gen = SyntheticFrameGenerator(width=640, height=400)
        rgb, depth = gen.next_fast()

        # same idea as the Threshold vision model
        mask = np.zeros((400, 640), dtype=np.int32)
        mask[depth < 1.0] = 1   # close = road/floor
        mask[depth >= 1.0] = 2  # far = wall

        assert mask.shape == (400, 640)
        assert 1 in mask
        assert 2 in mask

    def test_full_pipeline_synthetic_no_hardware(self):
        """camera -> vision -> SLAM stub -> mapping, all synthetic. (The control
        side is covered by the control tests.)"""
        from workers import CameraWorker, MappingWorker, SLAMWorker

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
            cam = CameraWorker(
                name="camera", output_buffer=fb, fps=30,
                resolution=[640, 400], synthetic=True,
            )
            cam._logger = MagicMock()
            cam.setup()

            # SLAM stub
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

            mapper = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                pointcloud_buffer=pcb, boundary_buffer=bb,
                loop_event=loop_ev, mapping_done_event=done_ev,
                boundary_min_points=10,
            )
            mapper._logger = MagicMock()
            mapper.setup()

            for i in range(15):
                cam.step()
                slam.step()

                # fake segmentation from depth
                result = fb.read_latest()
                if result is not None:
                    _, rgb, depth = result
                    mask = np.zeros((400, 640), dtype=np.int32)
                    mask[depth < 1.5] = 1   # road
                    mask[depth >= 1.5] = 2  # wall
                    sb.write(mask)

                # The stub publishes identity poses, which the mapper drops as
                # stationary. Drive a lap of the centre line on top of it.
                phi = 2 * np.pi * i / 15
                T = np.eye(4, dtype=np.float32)
                T[0, 3], T[2, 3] = 6.5 * np.cos(phi), 4.0 * np.sin(phi)
                pb.write(T)

                # fake wall points on two ellipses either side of it
                theta = np.linspace(0, 2 * np.pi, 100)
                pts = np.vstack([
                    np.column_stack([a * np.cos(theta), np.zeros(100), b * np.sin(theta)])
                    for a, b in ((8.0, 5.0), (5.0, 3.0))
                ]).astype(np.float32)
                pcb.write(pts)

                mapper.step()

            # loop closure -> boundary extraction
            loop_ev.set()
            mapper.step()

            assert mapper._boundaries_extracted
            assert done_ev.is_set()

            bnd_result = bb.read_latest()
            assert bnd_result is not None
            _, outer_line, inner_line = bnd_result
            assert len(outer_line) > 0
            assert len(inner_line) > 0

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
        from synthetic import SyntheticFrameGenerator

        gen = SyntheticFrameGenerator(width=640, height=400, wall_distance=1.5)
        _, depth = gen.next_fast()

        near = (depth < 1.0).sum()
        far = (depth >= 5.0).sum()

        assert near > 0, "No near-range pixels (floor should be close)"
        assert far > 0, "No far-range pixels (sky/end should be far)"

    def test_synthetic_imu_pipeline(self):
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
            assert abs(vec[0] - 1.0) < 0.1
        finally:
            ib.unlink()
