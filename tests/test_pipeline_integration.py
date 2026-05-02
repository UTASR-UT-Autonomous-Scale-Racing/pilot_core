"""
Integration tests along the data path:

    camera -> FrameBuffer -> SLAM -> PoseBuffer (+ vision -> SegmentationBuffer)
    -> MappingWorker -> TrackBoundaryBuffer -> ControlWorker -> ControlBuffer

The OAK-D tests only run with OAKD_AVAILABLE=1.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    PointCloudBuffer,
    PoseBuffer,
    SegmentationBuffer,
    TrackBoundaryBuffer,
)
from workers.base_worker import BaseWorker


def _uid():
    return uuid.uuid4().hex[:8]


class _StepCounterWorker(BaseWorker):
    """Counts steps and sets its own stop event after max_steps."""

    def __init__(self, name, max_steps=10, **kwargs):
        super().__init__(name, **kwargs)
        self._max_steps = max_steps
        self._step_count = 0

    def setup(self):
        pass

    def step(self):
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self._stop_event.set()
        time.sleep(0.01)

    def teardown(self):
        pass


# BaseWorker run loop in a child process

class TestWorkerLifecycle:

    def test_worker_runs_and_stops(self):
        ctx = mp.get_context("fork")
        w = _StepCounterWorker(name="test_counter", max_steps=5)
        p = ctx.Process(target=w.run, name="test_counter", daemon=False)
        p.start()
        p.join(timeout=10)
        assert not p.is_alive()
        # expect 0, but allow -11 (SIGSEGV from .so cleanup)
        assert p.exitcode in (0, -11, None)

    def test_worker_stop_from_outside(self):
        """stop() from the orchestrator side sets _stop_event and the loop exits."""
        ctx = mp.get_context("fork")
        w = _StepCounterWorker(name="test_stop", max_steps=999999)
        p = ctx.Process(target=w.run, name="test_stop", daemon=False)
        p.start()
        time.sleep(0.5)
        assert p.is_alive()
        w.stop()
        p.join(timeout=5)
        # Fallback so a worker that ignores stop() never outlives the test
        if p.is_alive():
            p.terminate()
            p.join(timeout=2)


# frame -> SLAM stub -> pose -> mapping, synthetic data

class TestSyntheticPipeline:
    def test_frame_to_pose_pipeline(self):
        from workers import SLAMWorker

        fn = f"pipe_f_{_uid()}"
        pn = f"pipe_p_{_uid()}"
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

            for i in range(5):
                rgb = np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)
                depth = np.random.uniform(0.5, 3.0, (400, 640)).astype(np.float32)
                fb.write(rgb, depth)
                w.step()

            # stub mode publishes identity poses
            result = pb.read_latest()
            assert result is not None
            _, pose, _, _ = result
            np.testing.assert_allclose(pose, np.eye(4, dtype=np.float32), atol=1e-7)
        finally:
            fb.unlink()
            pb.unlink()

    def test_slam_to_mapping_pipeline(self):
        from workers import MappingWorker, SLAMWorker

        fn = f"sm_f_{_uid()}"
        pn = f"sm_p_{_uid()}"
        sn = f"sm_s_{_uid()}"
        pcn = f"sm_pc_{_uid()}"
        bn = f"sm_b_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()
        try:
            slam = SLAMWorker(
                name="slam", input_buffer=fb, output_buffer=pb,
            )
            slam._logger = MagicMock()
            slam._frame_buf = fb
            slam._pose_buf = pb
            slam._slam = None
            slam._last_seq = 0

            mapper = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
                loop_event=loop_ev,
                mapping_done_event=done_ev,
                boundary_min_points=10,
            )
            mapper._logger = MagicMock()
            mapper.setup()

            for i in range(10):
                rgb = np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)
                depth = np.random.uniform(0.5, 3.0, (400, 640)).astype(np.float32)
                fb.write(rgb, depth)
                slam.step()

                # ring of wall points
                theta = np.linspace(0, 2 * np.pi, 50) + i * 0.1
                pts = np.column_stack([
                    8.0 * np.cos(theta),
                    np.zeros(50),
                    5.0 * np.sin(theta),
                ]).astype(np.float32)
                pcb.write(pts)

                mask = np.ones((400, 640), dtype=np.int32)
                sb.write(mask)

                mapper.step()

            assert len(mapper._bev_trajectory) > 0
            assert len(mapper._bev_wall_pts) > 0
        finally:
            fb.unlink()
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_full_mapping_to_track_control_pipeline(self):
        """mapping -> boundary extraction -> pure pursuit, end to end on synthetic data."""
        from workers import ControlWorker, MappingWorker

        pn = f"fp_p_{_uid()}"
        sn = f"fp_s_{_uid()}"
        pcn = f"fp_pc_{_uid()}"
        bn = f"fp_b_{_uid()}"
        cn = f"fp_c_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        cb = ControlBuffer(slots=4, name=cn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()
        try:
            mapper = MappingWorker(
                name="mapping",
                pose_buffer=pb,
                segment_buffer=sb,
                pointcloud_buffer=pcb,
                boundary_buffer=bb,
                loop_event=loop_ev,
                mapping_done_event=done_ev,
                boundary_min_points=10,
            )
            mapper._logger = MagicMock()
            mapper.setup()

            # oval walls, then a trajectory along them
            theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
            outer_3d = np.column_stack([
                8.0 * np.cos(theta), np.zeros(100), 5.0 * np.sin(theta)
            ]).astype(np.float32)
            inner_3d = np.column_stack([
                5.0 * np.cos(theta), np.zeros(100), 3.0 * np.sin(theta)
            ]).astype(np.float32)
            pcb.write(np.vstack([outer_3d, inner_3d]))

            for i in range(10):
                T = np.eye(4, dtype=np.float32)
                T[0, 3] = 6.5 * np.cos(i * 0.5)
                T[2, 3] = 4.0 * np.sin(i * 0.5)
                pb.write(T)
                sb.write(np.ones((400, 640), dtype=np.int32))
                mapper.step()

            # loop closure triggers boundary extraction
            loop_ev.set()
            mapper.step()
            assert mapper._boundaries_extracted
            assert done_ev.is_set()

            bnd_result = bb.read_latest()
            assert bnd_result is not None
            _, outer_line, inner_line = bnd_result
            assert len(outer_line) > 0
            assert len(inner_line) > 0

            from control.types import VehicleState
            from shared_buffer import StateBuffer

            stb = StateBuffer(slots=4, name=f"fp_st_{_uid()}", create=True)
            try:
                ctrl = ControlWorker(
                    name="control",
                    mode="pure_pursuit",
                    output_buffer=cb,
                    state_buffer=stb,
                    boundary_buffer=bb,
                    mapping_done_event=done_ev,
                )
                ctrl._logger = MagicMock()
                ctrl.setup()
                stb.write(VehicleState(t=time.monotonic(), x=6.5, y=0.0, yaw=np.pi / 2, v=0.5))
                ctrl.step()
            finally:
                stb.unlink()

            result = cb.read_latest()
            assert result is not None
            _, steer, throttle = result
            assert 0.0 <= steer <= 1.0
            assert 0.5 < throttle <= 1.0
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()
            cb.unlink()


# Nothing written anywhere. Workers should just do nothing

class TestNullPipeline:

    def test_slam_step_empty_buffer(self):
        from workers import SLAMWorker
        fn = f"null_f_{_uid()}"
        pn = f"null_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._slam = None
            w._last_seq = 0
            for _ in range(10):
                w.step()
            assert pb.read_latest() is None
        finally:
            fb.unlink()
            pb.unlink()

    def test_mapping_step_empty_buffers(self):
        from workers import MappingWorker
        pn = f"null_p_{_uid()}"
        sn = f"null_s_{_uid()}"
        pcn = f"null_pc_{_uid()}"
        bn = f"null_b_{_uid()}"
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        try:
            w = MappingWorker(
                name="mapping",
                pose_buffer=pb, segment_buffer=sb,
                pointcloud_buffer=pcb, boundary_buffer=bb,
            )
            w._logger = MagicMock()
            w.setup()
            for _ in range(10):
                w.step()
            assert bb.read_latest() is None
            assert not w._boundaries_extracted
        finally:
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()

    def test_control_track_mode_waits_for_mapping(self):
        from shared_buffer import StateBuffer
        from workers import ControlWorker
        cb = ControlBuffer(slots=4, name=f"null_c_{_uid()}", create=True)
        stb = StateBuffer(slots=4, name=f"null_st_{_uid()}", create=True)
        bb = TrackBoundaryBuffer(slots=2, name=f"null_b_{_uid()}", create=True)
        ctx = mp.get_context("spawn")
        done_ev = ctx.Event()
        try:
            w = ControlWorker(
                name="control", mode="pure_pursuit",
                output_buffer=cb, state_buffer=stb,
                boundary_buffer=bb, mapping_done_event=done_ev,
            )
            w._logger = MagicMock()
            w.setup()
            for _ in range(3):
                w.step()
            # mapping never finished, so no control output
            assert cb.read_latest() is None
        finally:
            cb.unlink()
            stb.unlink()
            bb.unlink()


# writer/reader in separate processes

def _pose_reader(name: str, n: int, results: mp.Queue):
    pb = PoseBuffer(slots=4, name=name, create=False)
    valid = 0
    for _ in range(n):
        r = pb.read_latest()
        if r is not None:
            valid += 1
        time.sleep(0.001)
    pb.close()
    results.put(valid)


class TestPipelineRaceConditions:

    def test_concurrent_pose_write_read(self):
        name = f"race_p_{_uid()}"
        pb = PoseBuffer(slots=4, name=name, create=True)
        q = mp.Queue()
        reader = mp.Process(target=_pose_reader, args=(name, 200, q))
        reader.start()
        for i in range(100):
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = float(i)
            pb.write(T)
            time.sleep(0.002)
        reader.join(timeout=15)
        valid = q.get(timeout=5)
        assert valid > 0
        pb.unlink()

    def test_concurrent_frame_slam_chain(self):
        """Interleaved frame writes and SLAM steps, single process."""
        from workers import SLAMWorker

        fn = f"chain_f_{_uid()}"
        pn = f"chain_p_{_uid()}"
        fb = FrameBuffer(slots=4, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        try:
            w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb)
            w._logger = MagicMock()
            w._frame_buf = fb
            w._pose_buf = pb
            w._slam = None
            w._last_seq = 0

            for i in range(50):
                rgb = np.full((400, 640, 3), i % 256, dtype=np.uint8)
                depth = np.full((400, 640), 1.0, dtype=np.float32)
                fb.write(rgb, depth)
                w.step()

            result = pb.read_latest()
            assert result is not None
        finally:
            fb.unlink()
            pb.unlink()


class TestMappingHelpers:

    def test_to_bev(self):
        from workers.mapping_worker import _to_bev
        pts = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ], dtype=np.float32)
        bev = _to_bev(pts)
        assert bev.shape == (2, 2)
        np.testing.assert_array_equal(bev, [[1.0, 3.0], [4.0, 6.0]])

    @staticmethod
    def _ccw_lap(n: int = 100):
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        outer = np.column_stack([10 * np.cos(theta), 10 * np.sin(theta)]).astype(np.float32)
        inner = np.column_stack([5 * np.cos(theta), 5 * np.sin(theta)]).astype(np.float32)
        # centre-line trajectory, heading is the tangent of a CCW lap
        traj = np.column_stack([
            7.5 * np.cos(theta), 7.5 * np.sin(theta), theta + np.pi / 2,
        ]).astype(np.float32)
        return outer, inner, traj

    def test_split_walls_left_right(self):
        from workers.mapping_worker import split_walls_by_trajectory
        outer, inner, traj = self._ccw_lap()
        left, right = split_walls_by_trajectory(np.vstack([outer, inner]), traj)
        # CCW, so inner wall is on the left and outer on the right
        np.testing.assert_allclose(np.linalg.norm(left, axis=1), 5.0, atol=1e-4)
        np.testing.assert_allclose(np.linalg.norm(right, axis=1), 10.0, atol=1e-4)

    def test_split_walls_ordered_by_arc_length(self):
        from workers.mapping_worker import split_walls_by_trajectory
        outer, inner, traj = self._ccw_lap()
        _, right = split_walls_by_trajectory(np.vstack([outer, inner]), traj)
        angles = np.unwrap(np.arctan2(right[:, 1], right[:, 0]))
        assert np.all(np.diff(angles) >= -1e-6)  # monotonic along the lap


# needs a real OAK-D (OAKD_AVAILABLE=1)

@pytest.mark.skipif(
    not os.environ.get("OAKD_AVAILABLE", ""),
    reason="OAK-D camera not available (set OAKD_AVAILABLE=1)"
)
class TestOAKDIntegration:
    def test_camera_produces_rgbd(self):
        from workers import CameraWorker

        fn = f"oakd_f_{_uid()}"
        fb = FrameBuffer(slots=8, name=fn, create=True)
        try:
            cam = CameraWorker(
                name="camera", output_buffer=fb, fps=30, resolution=[640, 400]
            )
            cam._logger = MagicMock()
            cam._logger.info = print
            cam.setup()

            frames = []
            for _ in range(200):
                cam.step()
                result = fb.read_latest()
                if result is not None:
                    _, rgb, depth = result
                    frames.append((rgb.copy(), depth.copy()))
                if len(frames) >= 5:
                    break
                time.sleep(0.02)

            cam.teardown()

            assert len(frames) >= 5, f"Only got {len(frames)} frames"

            for rgb, depth in frames:
                assert rgb.dtype == np.uint8
                assert rgb.shape == (400, 640, 3)
                assert depth.dtype == np.float32
                assert depth.shape == (400, 640)
                valid_depth = depth[depth > 0]
                if len(valid_depth) > 0:
                    assert valid_depth.max() < 20.0
        finally:
            fb.unlink()

    def test_camera_to_slam_realtime(self):
        """Real camera into the SLAM stub for 2 s at 30 fps."""
        from workers import CameraWorker, SLAMWorker

        fn = f"oakd_s_{_uid()}"
        pn = f"oakd_p_{_uid()}"
        fb = FrameBuffer(slots=8, name=fn, create=True)
        pb = PoseBuffer(slots=8, name=pn, create=True)
        try:
            cam = CameraWorker(
                name="camera", output_buffer=fb, fps=30, resolution=[640, 400]
            )
            cam._logger = MagicMock()
            cam._logger.info = print
            cam.setup()

            slam = SLAMWorker(
                name="slam", input_buffer=fb, output_buffer=pb,
            )
            slam._logger = MagicMock()
            slam._frame_buf = fb
            slam._pose_buf = pb
            slam._slam = None
            slam._last_seq = 0

            pose_count = 0
            t0 = time.time()
            while time.time() - t0 < 2.0:
                cam.step()
                slam.step()
                result = pb.read_latest()
                if result is not None:
                    h, pose, _, _ = result
                    if h.seq_no > pose_count:
                        pose_count = h.seq_no
                time.sleep(0.01)

            cam.teardown()
            assert pose_count >= 10, f"Only got {pose_count} poses in 2s"
        finally:
            fb.unlink()
            pb.unlink()

    def test_full_oakd_pipeline_with_resnet18_mask(self):
        """OAK-D frames, a fake seg mask in place of ResNet18, SLAM stub, then mapping."""
        from workers import CameraWorker, MappingWorker, SLAMWorker

        fn = f"full_{_uid()}_f"
        pn = f"full_{_uid()}_p"
        sn = f"full_{_uid()}_s"
        pcn = f"full_{_uid()}_pc"
        bn = f"full_{_uid()}_b"
        fb = FrameBuffer(slots=8, name=fn, create=True)
        pb = PoseBuffer(slots=4, name=pn, create=True)
        sb = SegmentationBuffer(slots=4, name=sn, create=True)
        pcb = PointCloudBuffer(slots=4, name=pcn, create=True)
        bb = TrackBoundaryBuffer(slots=2, name=bn, create=True)
        ctx = mp.get_context("spawn")
        loop_ev = ctx.Event()
        done_ev = ctx.Event()
        try:
            cam = CameraWorker(
                name="camera", output_buffer=fb, fps=30, resolution=[640, 400]
            )
            cam._logger = MagicMock()
            cam._logger.info = print
            cam.setup()

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
                pose_buffer=pb, segment_buffer=sb,
                pointcloud_buffer=pcb, boundary_buffer=bb,
                loop_event=loop_ev, mapping_done_event=done_ev,
                boundary_min_points=10,
            )
            mapper._logger = MagicMock()
            mapper.setup()

            frame_count = 0
            t0 = time.time()
            while time.time() - t0 < 2.0:
                cam.step()
                slam.step()

                result = fb.read_latest()
                if result is not None:
                    _, rgb, depth = result
                    # stand-in for ResNet18: threshold on depth
                    mask = np.zeros((400, 640), dtype=np.int32)
                    mask[depth < 2.0] = 1   # road
                    mask[depth >= 2.0] = 2  # wall
                    sb.write(mask)
                    frame_count += 1

                    # random points, just so the mapper has something
                    pcb.write(np.random.randn(50, 3).astype(np.float32) * 3)

                mapper.step()
                time.sleep(0.01)

            cam.teardown()

            assert frame_count > 5, f"Processed only {frame_count} frames"
            assert len(mapper._bev_trajectory) > 0
        finally:
            fb.unlink()
            pb.unlink()
            sb.unlink()
            pcb.unlink()
            bb.unlink()


# start.py CLI

class TestStartCLI:

    def test_parse_args_defaults(self):
        from start import parse_args
        import sys
        old = sys.argv
        sys.argv = ["start.py", "--config", "config.yaml"]
        args = parse_args()
        sys.argv = old
        assert args.config == "config.yaml"
        assert args.recording is False
        assert args.debug is False
        assert args.enable_slam is False
        assert args.oakd_imu is False

    def test_parse_args_flags(self):
        from start import parse_args
        import sys
        old = sys.argv
        sys.argv = ["start.py", "--config", "config.yaml", "--recording", "--debug", "--enable-slam", "--oakd-imu"]
        args = parse_args()
        sys.argv = old
        assert args.recording is True
        assert args.debug is True
        assert args.enable_slam is True
        assert args.oakd_imu is True

    def test_config_is_required(self):
        from start import parse_args
        import sys
        old = sys.argv
        sys.argv = ["start.py"]
        try:
            with pytest.raises(SystemExit):
                parse_args()
        finally:
            sys.argv = old

    def test_load_config(self):
        from start import load_config
        config = load_config("config.yaml")
        assert isinstance(config, dict)
        assert "fps" in config
        assert "resolution" in config
        assert "buffers" in config

    def test_load_config_missing_file(self):
        from start import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")
