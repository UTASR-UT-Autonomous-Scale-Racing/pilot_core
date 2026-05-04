"""
shared_buffer/: SlotHeader, the raw SharedRingBuffer and each typed buffer,
plus a few cross-process read/write checks at the end.
"""

from __future__ import annotations

import multiprocessing as mp
import struct
import time
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest

from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    PointCloudBuffer,
)
from shared_buffer.ring_buffer import SharedRingBuffer
from shared_buffer.schema import (
    HEADER_SIZE,
    SCHEMA_VERSION,
    DTypeTag,
    SlotHeader,
)


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestSlotHeader:
    def test_header_size_is_64(self):
        assert HEADER_SIZE == 64

    def test_pack_unpack_roundtrip(self):
        h = SlotHeader(
            seq_no=42,
            produced_ts=1234567890.123,
            read_ts=0.0,
            payload_size=128,
            schema_version=SCHEMA_VERSION,
            dtype_tag=int(DTypeTag.FRAME),
        )
        raw = h.pack()
        assert len(raw) == HEADER_SIZE
        h2 = SlotHeader.unpack(raw)
        assert h2.seq_no == 42
        assert h2.payload_size == 128
        assert h2.dtype_tag == int(DTypeTag.FRAME)
        assert abs(h2.produced_ts - 1234567890.123) < 1e-6

    def test_unpack_zeroed_buffer(self):
        h = SlotHeader.unpack(b"\x00" * HEADER_SIZE)
        assert h.seq_no == 0
        assert h.payload_size == 0


# raw ring buffer

class TestSharedRingBuffer:

    def test_smoke_create_close(self):
        name = f"test_rb_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=4, slot_payload_size=128, create=True)
        assert rb.name == name
        assert rb.slots == 4
        rb.unlink()

    def test_write_read_roundtrip(self):
        name = f"test_rb_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=4, slot_payload_size=64, create=True)
        payload = b"hello world" + b"\x00" * 53
        header = rb.write(payload)
        assert header.seq_no == 1
        result = rb.read_latest()
        assert result is not None
        h, data = result
        assert h.seq_no == 1
        assert data[:11] == b"hello world"
        rb.unlink()

    def test_read_before_write_returns_none(self):
        name = f"test_rb_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=2, slot_payload_size=32, create=True)
        assert rb.read_latest() is None
        rb.unlink()

    def test_overwrite_ring_returns_latest(self):
        name = f"test_rb_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=2, slot_payload_size=8, create=True)
        for i in range(10):
            rb.write(struct.pack("<Q", i))
        result = rb.read_latest()
        assert result is not None
        h, data = result
        assert h.seq_no == 10
        val = struct.unpack("<Q", data)[0]
        assert val == 9
        rb.unlink()

    def test_read_only_handle_cannot_write(self):
        name = f"test_rb_{_uid()}"
        rb_w = SharedRingBuffer(name=name, slots=2, slot_payload_size=16, create=True)
        rb_r = SharedRingBuffer(name=name, slots=2, slot_payload_size=16, create=False)
        with pytest.raises(RuntimeError, match="read-only"):
            rb_r.write(b"\x00" * 16)
        rb_r.close()
        rb_w.unlink()

    def test_context_manager(self):
        name = f"test_rb_{_uid()}"
        with SharedRingBuffer(name=name, slots=2, slot_payload_size=16, create=True) as rb:
            rb.write(b"\xff" * 16)
            result = rb.read_latest()
            assert result is not None
        # the context manager only closes, it doesn't unlink, so do that here
        try:
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

    def test_oversized_payload_is_rejected(self):
        """A payload larger than the slot raises instead of being silently truncated."""
        name = f"test_rb_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=2, slot_payload_size=8, create=True)
        try:
            with pytest.raises(ValueError):
                rb.write(b"\xAA" * 100)
            assert rb.read_latest() is None
        finally:
            rb.unlink()

    def test_latest_seq_and_explicit_timestamp(self):
        name = f"test_rb_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=2, slot_payload_size=8, create=True)
        try:
            assert rb.latest_seq() == 0
            rb.write(b"\x01", ts=123.5)
            rb.write(b"\x02")
            assert rb.latest_seq() == 2
            header, _ = rb.read_latest()
            assert header.seq_no == 2
            import time
            assert abs(header.produced_ts - time.monotonic()) < 1.0   # monotonic clock by default
        finally:
            rb.unlink()


class TestFrameBuffer:

    def test_smoke_init(self, frame_buffer):
        assert frame_buffer is not None

    def test_read_before_write_returns_none(self, frame_buffer):
        assert frame_buffer.read_latest() is None

    def test_write_read_roundtrip(self, frame_buffer, rgb_frame, depth_frame):
        frame_buffer.write(rgb_frame, depth_frame)
        result = frame_buffer.read_latest()
        assert result is not None
        header, rgb_out, depth_out = result
        assert header.seq_no == 1
        np.testing.assert_array_equal(rgb_out, rgb_frame)
        np.testing.assert_allclose(depth_out, depth_frame, atol=1e-6)

    def test_dtype_and_shape(self, frame_buffer, rgb_frame, depth_frame):
        frame_buffer.write(rgb_frame, depth_frame)
        _, rgb_out, depth_out = frame_buffer.read_latest()
        assert rgb_out.dtype == np.uint8
        assert depth_out.dtype == np.float32
        assert rgb_out.shape == (400, 640, 3)
        assert depth_out.shape == (400, 640)

    def test_multiple_writes_returns_latest(self, frame_buffer):
        for i in range(6):
            rgb = np.full((400, 640, 3), i % 256, dtype=np.uint8)
            depth = np.full((400, 640), float(i), dtype=np.float32)
            frame_buffer.write(rgb, depth)
        result = frame_buffer.read_latest()
        assert result is not None
        header, _, depth_out = result
        assert header.seq_no == 6
        np.testing.assert_allclose(depth_out, 5.0, atol=1e-6)


class TestIMUBuffer:

    def test_smoke_init(self, imu_buffer):
        assert imu_buffer is not None

    def test_read_before_write_returns_none(self, imu_buffer):
        assert imu_buffer.read_latest() is None

    def test_write_read_roundtrip(self, imu_buffer, imu_vector):
        imu_buffer.write(imu_vector)
        result = imu_buffer.read_latest()
        assert result is not None
        header, vec = result
        assert header.seq_no == 1
        np.testing.assert_allclose(vec, imu_vector, atol=1e-6)

    def test_imu_shape_and_dtype(self, imu_buffer, imu_vector):
        imu_buffer.write(imu_vector)
        _, vec = imu_buffer.read_latest()
        assert vec.shape == (9,)
        assert vec.dtype == np.float32

    def test_wrong_size_vector_write_is_rejected(self, imu_buffer):
        with pytest.raises(ValueError):
            imu_buffer.write(np.zeros(5, dtype=np.float32))
        with pytest.raises(ValueError):
            imu_buffer.write(np.zeros((3, 3), dtype=np.float32))
        assert imu_buffer.read_latest() is None


class TestSegmentationBuffer:

    def test_smoke_init(self, segmentation_buffer):
        assert segmentation_buffer is not None

    def test_read_before_write_returns_none(self, segmentation_buffer):
        assert segmentation_buffer.read_latest() is None

    def test_write_read_roundtrip(self, segmentation_buffer, segmentation_mask):
        segmentation_buffer.write(segmentation_mask)
        result = segmentation_buffer.read_latest()
        assert result is not None
        header, mask_out = result
        assert header.seq_no == 1
        np.testing.assert_array_equal(mask_out, segmentation_mask)

    def test_mask_shape(self, segmentation_buffer, segmentation_mask):
        segmentation_buffer.write(segmentation_mask)
        _, mask_out = segmentation_buffer.read_latest()
        assert mask_out.shape == (400, 640)


class TestPoseBuffer:

    def test_smoke_init(self, pose_buffer):
        assert pose_buffer is not None

    def test_read_before_write_returns_none(self, pose_buffer):
        assert pose_buffer.read_latest() is None

    def test_write_read_identity(self, pose_buffer):
        identity = np.eye(4, dtype=np.float32)
        pose_buffer.write(identity)
        result = pose_buffer.read_latest()
        assert result is not None
        _, pose_out, mp_out, speed = result
        np.testing.assert_allclose(pose_out, identity, atol=1e-7)
        assert mp_out.shape == (0, 3)
        assert speed == 0.0

    def test_write_pose_with_map_points(self, pose_buffer, pose_4x4, point_cloud):
        pts = point_cloud[:100]
        pose_buffer.write(pose_4x4, map_points=pts)
        result = pose_buffer.read_latest()
        assert result is not None
        _, pose_out, mp_out, _ = result
        np.testing.assert_allclose(pose_out, pose_4x4, atol=1e-6)
        assert mp_out.shape == (100, 3)
        np.testing.assert_allclose(mp_out, pts, atol=1e-6)

    def test_pose_null_map_points(self, pose_buffer):
        pose = np.eye(4, dtype=np.float32)
        pose_buffer.write(pose, map_points=None)
        _, _, mp_out, _ = pose_buffer.read_latest()
        assert mp_out.shape == (0, 3)

    def test_speed_round_trip(self, pose_buffer, point_cloud):
        pose_buffer.write(np.eye(4, dtype=np.float32), map_points=point_cloud[:10], speed=2.5)
        _, _, mp_out, speed = pose_buffer.read_latest()
        assert mp_out.shape == (10, 3)
        assert speed == pytest.approx(2.5)


class TestControlBuffer:

    def test_smoke_init(self, control_buffer):
        assert control_buffer is not None

    def test_read_before_write_returns_none(self, control_buffer):
        assert control_buffer.read_latest() is None

    def test_write_read_roundtrip(self, control_buffer):
        control_buffer.write(0.5, 0.3)
        result = control_buffer.read_latest()
        assert result is not None
        header, steer, throttle = result
        assert abs(steer - 0.5) < 1e-6
        assert abs(throttle - 0.3) < 1e-6

    def test_extreme_values(self, control_buffer):
        control_buffer.write(-1.0, 1.0)
        _, steer, throttle = control_buffer.read_latest()
        assert abs(steer - (-1.0)) < 1e-6
        assert abs(throttle - 1.0) < 1e-6

    def test_zero_values(self, control_buffer):
        control_buffer.write(0.0, 0.0)
        _, steer, throttle = control_buffer.read_latest()
        assert steer == 0.0
        assert throttle == 0.0


class TestPointCloudBuffer:

    def test_smoke_init(self, pointcloud_buffer):
        assert pointcloud_buffer is not None

    def test_read_before_write_returns_none(self, pointcloud_buffer):
        assert pointcloud_buffer.read_latest() is None

    def test_write_read_roundtrip(self, pointcloud_buffer, point_cloud):
        pointcloud_buffer.write(point_cloud)
        result = pointcloud_buffer.read_latest()
        assert result is not None
        header, pts_out = result
        assert header.seq_no == 1
        np.testing.assert_allclose(pts_out, point_cloud, atol=1e-6)

    def test_empty_point_cloud(self, pointcloud_buffer):
        empty = np.empty((0, 3), dtype=np.float32)
        pointcloud_buffer.write(empty)
        _, pts_out = pointcloud_buffer.read_latest()
        assert pts_out.shape == (0, 3)

    def test_large_point_cloud_clamped(self):
        name = f"test_pc_{_uid()}"
        buf = PointCloudBuffer(slots=2, name=name, create=True, max_points=100)
        big = np.random.randn(500, 3).astype(np.float32)
        buf.write(big)
        _, pts = buf.read_latest()
        assert pts.shape == (100, 3)
        np.testing.assert_allclose(pts, big[:100], atol=1e-6)
        buf.unlink()


class TestTrackBoundaryBuffer:

    def test_smoke_init(self, boundary_buffer):
        assert boundary_buffer is not None

    def test_read_before_write_returns_none(self, boundary_buffer):
        assert boundary_buffer.read_latest() is None

    def test_write_read_roundtrip(self, boundary_buffer, track_boundaries):
        outer, inner = track_boundaries
        boundary_buffer.write(outer, inner)
        result = boundary_buffer.read_latest()
        assert result is not None
        _, outer_out, inner_out = result
        np.testing.assert_allclose(outer_out, outer, atol=1e-6)
        np.testing.assert_allclose(inner_out, inner, atol=1e-6)

    def test_empty_boundaries(self, boundary_buffer):
        outer = np.empty((0, 2), dtype=np.float32)
        inner = np.empty((0, 2), dtype=np.float32)
        boundary_buffer.write(outer, inner)
        _, o, i = boundary_buffer.read_latest()
        assert o.shape == (0, 2)
        assert i.shape == (0, 2)

    def test_asymmetric_sizes(self, boundary_buffer):
        outer = np.random.randn(50, 2).astype(np.float32)
        inner = np.random.randn(200, 2).astype(np.float32)
        boundary_buffer.write(outer, inner)
        _, o, i = boundary_buffer.read_latest()
        assert o.shape == (50, 2)
        assert i.shape == (200, 2)


# cross-process write/read

def _reader_process(name: str, n_reads: int, results: mp.Queue):
    from shared_buffer import ControlBuffer
    buf = ControlBuffer(slots=8, name=name, create=False)
    valid = 0
    none_count = 0
    for _ in range(n_reads):
        r = buf.read_latest()
        if r is not None:
            valid += 1
        else:
            none_count += 1
        time.sleep(0.0005)
    buf.close()
    results.put({"valid": valid, "none": none_count})


class TestRaceConditions:

    def test_concurrent_write_read(self):
        name = f"test_race_{_uid()}"
        buf = ControlBuffer(slots=8, name=name, create=True)

        n_writes = 200
        n_reads = 300
        results_q = mp.Queue()

        reader = mp.Process(target=_reader_process, args=(name, n_reads, results_q))
        reader.start()

        for i in range(n_writes):
            buf.write(float(i) / n_writes, float(i) / n_writes)
            time.sleep(0.001)

        reader.join(timeout=10)
        assert not reader.is_alive()

        result = results_q.get(timeout=5)
        assert result["valid"] > 0
        buf.unlink()

    def test_multiple_concurrent_readers(self):
        """One writer, two readers at once, neither should crash."""
        name = f"test_mr_{_uid()}"
        buf = ControlBuffer(slots=8, name=name, create=True)

        q1, q2 = mp.Queue(), mp.Queue()
        r1 = mp.Process(target=_reader_process, args=(name, 100, q1))
        r2 = mp.Process(target=_reader_process, args=(name, 100, q2))
        r1.start()
        r2.start()

        for i in range(100):
            buf.write(float(i) / 100, float(i) / 100)
            time.sleep(0.001)

        r1.join(timeout=10)
        r2.join(timeout=10)

        res1 = q1.get(timeout=5)
        res2 = q2.get(timeout=5)
        assert res1["valid"] > 0
        assert res2["valid"] > 0
        buf.unlink()


def _frame_reader(name: str, n_reads: int, results: mp.Queue):
    from shared_buffer import FrameBuffer
    fb = FrameBuffer(slots=4, name=name, create=False)
    valid = 0
    for _ in range(n_reads):
        r = fb.read_latest()
        if r is not None:
            valid += 1
            _, rgb, depth = r
            assert rgb.shape == (400, 640, 3)
            assert depth.shape == (400, 640)
        time.sleep(0.005)
    fb.close()
    results.put(valid)


class TestFrameBufferRace:

    def test_concurrent_frame_write_read(self):
        name = f"test_fr_{_uid()}"
        buf = FrameBuffer(slots=4, name=name, create=True)
        q = mp.Queue()

        reader = mp.Process(target=_frame_reader, args=(name, 50, q))
        reader.start()

        for i in range(30):
            rgb = np.full((400, 640, 3), i % 256, dtype=np.uint8)
            depth = np.full((400, 640), float(i), dtype=np.float32)
            buf.write(rgb, depth)
            time.sleep(0.002)

        reader.join(timeout=15)
        reads = q.get(timeout=5)
        assert reads > 0
        buf.unlink()


# read_since / decode / stamping / hardening

class TestReadSince:

    def test_returns_all_newer_in_order(self):
        rb = SharedRingBuffer(name=f"test_rs_{_uid()}", slots=8, slot_payload_size=8, create=True)
        try:
            assert rb.read_since(0) == []
            for i in range(1, 6):
                rb.write(struct.pack("<Q", i))
            got = rb.read_since(2)
            assert [h.seq_no for h, _ in got] == [3, 4, 5]
            assert [struct.unpack("<Q", p)[0] for _, p in got] == [3, 4, 5]
            assert rb.read_since(5) == []
        finally:
            rb.unlink()

    def test_after_wrap_only_ring_depth_is_left(self):
        rb = SharedRingBuffer(name=f"test_rs_{_uid()}", slots=4, slot_payload_size=8, create=True)
        try:
            for i in range(1, 11):
                rb.write(struct.pack("<Q", i))
            assert [h.seq_no for h, _ in rb.read_since(0)] == [7, 8, 9, 10]
        finally:
            rb.unlink()

    def test_typed_read_since_decodes(self, imu_buffer):
        for i in range(3):
            imu_buffer.write(np.full(9, float(i), dtype=np.float32))
        got = imu_buffer.read_since(1)
        assert [h.seq_no for h, _ in got] == [2, 3]
        np.testing.assert_array_equal(got[-1][1], np.full(9, 2.0, dtype=np.float32))


class TestReadTsStamping:

    def test_plain_read_does_not_write_shared_memory(self):
        name = f"test_st_{_uid()}"
        rb = SharedRingBuffer(name=name, slots=2, slot_payload_size=8, create=True)
        reader = SharedRingBuffer(name=name, slots=2, slot_payload_size=8, create=False)
        try:
            rb.write(b"\x01" * 8)
            before = bytes(rb._shm.buf[: rb.slot_size])
            h, _ = reader.read_latest()
            assert h.read_ts > 0                      # local read time is still reported
            assert bytes(rb._shm.buf[: rb.slot_size]) == before
            reader.read_latest(stamp=True)
            (stored,) = struct.unpack_from("<d", rb._shm.buf, 16)
            assert stored > 0
        finally:
            reader.close()
            rb.unlink()


class TestSeqlockLayout:

    def test_trailing_seq_and_alignment(self):
        rb = SharedRingBuffer(name=f"test_sl_{_uid()}", slots=3, slot_payload_size=13, create=True)
        try:
            assert rb.slot_size % 64 == 0
            for _ in range(4):
                rb.write(b"\x07" * 13)
            # slot 0 now holds seq 4: header seq and trailer agree
            (seq,) = struct.unpack_from("<Q", rb._shm.buf, 0)
            (trailer,) = struct.unpack_from("<Q", rb._shm.buf, rb._trailer_rel)
            assert seq == trailer == 4
        finally:
            rb.unlink()

    def test_mismatched_trailer_is_rejected(self):
        rb = SharedRingBuffer(name=f"test_sl_{_uid()}", slots=1, slot_payload_size=8, create=True)
        try:
            rb.write(b"\x01" * 8)
            struct.pack_into("<Q", rb._shm.buf, rb._trailer_rel, 99)   # simulate a torn publish
            assert rb.read_latest() is None
            assert rb.read_since(0) == []
        finally:
            rb.unlink()


def _spawned_writer(buf, n, start):
    for i in range(n):
        buf.write(float(start + i), 0.5)
    buf.close()


class TestCounterAcrossProcesses:

    def test_restarted_producer_continues_sequence(self):
        """start.py hands the same buffer object to a restarted producer: seq keeps increasing."""
        ctx = mp.get_context("spawn")
        buf = ControlBuffer(slots=8, name=f"test_ctr_{_uid()}", create=True)
        try:
            for start in (0, 3):
                p = ctx.Process(target=_spawned_writer, args=(buf, 3, start))
                p.start()
                p.join(timeout=30)
                assert p.exitcode == 0
            got = buf.read_since(0)
            assert [h.seq_no for h, *_ in got] == [1, 2, 3, 4, 5, 6]
            assert [s for _, s, _ in got] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        finally:
            buf.unlink()


class TestDecodeMatchesReadLatest:

    def test_every_typed_buffer(self, tmp_path):
        from control.types import VehicleState
        from shared_buffer import (
            IMUBuffer, LidarBuffer, PoseBuffer, SegmentationBuffer, StateBuffer, TrackBoundaryBuffer,
        )

        u = _uid()
        cases = [
            (FrameBuffer(slots=2, name=f"d_f_{u}"),
             (np.zeros((4, 5, 3), np.uint8) + 7, np.ones((4, 5), np.float32))),
            (IMUBuffer(slots=2, name=f"d_i_{u}"), (np.arange(9, dtype=np.float32),)),
            (SegmentationBuffer(slots=2, name=f"d_s_{u}"), (np.eye(4, dtype=np.int32),)),
            (PoseBuffer(slots=2, name=f"d_p_{u}"), (np.eye(4), np.ones((3, 3)), 1.5)),
            (ControlBuffer(slots=2, name=f"d_c_{u}"), (0.25, 0.75)),
            (PointCloudBuffer(slots=2, name=f"d_pc_{u}", max_points=10), (np.ones((5, 3)),)),
            (TrackBoundaryBuffer(slots=2, name=f"d_b_{u}"), (np.ones((3, 2)), np.zeros((2, 2)))),
            (StateBuffer(slots=2, name=f"d_st_{u}"), (VehicleState(t=1.0, x=2.0, y=3.0, yaw=0.5, v=1.0, yaw_rate=0.1),)),
            (LidarBuffer(slots=2, name=f"d_l_{u}"), (np.ones(1080),)),
        ]
        try:
            for buf, args in cases:
                buf.write(*args)
                header, payload = buf.ring.read_latest()
                decoded = buf.decode(payload)
                via_read = buf.read_latest()[1:]
                if not isinstance(decoded, tuple):
                    decoded = (decoded,)
                assert len(decoded) == len(via_read), type(buf).__name__
                for a, b in zip(decoded, via_read):
                    if isinstance(a, np.ndarray):
                        np.testing.assert_array_equal(a, b)
                    else:
                        assert a == b
            seg = cases[2][0].read_latest()[1]
            assert seg.dtype == np.uint8
            assert cases[5][0].ring.dtype_tag == DTypeTag.POINTCLOUD
            assert cases[6][0].ring.dtype_tag == DTypeTag.BOUNDARY
        finally:
            for buf, _ in cases:
                buf.unlink()


def test_buffer_name_helper():
    from shared_buffer import buffer_name

    assert buffer_name(None, "frame_buf") == "frame_buf"
    assert buffer_name("", "frame_buf") == "frame_buf"
    assert buffer_name("run42", "frame_buf") == "run42_frame_buf"
    assert buffer_name("run42_", "frame_buf") == "run42_frame_buf"
