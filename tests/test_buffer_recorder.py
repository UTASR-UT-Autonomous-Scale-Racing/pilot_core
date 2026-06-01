"""BufferRecorderWorker file format, sample_k/top_n, and a round trip per buffer type."""

from __future__ import annotations

import os
import struct
import uuid
from unittest.mock import MagicMock

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
from shared_buffer.schema import SCHEMA_VERSION
from workers.buffer_recorder_worker import (
    FILE_HEADER_SIZE,
    FILE_MAGIC,
    BufferRecorderWorker,
)


def _uid():
    return uuid.uuid4().hex[:8]


# recorder wired to a real shm buffer

def _make_recorder(
    buf,
    buf_name: str,
    tmp_path,
    sample_k: int = 1,
    top_n: int = 0,
    **extra,
) -> BufferRecorderWorker:
    w = BufferRecorderWorker(
        name=f"rec_{buf_name}",
        buffer=buf,
        buffer_name=buf_name,
        output_dir=str(tmp_path),
        sample_k=sample_k,
        top_n=top_n,
        **extra,
    )
    w._logger = MagicMock()
    w.setup()
    return w


def _bin_files(tmp_path) -> list[str]:
    return sorted(
        os.path.join(tmp_path, f)
        for f in os.listdir(tmp_path)
        if f.endswith(".bin")
    )


class TestFileHeader:

    def test_header_size_is_64(self):
        assert FILE_HEADER_SIZE == 64

    def test_header_written_on_setup(self, tmp_path):
        name = f"fh_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            w.teardown()
            files = _bin_files(tmp_path)
            assert len(files) == 1
            with open(files[0], "rb") as f:
                hdr = BufferRecorderWorker.read_file_header(f)
            assert hdr["magic"] == FILE_MAGIC
            assert hdr["schema_version"] == SCHEMA_VERSION
            assert hdr["sample_k"] == 1
            assert hdr["top_n"] == 0
            assert hdr["session_ts"] > 0
        finally:
            buf.unlink()

    def test_header_reflects_custom_params(self, tmp_path):
        name = f"fh2_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, sample_k=5, top_n=100)
            w.teardown()
            with open(_bin_files(tmp_path)[0], "rb") as f:
                hdr = BufferRecorderWorker.read_file_header(f)
            assert hdr["sample_k"] == 5
            assert hdr["top_n"] == 100
        finally:
            buf.unlink()


# round trip

class TestRecordRoundTrip:

    def test_write_and_read_back_imu(self, tmp_path):
        name = f"rr_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)

            vec = np.arange(9, dtype=np.float32)
            buf.write(vec)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)

            assert rec is not None
            hdr, payload = rec
            assert hdr["seq_no"] == 1
            assert hdr["payload_size"] == 9 * 4
            recovered = np.frombuffer(payload, dtype=np.float32)
            np.testing.assert_array_equal(recovered, vec)
        finally:
            buf.unlink()

    def test_multiple_records(self, tmp_path):
        name = f"mr_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)

            for i in range(5):
                buf.write(np.full(9, float(i), dtype=np.float32))
                w.step()
            w.teardown()

            records = []
            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                while True:
                    rec = BufferRecorderWorker.read_record(f)
                    if rec is None:
                        break
                    records.append(rec)

            assert len(records) == 5
            for i, (hdr, payload) in enumerate(records):
                assert hdr["seq_no"] == i + 1
        finally:
            buf.unlink()


# sample_k decimation

class TestSampleK:

    def test_every_other(self, tmp_path):
        # sample_k=2: only even seq_no should land in the file
        name = f"sk_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, sample_k=2)

            for _ in range(10):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()
            w.teardown()

            records = []
            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                while True:
                    rec = BufferRecorderWorker.read_record(f)
                    if rec is None:
                        break
                    records.append(rec)

            assert len(records) == 5
            for hdr, _ in records:
                assert hdr["seq_no"] % 2 == 0
        finally:
            buf.unlink()

    def test_sample_k_3(self, tmp_path):
        name = f"sk3_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, sample_k=3)

            for _ in range(9):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()
            w.teardown()

            records = []
            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                while True:
                    rec = BufferRecorderWorker.read_record(f)
                    if rec is None:
                        break
                    records.append(rec)

            assert len(records) == 3
            for hdr, _ in records:
                assert hdr["seq_no"] % 3 == 0
        finally:
            buf.unlink()


def _read_all(path):
    with open(path, "rb") as f:
        hdr = BufferRecorderWorker.read_file_header(f)
        recs = []
        while True:
            rec = BufferRecorderWorker.read_record(f)
            if rec is None:
                break
            recs.append(rec)
        # a valid file ends exactly at the last record
        assert f.read() == b""
    return hdr, recs


class TestTopN:

    def test_top_n_rotates_files(self, tmp_path):
        """top_n caps records per file; the next records go to a new file."""
        name = f"tn_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, top_n=3, keep_files=0)
            for _ in range(7):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()
            w.teardown()
            files = _bin_files(tmp_path)
            assert len(files) == 3
            seqs = []
            for i, path in enumerate(files):
                hdr, recs = _read_all(path)
                assert hdr["part"] == i
                seqs += [h["seq_no"] for h, _ in recs]
            assert seqs == list(range(1, 8))
            assert w._records_written == 7
        finally:
            buf.unlink()

    def test_keep_files_deletes_oldest_parts(self, tmp_path):
        name = f"tk_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, top_n=2, keep_files=2)
            for _ in range(9):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()
            w.teardown()
            files = _bin_files(tmp_path)
            assert len(files) == 2
            seqs = [h["seq_no"] for p in files for h, _ in _read_all(p)[1]]
            assert seqs == [7, 8, 9]    # parts 3 and 4, oldest parts deleted
        finally:
            buf.unlink()

    def test_stop_policy(self, tmp_path):
        name = f"ts_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, top_n=3, on_full="stop")
            for _ in range(6):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()
            w.teardown()
            files = _bin_files(tmp_path)
            assert len(files) == 1
            assert [h["seq_no"] for h, _ in _read_all(files[0])[1]] == [1, 2, 3]
        finally:
            buf.unlink()

    def test_variable_size_payloads_stay_valid(self, tmp_path):
        """Regression: in-place wrap-around corrupted files with variable-size payloads."""
        name = f"tv_{_uid()}"
        buf = PointCloudBuffer(slots=4, name=name, create=True, max_points=500)
        rng = np.random.default_rng(0)
        written = []
        try:
            w = _make_recorder(buf, "pointcloud", tmp_path, top_n=4, keep_files=0)
            for _ in range(15):
                pts = rng.normal(size=(int(rng.integers(0, 500)), 3)).astype(np.float32)
                written.append(pts)
                buf.write(pts)
                w.step()
            w.teardown()
            decoded = [PointCloudBuffer.decode(p) for f in _bin_files(tmp_path) for _, p in _read_all(f)[1]]
            assert len(decoded) == 15
            for a, b in zip(decoded, written):
                np.testing.assert_array_equal(a, b)
        finally:
            buf.unlink()

    def test_top_n_zero_means_unlimited(self, tmp_path):
        name = f"tn0_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, top_n=0)

            for _ in range(20):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()

            assert w._records_written == 20
            w.teardown()
            assert len(_bin_files(tmp_path)) == 1
        finally:
            buf.unlink()


class TestNoDrops:

    def test_burst_between_polls_is_recorded(self, tmp_path):
        """Several writes between two steps: all of them land in the file."""
        name = f"nd_{_uid()}"
        buf = IMUBuffer(slots=16, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            for i in range(10):
                buf.write(np.full(9, float(i), dtype=np.float32))
            w.step()
            for i in range(10, 13):
                buf.write(np.full(9, float(i), dtype=np.float32))
            w.step()
            w.teardown()
            _, recs = _read_all(_bin_files(tmp_path)[0])
            assert [h["seq_no"] for h, _ in recs] == list(range(1, 14))
            assert [IMUBuffer.decode(p)[0] for _, p in recs] == [float(i) for i in range(13)]
            assert all(h["read_ts"] >= h["produced_ts"] for h, _ in recs)
            assert w.streams["imu"].dropped == 0
        finally:
            buf.unlink()

    def test_sample_k_counts_received_samples(self, tmp_path):
        name = f"nk_{_uid()}"
        buf = IMUBuffer(slots=16, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, sample_k=4)
            for _ in range(12):
                buf.write(np.zeros(9, dtype=np.float32))
            w.step()
            w.teardown()
            _, recs = _read_all(_bin_files(tmp_path)[0])
            assert [h["seq_no"] for h, _ in recs] == [4, 8, 12]
        finally:
            buf.unlink()

    def test_ring_overrun_is_counted_as_dropped(self, tmp_path):
        name = f"no_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            buf.write(np.zeros(9, dtype=np.float32))
            w.step()
            for _ in range(10):          # laps the 4-slot ring
                buf.write(np.zeros(9, dtype=np.float32))
            w.step()
            w.teardown()
            _, recs = _read_all(_bin_files(tmp_path)[0])
            assert [h["seq_no"] for h, _ in recs] == [1, 8, 9, 10, 11]
            assert w.streams["imu"].dropped == 6
        finally:
            buf.unlink()


class TestMultiBuffer:

    def test_one_recorder_many_buffers(self, tmp_path):
        from control.types import VehicleState
        from shared_buffer import LidarBuffer, StateBuffer

        u = _uid()
        imu = IMUBuffer(slots=8, name=f"mi_{u}", create=True)
        state = StateBuffer(slots=8, name=f"ms_{u}", create=True)
        lidar = LidarBuffer(slots=4, name=f"ml_{u}", create=True)
        try:
            w = BufferRecorderWorker(
                name="recorder",
                output_dir=str(tmp_path),
                sample_k=1,
                buffers={
                    "imu": {"buffer": imu, "sample_k": 2},
                    "state": state,
                    "lidar": {"buffer": lidar, "top_n": 2, "keep_files": 0},
                    "missing": None,             # buffers that don't exist in this config
                },
            )
            w._logger = MagicMock()
            w.setup()
            for i in range(4):
                imu.write(np.full(9, float(i), dtype=np.float32))
                state.write(VehicleState(t=float(i), x=1.0, y=2.0, yaw=0.1, v=0.5, yaw_rate=0.0))
                lidar.write(np.full(1080, float(i), dtype=np.float32))
                w.step()
            w.teardown()

            files = _bin_files(tmp_path)
            by_name = {}
            for p in files:
                hdr, recs = _read_all(p)
                by_name.setdefault(hdr["buffer_name"], []).extend(recs)
            assert sorted(by_name) == ["imu", "lidar", "state"]
            assert len(by_name["imu"]) == 2
            assert len(by_name["lidar"]) == 4
            assert len([p for p in files if os.path.basename(p).startswith("lidar_")]) == 2
            s = [StateBuffer.decode(p) for _, p in by_name["state"]]
            assert [v.t for v in s] == [0.0, 1.0, 2.0, 3.0]
        finally:
            for b in (imu, state, lidar):
                b.unlink()


class TestFlushAndValidation:

    def test_periodic_flush(self, tmp_path):
        name = f"fl_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, flush_interval=0.0)
            buf.write(np.zeros(9, dtype=np.float32))
            w.step()
            # flushed without teardown
            assert os.path.getsize(_bin_files(tmp_path)[0]) == FILE_HEADER_SIZE + 32 + 36
            w.teardown()
        finally:
            buf.unlink()

    def test_bad_magic_rejected(self, tmp_path):
        import io

        import pytest

        from workers.buffer_recorder_worker import RecordingFormatError

        with pytest.raises(RecordingFormatError):
            BufferRecorderWorker.read_file_header(io.BytesIO(b"NOTAREC!" + b"\x00" * 56))
        with pytest.raises(RecordingFormatError):
            BufferRecorderWorker.read_file_header(io.BytesIO(b"short"))

    def test_unsupported_schema_rejected(self, tmp_path):
        import io

        import pytest

        from workers.buffer_recorder_worker import RecordingFormatError

        raw = FILE_MAGIC + struct.pack("<H", 999) + b"\x00" * 54
        with pytest.raises(RecordingFormatError):
            BufferRecorderWorker.read_file_header(io.BytesIO(raw))

    def test_dtype_tag_from_buffer(self, tmp_path):
        from shared_buffer.schema import DTypeTag

        name = f"dt_{_uid()}"
        buf = PointCloudBuffer(slots=2, name=name, create=True, max_points=10)
        try:
            w = _make_recorder(buf, "pointcloud", tmp_path)
            w.teardown()
            hdr, _ = _read_all(_bin_files(tmp_path)[0])
            assert hdr["dtype_tag"] == DTypeTag.POINTCLOUD
            assert hdr["buffer_name"] == "pointcloud"
        finally:
            buf.unlink()


# one round trip per typed buffer

class TestFrameRecorder:

    def test_frame_round_trip(self, tmp_path):
        name = f"fr_{_uid()}"
        buf = FrameBuffer(slots=2, name=name, create=True)
        try:
            w = _make_recorder(buf, "frame", tmp_path)

            rgb = np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)
            depth = np.random.uniform(0.2, 5.0, (400, 640)).astype(np.float32)
            buf.write(rgb, depth)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)

            assert rec is not None
            hdr, payload = rec
            assert hdr["seq_no"] == 1
            assert hdr["payload_size"] > 0
            # 16 bytes meta + rgb + depth
            expected_min = 16 + 400 * 640 * 3 + 400 * 640 * 4
            assert len(payload) == expected_min
        finally:
            buf.unlink()


class TestIMURecorder:

    def test_imu_round_trip(self, tmp_path):
        name = f"ir_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            vec = np.array([1, 2, 3, 4, 5, 6, 7, 0.5, 0.3], dtype=np.float32)
            buf.write(vec)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)
            _, payload = rec
            recovered = np.frombuffer(payload, dtype=np.float32)
            np.testing.assert_array_almost_equal(recovered, vec)
        finally:
            buf.unlink()


class TestSegmentationRecorder:

    def test_segmentation_round_trip(self, tmp_path):
        name = f"sr_{_uid()}"
        buf = SegmentationBuffer(slots=2, name=name, create=True)
        try:
            w = _make_recorder(buf, "segmentation", tmp_path)
            mask = np.random.randint(0, 3, (400, 640), dtype=np.int32)
            buf.write(mask)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)
            assert rec is not None
            _, payload = rec
            # 8 bytes meta + 400x640 uint8 mask
            assert len(payload) == 8 + 400 * 640
            np.testing.assert_array_equal(SegmentationBuffer.decode(payload), mask)
        finally:
            buf.unlink()


class TestPoseRecorder:

    def test_pose_round_trip(self, tmp_path):
        name = f"pr_{_uid()}"
        buf = PoseBuffer(slots=2, name=name, create=True)
        try:
            w = _make_recorder(buf, "pose", tmp_path)
            pose = np.eye(4, dtype=np.float32)
            buf.write(pose)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)
            assert rec is not None
            _, payload = rec
            # 64 bytes pose + 4 bytes count (0) + 4 bytes speed
            assert len(payload) == 72
        finally:
            buf.unlink()


class TestControlRecorder:

    def test_control_round_trip(self, tmp_path):
        name = f"cr_{_uid()}"
        buf = ControlBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "control", tmp_path)
            buf.write(0.42, 0.73)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)
            _, payload = rec
            s, t = struct.unpack("<ff", payload[:8])
            assert abs(s - 0.42) < 1e-5
            assert abs(t - 0.73) < 1e-5
        finally:
            buf.unlink()


class TestPointCloudRecorder:

    def test_pc_round_trip(self, tmp_path):
        name = f"pcr_{_uid()}"
        buf = PointCloudBuffer(slots=2, name=name, create=True)
        try:
            w = _make_recorder(buf, "pointcloud", tmp_path)
            pts = np.random.randn(100, 3).astype(np.float32)
            buf.write(pts)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)
            assert rec is not None
            _, payload = rec
            # 4 bytes count + 100*3*4 float32
            assert len(payload) == 4 + 100 * 3 * 4
        finally:
            buf.unlink()


class TestBoundaryRecorder:

    def test_boundary_round_trip(self, tmp_path):
        name = f"br_{_uid()}"
        buf = TrackBoundaryBuffer(slots=2, name=name, create=True)
        try:
            w = _make_recorder(buf, "boundary", tmp_path)
            outer = np.random.randn(50, 2).astype(np.float32)
            inner = np.random.randn(40, 2).astype(np.float32)
            buf.write(outer, inner)
            w.step()
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                rec = BufferRecorderWorker.read_record(f)
            assert rec is not None
            _, payload = rec
            # 4 + 50*2*4 + 4 + 40*2*4
            expected = 4 + 50 * 8 + 4 + 40 * 8
            assert len(payload) == expected
        finally:
            buf.unlink()


class TestEdgeCases:

    def test_no_data_available(self, tmp_path):
        name = f"nd_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            w.step()
            assert w._records_written == 0
            w.teardown()
        finally:
            buf.unlink()

    def test_duplicate_seq_not_recorded_twice(self, tmp_path):
        name = f"dup_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            buf.write(np.zeros(9, dtype=np.float32))
            w.step()
            w.step()  # same seq_no
            assert w._records_written == 1
            w.teardown()
        finally:
            buf.unlink()

    def test_read_record_at_eof(self, tmp_path):
        name = f"eof_{_uid()}"
        buf = IMUBuffer(slots=4, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path)
            w.teardown()

            with open(_bin_files(tmp_path)[0], "rb") as f:
                BufferRecorderWorker.read_file_header(f)
                assert BufferRecorderWorker.read_record(f) is None
        finally:
            buf.unlink()


class TestStartPyRecorderConfig:
    """recording_config in the shipped configs only uses keys start.py knows."""

    BUFFERS = {"frame", "imu", "segmentation", "pose", "control", "actuation", "pointcloud", "keyframes", "boundary",
               "state", "lidar"}
    GLOBAL_KEYS = {"sample_k", "top_n", "on_full", "keep_files", "flush_interval"}
    PER_BUFFER_KEYS = {"sample_k", "top_n", "on_full", "keep_files"}

    @pytest.mark.parametrize("name", ["config.yaml", "config_cpu.yaml", "config_dev_local.yaml", "config_autodrive.yaml"])
    def test_recording_config_keys(self, name, repo_root):
        import os
        import yaml
        with open(os.path.join(repo_root, name)) as f:
            rec_cfg = yaml.safe_load(f).get("recording_config") or {}
        for key, value in rec_cfg.items():
            if isinstance(value, dict):
                assert key in self.BUFFERS, f"{name}: unknown buffer {key}"
                assert set(value) <= self.PER_BUFFER_KEYS, f"{name}: {key} has {set(value)}"
            else:
                assert key in self.GLOBAL_KEYS, f"{name}: unknown key {key}"
