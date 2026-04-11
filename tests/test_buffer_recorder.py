"""
Tests for BufferRecorderWorker — per-buffer binary recorder.

Covers:
    - File header writing & reading
    - Record writing & reading (round-trip)
    - sample_k decimation
    - top_n record cap + wraparound
    - Integration with each typed buffer (Frame, IMU, Seg, Pose, Control, PC, Boundary)
    - Edge cases: empty reads, back-to-back writes, zero payload
"""

from __future__ import annotations

import os
import struct
import time
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
from shared_buffer.schema import DTypeTag, SCHEMA_VERSION
from workers.buffer_recorder_worker import (
    FILE_HEADER_SIZE,
    FILE_MAGIC,
    RECORD_HEADER_SIZE,
    BufferRecorderWorker,
)


def _uid():
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# Helpers: build a recorder wired to a real shared-memory buffer
# ---------------------------------------------------------------------------

def _make_recorder(
    buf,
    buf_name: str,
    tmp_path,
    sample_k: int = 1,
    top_n: int = 0,
    dtype_tag: int = 0,
) -> BufferRecorderWorker:
    w = BufferRecorderWorker(
        name=f"rec_{buf_name}",
        buffer=buf,
        buffer_name=buf_name,
        output_dir=str(tmp_path),
        sample_k=sample_k,
        top_n=top_n,
        dtype_tag=dtype_tag,
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


# ---------------------------------------------------------------------------
# File header
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Record round-trip
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Decimation (sample_k)
# ---------------------------------------------------------------------------

class TestSampleK:

    def test_every_other(self, tmp_path):
        """sample_k=2 → only even seq_no get recorded."""
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


# ---------------------------------------------------------------------------
# Top-N cap
# ---------------------------------------------------------------------------

class TestTopN:

    def test_top_n_limits_records(self, tmp_path):
        """After top_n records the file pointer wraps — total file size stays bounded."""
        name = f"tn_{_uid()}"
        buf = IMUBuffer(slots=8, name=name, create=True)
        try:
            w = _make_recorder(buf, "imu", tmp_path, top_n=3)

            # Write 6 records — should wrap once
            for _ in range(6):
                buf.write(np.zeros(9, dtype=np.float32))
                w.step()

            # Internal counter should have been reset
            assert w._records_written == 3
            w.teardown()
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
        finally:
            buf.unlink()


# ---------------------------------------------------------------------------
# Typed buffer integration — one test per buffer type
# ---------------------------------------------------------------------------

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
            # Payload should contain 16 bytes meta + rgb + depth
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
            # 8 bytes meta + 400*640*4 float32
            assert len(payload) == 8 + 400 * 640 * 4
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


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_no_data_available(self, tmp_path):
        """step() with nothing written to buffer → no record."""
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
        """Reading the same seq_no again should not produce a second record."""
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
        """read_record returns None at end of file."""
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


# ---------------------------------------------------------------------------
# Integration: verify start.py recorder spawning logic
# ---------------------------------------------------------------------------

class TestStartPyRecorderConfig:
    """Validate that the config structure produces the right number of recorders."""

    def test_recording_config_keys(self):
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        rec_cfg = cfg.get("recording_config", {})
        assert "sample_k" in rec_cfg
        assert "top_n" in rec_cfg
        # Per-buffer override keys
        for key in ("frame", "imu", "segmentation", "pose", "control"):
            assert key in rec_cfg, f"Missing per-buffer config for {key}"

    def test_per_buffer_overrides(self):
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        rec_cfg = cfg["recording_config"]
        # Frame should sample less frequently than IMU
        assert rec_cfg["frame"]["sample_k"] >= rec_cfg["imu"]["sample_k"]
