"""scripts/read_recording.py on recordings made by the real BufferRecorderWorker."""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.read_recording import iter_records, list_files, load_buffer, main, summarize
from shared_buffer import ControlBuffer, IMUBuffer, PointCloudBuffer
from workers.buffer_recorder_worker import BufferRecorderWorker


def _uid():
    return uuid.uuid4().hex[:8]


@pytest.fixture
def recording(tmp_path):
    u = _uid()
    imu = IMUBuffer(slots=16, name=f"rr_i_{u}")
    ctrl = ControlBuffer(slots=16, name=f"rr_c_{u}")
    pc = PointCloudBuffer(slots=4, name=f"rr_p_{u}", max_points=64)
    try:
        w = BufferRecorderWorker(
            name="recorder", output_dir=str(tmp_path),
            buffers={"imu": imu, "control": {"buffer": ctrl, "sample_k": 2},
                     "pointcloud": {"buffer": pc, "top_n": 3, "keep_files": 0}},
        )
        w._logger = MagicMock()
        w.setup()
        t0 = 1000.0
        for i in range(10):
            imu.write(np.full(9, float(i), np.float32), ts=t0 + i * 0.01)
            ctrl.write(i / 10, 0.5, ts=t0 + i * 0.01)
            pc.write(np.full((i, 3), float(i), np.float32), ts=t0 + i * 0.01)
            w.step()
        w.teardown()
        yield str(tmp_path)
    finally:
        for b in (imu, ctrl, pc):
            b.unlink()


def test_list_files_orders_parts(recording):
    files = list_files(recording, buffer="pointcloud")
    assert [f.part for f in files] == [0, 1, 2, 3]
    assert {f.buffer for f in list_files(recording)} == {"imu", "control", "pointcloud"}


def test_iter_records_decodes(recording):
    imu = list(iter_records(recording, buffer="imu"))
    assert [r.seq_no for r in imu] == list(range(1, 11))
    np.testing.assert_array_equal(imu[3].value, np.full(9, 3.0, np.float32))

    ctrl = load_buffer(recording, "control")
    assert [r.seq_no for r in ctrl] == [2, 4, 6, 8, 10]
    steering, throttle = ctrl[0].value
    assert steering == pytest.approx(0.1) and throttle == pytest.approx(0.5)

    pcs = list(iter_records(recording, buffer="pointcloud"))
    assert [r.value.shape for r in pcs] == [(i, 3) for i in range(10)]


def test_summary(recording):
    s = summarize(recording)
    assert s["imu"]["records"] == 10
    assert s["imu"]["duration_s"] == pytest.approx(0.09)
    assert s["imu"]["rate_hz"] == pytest.approx(100.0)
    assert s["control"]["records"] == 5 and s["control"]["sample_k"] == 2
    assert s["pointcloud"]["files"] == 4
    assert s["pointcloud"]["dtype"] == "POINTCLOUD"
    assert all(v["truncated_bytes"] == 0 and v["out_of_order"] == 0 for v in s.values())


def test_truncated_tail_and_foreign_files(recording, capsys):
    path = list_files(recording, buffer="imu")[0].path
    with open(path, "ab") as f:
        f.write(b"\x01\x02\x03")               # partial record, e.g. after a crash
    with open(os.path.join(recording, "junk_1_0000.bin"), "wb") as f:
        f.write(b"not a recording" * 10)
    s = summarize(recording)
    assert s["imu"]["records"] == 10
    assert s["imu"]["truncated_bytes"] == 3
    assert "junk" not in s
    assert "skipping" in capsys.readouterr().err


def test_cli(recording, capsys):
    assert main([recording, "--dump", "2"]) == 0
    out = capsys.readouterr().out
    assert "imu" in out and "pointcloud" in out and "seq=2" in out
