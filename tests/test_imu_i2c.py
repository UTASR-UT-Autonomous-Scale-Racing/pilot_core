"""IMUWorker's I2C path against a fake bus speaking the Arduino's 60-byte frame."""

from __future__ import annotations

import struct
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import IMUBuffer
import workers.imu_worker as imu_mod
from workers.imu_worker import I2C_FRAME_SIZE, IMUWorker, parse_i2c_frame


def _frame(quat=(1.0, 0.0, 0.0, 0.0), acc_g=(0.0, 0.0, 1.0), throttle=1500, steer=1461):
    # same order as firmware/firmware/jetson_i2c.cpp jetson_request_event
    return struct.pack(
        "<4f3f3f3fIHH", *quat, *acc_g, 0.1, 0.2, 0.3, 10.0, 20.0, 30.0, 123456, throttle, steer,
    )


class _FakeMsg:
    def __init__(self, addr, n):
        self.addr, self.len, self.data = addr, n, b""

    def __bytes__(self):
        return self.data


class _FakeSMBus2:
    """Stands in for the smbus2 module: SMBus + i2c_msg.read."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.reads = []
        fake = self

        class _Bus:
            def __init__(self, bus):
                pass

            def i2c_rdwr(self, msg):
                fake.reads.append((msg.addr, msg.len))
                item = fake.frames.pop(0)
                if isinstance(item, Exception):
                    raise item
                msg.data = item

            def read_i2c_block_data(self, *a):
                raise AssertionError("SMBus block reads are limited to 32 bytes")

            def close(self):
                pass

        class _Msg:
            @staticmethod
            def read(addr, n):
                return _FakeMsg(addr, n)

        self.SMBus = _Bus
        self.i2c_msg = _Msg


def _worker(monkeypatch, frames):
    fake = _FakeSMBus2(frames)
    monkeypatch.setattr(imu_mod, "smbus2", fake)
    ib = IMUBuffer(slots=8, name=f"imu_i2c_{uuid.uuid4().hex[:8]}", create=True)
    w = IMUWorker(name="imu", output_buffer=ib, i2c_address=0x08, i2c_bus=1, poll_hz=1000)
    w._logger = MagicMock()
    w.setup()
    return w, ib, fake


def test_frame_is_60_bytes():
    assert I2C_FRAME_SIZE == 60 == len(_frame())


def test_parse_converts_g_to_mps2():
    vec = parse_i2c_frame(_frame(acc_g=(0.5, -0.25, 1.0)))
    np.testing.assert_allclose(vec[:4], [1, 0, 0, 0])
    np.testing.assert_allclose(vec[4:7], [0.5 * 9.80665, -0.25 * 9.80665, 9.80665], rtol=1e-6)
    assert vec[7] == 1500 and vec[8] == 1461


def test_parse_rejects_handshake_padding():
    assert parse_i2c_frame(b"\x01\x02\x03\x04" + b"\xff" * 56) is None
    assert parse_i2c_frame(b"\x00" * 10) is None


def test_worker_reads_60_bytes_and_publishes(monkeypatch):
    w, ib, fake = _worker(monkeypatch, [_frame(acc_g=(0.0, 0.0, 1.0), throttle=1600)])
    try:
        w.step()
        assert fake.reads == [(0x08, 60)]
        header, vec = ib.read_latest()
        assert vec.shape == (9,)
        assert vec[6] == pytest.approx(9.80665, rel=1e-6)
        assert vec[7] == 1600
    finally:
        w.teardown()
        ib.unlink()


def test_worker_survives_oserror(monkeypatch):
    w, ib, fake = _worker(monkeypatch, [OSError(121, "Remote I/O error"), _frame()])
    try:
        w.step()                      # error: logged, nothing published, backs off
        assert ib.read_latest() is None
        assert w._errors == 1
        w._next_poll = 0.0            # skip the back-off wait
        w.step()
        assert ib.read_latest() is not None
        assert w._errors == 0
    finally:
        w.teardown()
        ib.unlink()


def test_raw_parse_converts_units():
    from workers.imu_worker import parse_i2c_raw

    raw = parse_i2c_raw(_frame(acc_g=(0.5, -0.25, 1.0)))     # gyro in _frame: 0.1, 0.2, 0.3 deg/s
    np.testing.assert_allclose(raw[:3], [0.5 * 9.80665, -0.25 * 9.80665, 9.80665], rtol=1e-6)
    np.testing.assert_allclose(raw[3:], np.radians([0.1, 0.2, 0.3]), rtol=1e-6)
    assert parse_i2c_raw(b"\x01\x02\x03\x04" + b"\xff" * 56) is None


def test_worker_writes_each_new_rear_imu_sample_once(monkeypatch):
    from shared_buffer import RawIMUBuffer

    rb = RawIMUBuffer(slots=16, name=f"rear_{uuid.uuid4().hex[:8]}", create=True)
    a, b = _frame(acc_g=(0.0, 0.0, 1.0)), _frame(acc_g=(0.1, 0.0, 1.0))
    fake = _FakeSMBus2([a, a, b, b, b])          # the Jetson polls faster than new samples come
    monkeypatch.setattr(imu_mod, "smbus2", fake)
    ib = IMUBuffer(slots=8, name=f"imu_i2c_{uuid.uuid4().hex[:8]}", create=True)
    w = IMUWorker(name="imu", output_buffer=ib, raw_output_buffer=rb, i2c_address=0x08, i2c_bus=1, poll_hz=1000)
    w._logger = MagicMock()
    w.setup()
    try:
        for _ in range(5):
            w._next_poll = 0.0
            w.step()
        recs = rb.read_since(0)
        assert len(recs) == 2
        assert recs[1][1][0] == pytest.approx(0.1 * 9.80665, rel=1e-6)
        assert len(ib.read_since(0)) == 5        # imu_buf keeps every poll, as before
    finally:
        w.teardown()
        ib.unlink()
        rb.unlink()
