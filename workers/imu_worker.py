"""
Arduino IMU over I2C -> IMUBuffer vectors
[qw, qx, qy, qz, ax, ay, az, throttle_raw, steering_raw], accel in m/s^2,
stamped with host time.monotonic() at the read. The OAK-D IMU is streamed by CameraWorker.

With a raw_output_buffer (RawIMUBuffer), also the 6-axis sample [ax, ay, az (m/s^2),
gx, gy, gz (rad/s)] in the ICM-20948's axes, for the vehicle-state EKF (rear_imu in the
config): the firmware sends accel in g and gyro in deg/s, bias-corrected at its start-up
(it also forces accel to (0, 0, 1 g) at rest, so the mounting can't be seen at rest:
scripts/check_rear_imu.py finds it from motion). The Jetson polls the latest sample, so a
poll can return the one it already had: those repeats are not written.
"""

from __future__ import annotations

import struct
import time
from typing import Any

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import IMUBuffer

try:
    import smbus2
except ImportError:
    smbus2 = None

G = 9.80665  # m/s^2 per g

# Arduino frame (firmware jetson_i2c.cpp): quat wxyz, accel xyz (g), gyro xyz (deg/s),
# mag xyz (float32 each), micros() of the last DMP quaternion update (uint32, wraps every
# ~71.6 min), throttle, steering raw (uint16)
I2C_FRAME = struct.Struct("<4f3f3f3fIHH")
I2C_FRAME_SIZE = I2C_FRAME.size  # 60


def parse_i2c_frame(raw: bytes) -> np.ndarray | None:
    """60-byte Arduino frame -> the 9-float IMUBuffer vector
    [qw, qx, qy, qz, ax, ay, az, throttle_raw, steering_raw], accel in m/s^2.

    None if the frame is short, has non-finite floats or a quaternion longer
    than 1.5 (the handshake reply padded with 0xFF, or a garbled read). An
    all-zero quaternion (IMU fusion not ready yet) is passed through.
    """
    if len(raw) < I2C_FRAME_SIZE:
        return None
    v = I2C_FRAME.unpack_from(raw)
    quat = np.array(v[0:4], dtype=np.float64)
    accel = np.array(v[4:7], dtype=np.float64) * G
    throttle, steering = v[14], v[15]
    if not (np.all(np.isfinite(quat)) and np.all(np.isfinite(accel))):
        return None
    if float(np.linalg.norm(quat)) > 1.5:
        return None
    return np.array(
        [*quat, *accel, float(throttle), float(steering)], dtype=np.float32
    )


def parse_i2c_raw(raw: bytes) -> np.ndarray | None:
    """60-byte Arduino frame -> (6,) [ax, ay, az (m/s^2), gx, gy, gz (rad/s)], sensor axes.
    None if short, non-finite or out of the ICM-20948's ranges (16 g, 2000 deg/s)."""
    if len(raw) < I2C_FRAME_SIZE:
        return None
    v = I2C_FRAME.unpack_from(raw)
    acc_g, gyro_dps = np.array(v[4:7], dtype=np.float64), np.array(v[7:10], dtype=np.float64)
    if not (np.all(np.isfinite(acc_g)) and np.all(np.isfinite(gyro_dps))):
        return None
    if np.abs(acc_g).max() > 16.5 or np.abs(gyro_dps).max() > 2050.0:
        return None
    return np.concatenate([acc_g * G, np.radians(gyro_dps)]).astype(np.float32)


class IMUWorker(BaseWorker):
    """IMU worker. I2C errors are retried with back-off (up to 1 s), never fatal.
    Without a full I2C address/bus it idles: no fake data when a real sensor is
    expected.

    Config kwargs:
      output_buffer       IMUBuffer
      raw_output_buffer   RawIMUBuffer for the 6-axis sample (optional, rear_imu)
      i2c_address     Arduino I2C address
      i2c_bus         I2C bus number
      poll_hz         I2C polling rate, default 100
    """

    def __init__(self, name: str = "imu", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._imu_buf: IMUBuffer | None = None
        self._bus = None
        self._dummy = False
        self._errors = 0
        self._period = 0.01
        self._next_poll = 0.0
        self._raw_buf = None
        self._last_raw: bytes | None = None
        self._raw_written = 0
        self._raw_repeats = 0
        self._stats_t = 0.0

    def setup(self) -> None:
        self._imu_buf = self.config["output_buffer"]
        self._raw_buf = self.config.get("raw_output_buffer")

        i2c_address = self.config.get("i2c_address")
        i2c_bus = self.config.get("i2c_bus")
        if i2c_address is None or i2c_bus is None:
            self._dummy = True
            self.logger.warning(
                "No I2C address/bus configured — IMU worker idle (no data). "
                "Set use_oakd_imu: true to stream the OAK-D IMU instead."
            )
            return
        self._setup_i2c(i2c_address, i2c_bus)

    def _setup_i2c(self, address: int, bus: int) -> None:
        if smbus2 is None:
            self._dummy = True
            self.logger.warning("smbus2 not installed — IMU running in dummy mode (no data)")
            return
        self._i2c_address = address
        self._bus = smbus2.SMBus(bus)
        self._period = 1.0 / float(self.config.get("poll_hz", 100.0))
        self._next_poll = time.monotonic()
        self.logger.info(
            f"I2C IMU ready on bus {bus}, address 0x{address:02x}, "
            f"polling at {1.0 / self._period:.0f} Hz"
        )

    def _step_i2c(self) -> None:
        now = time.monotonic()
        if now < self._next_poll:
            time.sleep(self._next_poll - now)
        self._next_poll = max(self._next_poll + self._period, time.monotonic())

        try:
            # SMBus block reads stop at 32 bytes, so use a plain I2C read
            msg = smbus2.i2c_msg.read(self._i2c_address, I2C_FRAME_SIZE)
            self._bus.i2c_rdwr(msg)
        except OSError as exc:
            self._errors += 1
            backoff = min(0.05 * 2 ** min(self._errors, 5), 1.0)
            if self._errors in (1, 10) or self._errors % 100 == 0:
                self.logger.warning(
                    f"I2C read failed ({exc}), {self._errors} in a row — retry in {backoff:.2f}s"
                )
            self._next_poll = time.monotonic() + backoff
            return
        ts = time.monotonic()
        if self._errors:
            self.logger.info(f"I2C reads back after {self._errors} errors")
            self._errors = 0

        data = bytes(msg)
        vec = parse_i2c_frame(data)
        if vec is None:
            # e.g. the 4-byte handshake reply or a garbled transfer
            return
        self._imu_buf.write(vec, ts=ts)
        if self._raw_buf is not None:
            self._write_raw(data, ts)

    def _write_raw(self, data: bytes, ts: float) -> None:
        sample = parse_i2c_raw(data)
        if sample is None:
            return
        key = data[16:40]                      # accel + gyro bytes: unchanged = the same sample
        if key == self._last_raw:
            self._raw_repeats += 1
        else:
            self._last_raw = key
            self._raw_buf.write(sample, ts=ts)
            self._raw_written += 1
        if ts - self._stats_t >= 30.0:
            if self._stats_t:
                n = self._raw_written + self._raw_repeats
                self.logger.info(f"rear IMU: {self._raw_written / (ts - self._stats_t):.0f} new samples/s, "
                                 f"{100.0 * self._raw_repeats / max(n, 1):.0f} % of polls repeated the last one")
            self._stats_t, self._raw_written, self._raw_repeats = ts, 0, 0

    def step(self) -> None:
        if self._dummy:
            time.sleep(0.1)
        else:
            self._step_i2c()

    def teardown(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self.logger.info("I2C bus closed")
