"""
Reads the Arduino IMU over I2C (or a synthetic generator) and writes 9-float
vectors into an IMUBuffer:

    [qw, qx, qy, qz, ax, ay, az, throttle, steering]

i.e. orientation quaternion, accel in m/s², then the Arduino's raw
throttle/steering feedback. The Arduino's 60-byte frame (see I2C_FRAME) also
has gyro, magnetometer and its own timestamp; those are not published.
Samples are stamped with the host time.monotonic() of the read.

The OAK-D's own IMU isn't read here. Only one process can open a DepthAI
device, so CameraWorker streams it (use_oakd_imu: true, wired up in start.py).
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

# What the Arduino sends per request (firmware/firmware/jetson_i2c.cpp,
# jetson_request_event), little-endian, 60 bytes:
#   quat w, x, y, z            4 x float32
#   accel x, y, z (g)          3 x float32 (already divided by 1000 on the Arduino)
#   gyro x, y, z               3 x float32
#   mag x, y, z                3 x float32
#   timestamp (Arduino ms)     uint32
#   throttle, steering (raw)   2 x uint16
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


class IMUWorker(BaseWorker):
    """IMU worker. kwargs: output_buffer, i2c_address, i2c_bus, synthetic,
    poll_hz (I2C polling rate, default 100).

    I2C errors (OSError) are logged and retried with a growing back-off
    (up to 1 s); they never crash the worker.

    With no synthetic flag and no complete I2C address/bus it just idles and
    publishes nothing. Deliberately does not fall back to fake data when a
    real sensor is expected.
    """

    def __init__(self, name: str = "imu", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._imu_buf: IMUBuffer | None = None
        self._bus = None
        self._dummy = False
        self._synthetic = False
        self._synth_gen = None
        self._errors = 0
        self._period = 0.01
        self._next_poll = 0.0

    def setup(self) -> None:
        import os

        self._imu_buf = self.config["output_buffer"]

        use_synthetic = (
            self.config.get("synthetic", False)
            or os.environ.get("PILOT_SYNTHETIC", "") == "1"
        )

        if use_synthetic:
            self._setup_synthetic()
            return

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
        # rate limit: one read per period
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

        vec = parse_i2c_frame(bytes(msg))
        if vec is None:
            # e.g. the 4-byte handshake reply or a garbled transfer
            return
        self._imu_buf.write(vec, ts=ts)

    def _setup_synthetic(self) -> None:
        from synthetic import SyntheticIMUGenerator

        self._synth_gen = SyntheticIMUGenerator()
        self._synthetic = True
        self._dummy = False
        self.logger.info("Synthetic IMU ready")

    def _step_synthetic(self) -> None:
        vec = self._synth_gen.next()
        self._imu_buf.write(vec)
        time.sleep(0.005)  # ~200 Hz

    def step(self) -> None:
        if self._synthetic:
            self._step_synthetic()
        elif self._dummy:
            time.sleep(0.1)
        else:
            self._step_i2c()

    def teardown(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self.logger.info("I2C bus closed")
