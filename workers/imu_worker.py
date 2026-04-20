"""
IMUWorker — reads IMU data from the Arduino over I2C (or a synthetic
generator) and writes 9-float vectors into an IMUBuffer.

The OAK-D on-board IMU is *not* handled here: a DepthAI device can only be
opened by one process, so it is streamed by :class:`CameraWorker` instead
(``use_oakd_imu: true`` in the config wires this up in ``start.py``).

IMUBuffer vector layout (9 x float32):
``[qw, qx, qy, qz, ax, ay, az, throttle, steering]`` — orientation
quaternion, accelerometer in m/s², then the Arduino's raw throttle /
steering feedback values.
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


class IMUWorker(BaseWorker):
    """Worker process for IMU sensor data.

    Config kwargs
    -------------
    output_buffer : IMUBuffer
        Shared IMUBuffer to write into.
    i2c_address : int | None
        I2C slave address of the Arduino.
    i2c_bus : int | None
        I2C bus number.
    synthetic : bool
        Generate synthetic IMU data instead of reading hardware.

    If neither synthetic mode nor a complete I2C address/bus is configured
    the worker idles and publishes nothing — it never substitutes fake data
    for a real sensor.
    """

    def __init__(self, name: str = "imu", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._imu_buf: IMUBuffer | None = None
        self._bus = None
        self._dummy = False
        self._synthetic = False
        self._synth_gen = None

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

    # --- I2C path ---

    def _setup_i2c(self, address: int, bus: int) -> None:
        if smbus2 is None:
            self._dummy = True
            self.logger.warning("smbus2 not installed — IMU running in dummy mode (no data)")
            return
        self._i2c_address = address
        self._bus = smbus2.SMBus(bus)
        self.logger.info(f"I2C IMU ready on bus {bus}, address 0x{address:02x}")

    def _step_i2c(self) -> None:
        # 32-byte little-endian block: 7 x float32 (qw, qx, qy, qz, ax, ay, az;
        # accel in milli-g) + 2 x uint16 (throttle, steering feedback)
        raw = bytes(self._bus.read_i2c_block_data(self._i2c_address, 0, 32))
        values = struct.unpack("<fffffffHH", raw)
        w, x, y, z, ax, ay, az, throttle, steering = values
        # milli-g → m/s²
        ax *= 0.00980665
        ay *= 0.00980665
        az *= 0.00980665
        vec = np.array(
            [w, x, y, z, ax, ay, az, float(throttle), float(steering)],
            dtype=np.float32,
        )
        self._imu_buf.write(vec)

    # --- synthetic IMU ---

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

    # --- dispatch ---

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
