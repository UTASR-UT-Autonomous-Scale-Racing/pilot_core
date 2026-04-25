"""
Reads the Arduino IMU over I2C (or a synthetic generator) and writes 9-float
vectors into an IMUBuffer:

    [qw, qx, qy, qz, ax, ay, az, throttle, steering]

i.e. orientation quaternion, accel in m/s², then the Arduino's raw
throttle/steering feedback.

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


class IMUWorker(BaseWorker):
    """IMU worker. kwargs: output_buffer, i2c_address, i2c_bus, synthetic.

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
        self.logger.info(f"I2C IMU ready on bus {bus}, address 0x{address:02x}")

    def _step_i2c(self) -> None:
        # 32 bytes, little-endian: 7 x float32 (qw qx qy qz ax ay az, accel in
        # milli-g) then 2 x uint16 throttle/steering feedback
        raw = bytes(self._bus.read_i2c_block_data(self._i2c_address, 0, 32))
        values = struct.unpack("<fffffffHH", raw)
        w, x, y, z, ax, ay, az, throttle, steering = values
        # milli-g -> m/s^2
        ax *= 0.00980665
        ay *= 0.00980665
        az *= 0.00980665
        vec = np.array(
            [w, x, y, z, ax, ay, az, float(throttle), float(steering)],
            dtype=np.float32,
        )
        self._imu_buf.write(vec)

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
