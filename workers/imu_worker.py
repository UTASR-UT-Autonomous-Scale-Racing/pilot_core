"""
IMUWorker — reads IMU data from I2C (Arduino) or OAK-D on-board IMU
and writes 9-float vectors into an IMUBuffer.

Replaces ``pilot_ingest.I2CDataProcessor.read()`` loop and the IMU portion
of ``SegmentProcessor.read_stream()``.
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

try:
    import depthai as dai
except ImportError:
    dai = None


class IMUWorker(BaseWorker):
    """Worker process for IMU sensor data.

    Config kwargs
    -------------
    output_buffer : IMUBuffer
        Shared IMUBuffer to write into.
    i2c_address : int | None
        I2C slave address of the Arduino.  ``None`` → use OAK-D IMU.
    i2c_bus : int | None
        I2C bus number.  ``None`` → use OAK-D IMU.
    use_oakd_imu : bool
        Explicit flag — if True, use OAK-D IMU regardless of I2C settings.
    """

    def __init__(self, name: str = "imu", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._imu_buf: IMUBuffer | None = None
        self._bus = None
        self._use_oakd = False
        self._dummy = False
        self._device = None
        self._queue = None

    def setup(self) -> None:
        self._imu_buf = self.config["output_buffer"]

        i2c_address = self.config.get("i2c_address")
        i2c_bus = self.config.get("i2c_bus")
        self._use_oakd = self.config.get("use_oakd_imu", False)

        if self._use_oakd or i2c_address is None or i2c_bus is None:
            self._use_oakd = True
            self._setup_oakd_imu()
        else:
            self._setup_i2c(i2c_address, i2c_bus)

    #  I2C path 

    def _setup_i2c(self, address: int, bus: int) -> None:
        if smbus2 is None:
            self._dummy = True
            self.logger.warning("smbus2 not installed — IMU running in dummy mode (no data)")
            return
        self._i2c_address = address
        self._bus = smbus2.SMBus(bus)
        self.logger.info(f"I2C IMU ready on bus {bus}, address 0x{address:02x}")

    def _step_i2c(self) -> None:
        raw = bytes(self._bus.read_i2c_block_data(self._i2c_address, 0, 32))
        values = struct.unpack("<fffffffHH", raw)
        w, x, y, z, ax, ay, az, throttle, steering = values
        # Convert accelerometer from g → m/s²
        ax *= 0.00980665
        ay *= 0.00980665
        az *= 0.00980665
        vec = np.array(
            [w, x, y, z, ax, ay, az, float(throttle), float(steering)],
            dtype=np.float32,
        )
        self._imu_buf.write(vec)

    #  OAK-D IMU path ─

    def _setup_oakd_imu(self) -> None:
        if dai is None:
            raise ImportError("depthai is required for OAK-D IMU")

        pipeline = dai.Pipeline()
        imu = pipeline.create(dai.node.IMU)
        imu.enableIMUSensor(
            [dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200
        )
        imu.setBatchReportThreshold(2)
        imu.setMaxBatchReports(10)

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("imu")
        imu.out.link(xout.input)

        self._device = dai.Device(pipeline)
        self._queue = self._device.getOutputQueue("imu", maxSize=16, blocking=False)
        self.logger.info("OAK-D IMU ready @ 200 Hz")

    def _step_oakd(self) -> None:
        imu_data = self._queue.tryGet()
        if imu_data is None:
            time.sleep(0.001)
            return

        for packet in imu_data.packets:
            accel = packet.acceleroMeter
            gyro = packet.gyroscope
            vec = np.array(
                [
                    0.0, 0.0, 0.0, 0.0,  # placeholder quaternion (w,x,y,z)
                    accel.x, accel.y, accel.z,
                    gyro.x, gyro.y,  # repurpose last 2 slots for gyro x,y
                ],
                dtype=np.float32,
            )[:9]
            self._imu_buf.write(vec)

    #  dispatch 

    def step(self) -> None:
        if self._dummy:
            time.sleep(0.1)
            return
        if self._use_oakd:
            self._step_oakd()
        else:
            self._step_i2c()

    def teardown(self) -> None:
        if self._bus is not None:
            self._bus.close()
            self.logger.info("I2C bus closed")
        if self._device is not None:
            self._device.close()
            self.logger.info("OAK-D IMU device closed")
