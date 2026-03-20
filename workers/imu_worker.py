"""
IMUWorker — reads IMU data from I2C (Arduino) or OAK-D on-board IMU
and writes 16-float vectors into an IMUBuffer.

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
        self._synthetic = False
        self._synth_gen = None
        self._device = None
        self._queue = None
        self._sample_count: int = 0
        self._i2c_error_count: int = 0
        self._last_i2c_error_log_ts: float = 0.0

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
        self._use_oakd = self.config.get("use_oakd_imu", False)

        if self._use_oakd or i2c_address is None or i2c_bus is None:
            self._use_oakd = True
            try:
                self._setup_oakd_imu()
            except (ImportError, Exception) as exc:
                self.logger.warning(
                    f"OAK-D IMU unavailable ({exc}) — falling back to synthetic IMU"
                )
                self._setup_synthetic()
        else:
            self._setup_i2c(i2c_address, i2c_bus)

    #  I2C path 

    def _setup_i2c(self, address: int, bus: int) -> None:
        if smbus2 is None:
            self._dummy = True
            self.logger.warning("smbus2 not installed — IMU running in dummy mode (no data)")
            return
        self._i2c_address = address
        try:
            self._bus = smbus2.SMBus(bus)
        except Exception as exc:
            self._dummy = True
            self.logger.warning(f"Cannot open I2C bus {bus}: {exc} — IMU in dummy mode")
            return
        self.logger.info(f"I2C IMU ready on bus {bus}, address 0x{address:02x}")

        # Handshake with Arduino
        try:
            # Keep this handshake aligned with the legacy ingestion path.
            bytes_to_send = bytearray("10101001", encoding="utf-8")
            self._bus.write_block_data(self._i2c_address, 0, bytes_to_send)
            time.sleep(1.0)
            received = struct.unpack(
                "<HH",
                bytes(self._bus.read_i2c_block_data(self._i2c_address, 0, 4)),
            )
            idx_0, idx_1 = received
            if not (idx_0 == 1461 and idx_1 == 1397):
                raise RuntimeError(
                    f"Unexpected handshake response: idx_0={idx_0}, idx_1={idx_1}"
                )
            self.logger.info(
                "I2C handshake successful (idx_0=%s, idx_1=%s)",
                idx_0,
                idx_1,
            )
        except Exception as exc:
            self._dummy = True
            if self._bus is not None:
                try:
                    self._bus.close()
                except Exception:
                    pass
                self._bus = None
            self.logger.warning(
                f"I2C handshake failed: {exc} — IMU in dummy mode"
            )

    def _step_i2c(self) -> None:
        try:
            msg = smbus2.i2c_msg.read(self._i2c_address, 60)
            self._bus.i2c_rdwr(msg)
            data = list(msg)
            self._i2c_error_count = 0
        except OSError as exc:
            self._i2c_error_count += 1
            now = time.time()
            if (
                self._i2c_error_count <= 3
                or (now - self._last_i2c_error_log_ts) >= 1.0
            ):
                self.logger.warning(
                    "I2C read error #%d: %s",
                    self._i2c_error_count,
                    exc,
                )
                self._last_i2c_error_log_ts = now
            time.sleep(0.01)
            return

        raw = bytes(data)
        values = struct.unpack('fffffffffffffIHH', raw)

        w, x, y, z, ax, ay, az, gx, gy, gz, mx, my, mz, timestamp, throttle, steering = values
        # ax, ay, az -> milli g's
        ax *= 0.00980665  # Convert g to m/s^2
        ay *= 0.00980665  # Convert g to m/s^2
        az *= 0.00980665  # Convert g to m/s^2
        # gx, gy, gz -> degrees per second
        # mx, my, mz -> microteslas
        # timestamp -> microseconds (Arduino boot timer)

        # Create a Vector object with the processed data
        #sensors = Vector([w, x, y, z, ax, ay, az, throttle, steering], time.time())
        vec = np.array(
            [w, x, y, z, ax, ay, az, gx, gy, gz, mx, my, mz, float(throttle), float(steering), timestamp],
            dtype=np.float32,
        )
        self._imu_buf.write(vec)

        self._sample_count += 1
        if self._sample_count % 200 == 1:
            self.logger.debug(
                f"IMU #{self._sample_count}  "
                f"quat=[{w:.3f},{x:.3f},{y:.3f},{z:.3f}]  "
                f"accel=[{ax:.4f},{ay:.4f},{az:.4f}]  "
                f"gyro=[{gx:.2f},{gy:.2f},{gz:.2f}]  "
                f"mag=[{mx:.1f},{my:.1f},{mz:.1f}]  "
                f"thr={throttle} steer={steering}  "
                f"timestamp={timestamp}"
                
            )

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
                    0.0, 0.0, 0.0, 0.0,  # quaternion placeholder (w,x,y,z)
                    float(accel.x), float(accel.y), float(accel.z),
                    float(gyro.x), float(gyro.y), float(gyro.z),
                    0.0, 0.0, 0.0,  # magnetometer placeholder (mx,my,mz)
                    0.0, 0.0,  # raw throttle, steering placeholders
                    float(time.time()),
                ],
                dtype=np.float32,
            )
            self._imu_buf.write(vec)

            self._sample_count += 1
            if self._sample_count % 200 == 1:
                self.logger.debug(
                    f"IMU #{self._sample_count}  "
                    f"accel=[{accel.x:.4f},{accel.y:.4f},{accel.z:.4f}]  "
                    f"gyro=[{gyro.x:.2f},{gyro.y:.2f},{gyro.z:.2f}]"
                )

    # ── synthetic IMU ──────────────────────────────────────────────

    def _setup_synthetic(self) -> None:
        from synthetic import SyntheticIMUGenerator

        self._synth_gen = SyntheticIMUGenerator()
        self._synthetic = True
        self._dummy = False
        self._use_oakd = False
        self.logger.info("Synthetic IMU ready")

    def _step_synthetic(self) -> None:
        vec = self._synth_gen.next()
        self._imu_buf.write(vec)
        self._sample_count += 1
        if self._sample_count % 200 == 1:
            self.logger.debug(
                f"synth IMU #{self._sample_count}  vec={vec[:7]}"
            )
        time.sleep(0.005)  # ~200 Hz

    #  dispatch 

    def step(self) -> None:
        if getattr(self, '_synthetic', False):
            self._step_synthetic()
            return
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
