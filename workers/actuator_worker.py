"""
ActuatorWorker — reads normalised control values from ControlBuffer,
maps them to PWM via ``pilot_controller.Controller``, and writes
to the Arduino over I2C.

Replaces ``I2CDataProcessor.write()`` in the old pipeline.
"""

from __future__ import annotations

import math
import time
from typing import Any

from .base_worker import BaseWorker
from shared_buffer import ControlBuffer

try:
    import smbus2
except ImportError:
    smbus2 = None


class ActuatorWorker(BaseWorker):
    """Worker process that drives steering/throttle actuators via I2C.

    Config kwargs
    -------------
    input_buffer : ControlBuffer
        Shared ControlBuffer to read from.
    i2c_address : int | None
        I2C slave address of the Arduino.
    i2c_bus : int | None
        I2C bus number.
    """

    def __init__(self, name: str = "actuator", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._ctrl_buf: ControlBuffer | None = None
        self._bus = None
        self._i2c_address: int = 0
        self._controller = None
        self._last_seq = 0
        self._dummy = False

    def setup(self) -> None:
        self._ctrl_buf = self.config["input_buffer"]

        i2c_address = self.config.get("i2c_address")
        i2c_bus = self.config.get("i2c_bus")

        if i2c_address is None or i2c_bus is None:
            self._dummy = True
            self.logger.warning("No I2C address/bus — running in dummy mode (no actuator writes)")
        else:
            if smbus2 is None:
                self._dummy = True
                self.logger.warning("smbus2 not installed — running in dummy mode")
            else:
                self._i2c_address = i2c_address
                self._bus = smbus2.SMBus(i2c_bus)
                self.logger.info(f"I2C actuator ready on bus {i2c_bus}, address 0x{i2c_address:02x}")

        # PWM mapper from pilot_controller
        from pilot_controller.control_output import Controller
        self._controller = Controller()

    def step(self) -> None:
        result = self._ctrl_buf.read_latest()
        if result is None:
            time.sleep(0.005)
            return

        header, steering_norm, throttle_norm = result

        # Skip if we already sent this command
        if header.seq_no <= self._last_seq:
            time.sleep(0.005)
            return
        self._last_seq = header.seq_no

        # Last line of defence: an out-of-range command must never crash the
        # actuator loop (the car would hold its previous PWM).  Clamp and warn.
        if not (0.0 <= steering_norm <= 1.0 and 0.0 <= throttle_norm <= 1.0):
            self.logger.warning(
                f"Control command out of [0, 1] range "
                f"(steering={steering_norm:.3f}, throttle={throttle_norm:.3f}) — clamping"
            )
            # NaN → neutral steering / zero throttle
            steering_norm = min(max(steering_norm, 0.0), 1.0) if math.isfinite(steering_norm) else 0.5
            throttle_norm = min(max(throttle_norm, 0.0), 1.0) if math.isfinite(throttle_norm) else 0.0

        # Map normalised → PWM
        pwm_steering, pwm_throttle = self._controller.get_normalized_action_output(
            steering_norm, throttle_norm
        )

        if not self._dummy:
            self._write_i2c(pwm_steering, pwm_throttle)

    def _write_i2c(self, steering: int, throttle: int) -> None:
        """Send PWM values to the Arduino over I2C."""
        data_str = str(steering) + str(throttle)
        data_bytes = bytearray(data_str, encoding="utf-8")
        try:
            self._bus.write_block_data(self._i2c_address, 0, data_bytes)
        except Exception:
            self.logger.exception("I2C write failed")

    def teardown(self) -> None:
        # Send neutral values before shutting down
        if self._controller is not None and not self._dummy:
            from pilot_controller.control_output import STEER_NEURAL, THROTTLE_NEURAL
            self._write_i2c(STEER_NEURAL, THROTTLE_NEURAL)
        if self._bus is not None:
            self._bus.close()
            self.logger.info("I2C bus closed")
