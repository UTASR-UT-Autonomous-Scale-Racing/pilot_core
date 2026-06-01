"""
ControlBuffer ([0, 1], 0.5 = neutral) -> SafetySupervisor -> PWM over I2C to the
Arduino and/or signed [-1, 1] (steering +1 = left) over ZMQ to the sim, and/or the
supervised command into an output ControlBuffer (AutoDRIVEWorker sends that to the sim).
I2C is silent while disarmed so the Arduino passes the RC receiver through; once
armed, commands go out at least every 100 ms (Arduino times out to RC / neutral).
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from control.types import Command

from .base_worker import BaseWorker
from .control_worker import signed_to_unit, unit_to_signed
from shared_buffer import ControlBuffer

try:
    import smbus2
except ImportError:
    smbus2 = None

try:
    import zmq
except ImportError:
    zmq = None


class ActuatorWorker(BaseWorker):
    """Drives steering/throttle via I2C (Arduino) and/or ZMQ (simulator).

    Config kwargs:
      input_buffer   ControlBuffer to read from
      i2c_address    Arduino slave address; None (or no i2c_bus) = dummy mode
      i2c_bus        I2C bus number
      zmq_enabled    publish to the simulator (default False)
      zmq_port       PUB bind port (default 5556)
      output_buffer  ControlBuffer for the supervised command, written every tick
                     (same [0, 1] units as input_buffer); None = not written
      armed_event    neutral (I2C silent) unless set; None = always armed (tests only)
      state_buffer   vehicle state for the speed cap (else a plain throttle cap)
      safety         SafetyConfig overrides
      pwm            PWMConfig overrides (pulse widths)
      reset_event    if set, the next ZMQ message carries reset: true
      rate_hz        output loop rate, default 50

    Every tick is sent through SafetySupervisor, so a stalled controller
    still ends with the car at neutral.
    """

    def __init__(self, name: str = "actuator", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._ctrl_buf: ControlBuffer | None = None
        self._bus = None
        self._i2c_address: int = 0
        self._pwm = None
        self._last_seq = 0
        self._dummy = False
        self._zmq_enabled = False
        self._zmq_context = None
        self._zmq_pub = None
        self._reset_event = None
        self._out_buf: ControlBuffer | None = None
        self._supervisor = None
        self._armed_event = None
        self._state_buf = None
        self._period = 0.02
        self._next_tick = 0.0
        self._last_pwm = None
        self._last_i2c_write = 0.0
        self._i2c_active = False       # written since the last arm
        self._last_status = None

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

        self._reset_event = self.config.get("reset_event")
        self._out_buf = self.config.get("output_buffer")

        self._zmq_enabled = self.config.get("zmq_enabled", False)
        if self._zmq_enabled:
            if zmq is None:
                self.logger.warning("pyzmq not installed — ZMQ disabled")
                self._zmq_enabled = False
            else:
                zmq_port = self.config.get("zmq_port", 5556)
                try:
                    self._zmq_context = zmq.Context()
                    self._zmq_pub = self._zmq_context.socket(zmq.PUB)
                    self._zmq_pub.setsockopt(zmq.SNDHWM, 10)
                    self._zmq_pub.setsockopt(zmq.LINGER, 1000)
                    self._zmq_pub.bind(f"tcp://*:{zmq_port}")
                    time.sleep(0.5)  # give subscribers time to connect before the first send
                    self.logger.info(f"ZMQ control publisher bound to port {zmq_port}")
                except Exception as e:
                    self.logger.warning(f"Failed to bind ZMQ publisher: {e}")
                    self._zmq_enabled = False
                    if self._zmq_pub:
                        self._zmq_pub.close()
                    if self._zmq_context:
                        self._zmq_context.term()

        from control.pwm import PWMConfig, PWMMapper
        self._pwm = PWMMapper(PWMConfig(**self.config.get("pwm", {})))

        from control.safety import SafetyConfig, SafetySupervisor
        self._supervisor = SafetySupervisor(SafetyConfig(**self.config.get("safety", {})))
        self._armed_event = self.config.get("armed_event")
        self._state_buf = self.config.get("state_buffer")
        self._period = 1.0 / float(self.config.get("rate_hz", 50.0))
        cfg = self._supervisor.cfg
        self.logger.info(
            f"Safety: timeout={cfg.command_timeout * 1000:.0f} ms, speed cap={cfg.speed_cap} m/s, "
            f"arming={'terminal' if self._armed_event is not None else 'always armed'}"
        )

    def step(self) -> None:
        try:
            self._tick()
        except Exception:
            self._write_neutral()
            raise

    def _write_neutral(self) -> None:
        if self._out_buf is not None:
            self._out_buf.write(*signed_to_unit(0.0, 0.0))
        if self._zmq_enabled and self._zmq_pub is not None:
            self._publish_zmq(0.0, 0.0)
        if not self._dummy and self._pwm is not None:
            self._write_i2c(*self._pwm.neutral)

    def _tick(self) -> None:
        now = time.monotonic()
        if now < self._next_tick:
            time.sleep(min(self._next_tick - now, 0.005))
            return
        self._next_tick = now + self._period

        # ControlBuffer is [0, 1], convert to signed
        command = command_time = None
        result = self._ctrl_buf.read_latest()
        if result is not None:
            header, steering_norm, throttle_norm = result
            steering_norm, throttle_norm = self._sanitize(steering_norm, throttle_norm)
            s_signed, t_signed = unit_to_signed(steering_norm, throttle_norm)
            command, command_time = Command(s_signed, t_signed), header.produced_ts

        armed = self._armed_event.is_set() if self._armed_event is not None else True
        out = self._supervisor.step(now, command, command_time, armed, speed=self._speed(now))
        self._log_status()

        steering_norm, throttle_norm = signed_to_unit(out.steering, out.throttle)

        if self._out_buf is not None:
            self._out_buf.write(steering_norm, throttle_norm, ts=now)

        if self._zmq_enabled:
            reset = False
            if self._reset_event is not None and self._reset_event.is_set():
                reset = True
                self._reset_event.clear()
            self._publish_zmq(out.steering, out.throttle, reset)

        # Hardware. Only write on change, plus a 100 ms keep-alive. On disarm,
        # one last neutral write, then silence so the RC receiver takes over.
        pwm = self._pwm.to_pwm(steering_norm, throttle_norm)
        if not self._dummy:
            if armed and (pwm != self._last_pwm or now - self._last_i2c_write > 0.1):
                self._write_i2c(*pwm)
                self._last_i2c_write = now
                self._i2c_active = True
            elif not armed and self._i2c_active:
                self._write_i2c(*self._pwm.neutral)
                self._i2c_active = False
        self._last_pwm = pwm

    def _sanitize(self, steering_norm: float, throttle_norm: float) -> tuple[float, float]:
        """Clamp out-of-range or NaN commands rather than crash the loop."""
        if 0.0 <= steering_norm <= 1.0 and 0.0 <= throttle_norm <= 1.0:
            return steering_norm, throttle_norm
        self.logger.warning(
            f"Control command out of [0, 1] range "
            f"(steering={steering_norm:.3f}, throttle={throttle_norm:.3f}) — clamping"
        )
        # NaN -> 0.5 (neutral)
        steering_norm = min(max(steering_norm, 0.0), 1.0) if math.isfinite(steering_norm) else 0.5
        throttle_norm = min(max(throttle_norm, 0.0), 1.0) if math.isfinite(throttle_norm) else 0.5
        return steering_norm, throttle_norm

    def _speed(self, now: float):
        if self._state_buf is None:
            return None
        result = self._state_buf.read_latest()
        if result is None:
            return None
        _, state = result
        return state.v if now - state.t < 0.3 else None

    def _log_status(self) -> None:
        st = self._supervisor.status
        key = (st.armed, st.watchdog_tripped)
        if key == self._last_status:
            return
        if self._last_status is not None or st.armed:
            if not st.armed:
                self.logger.warning("DISARMED — outputs neutral")
            elif st.watchdog_tripped:
                self.logger.warning("Command watchdog — no fresh commands, throttle neutral")
            else:
                self.logger.info("ARMED — driving")
        self._last_status = key

    def _write_i2c(self, steering: int, throttle: int) -> None:
        """SMBus block write to register 0: ASCII steering then throttle pulse
        widths in us, no separator (e.g. "14611397")."""
        data_str = str(steering) + str(throttle)
        data_bytes = bytearray(data_str, encoding="utf-8")
        try:
            self._bus.write_block_data(self._i2c_address, 0, data_bytes)
        except Exception:
            self.logger.exception("I2C write failed")

    def _publish_zmq(self, steering: float, throttle: float, reset: bool = False) -> None:
        command = {
            'throttle': float(throttle),
            'steering': float(steering),
            'reset': reset,
        }

        try:
            cmd_json = json.dumps(command)
            self._zmq_pub.send_string(cmd_json, zmq.NOBLOCK)
        except Exception as e:
            self.logger.debug(f"ZMQ publish failed: {e}")

    def teardown(self) -> None:
        # leave everything at neutral on the way out
        if self._out_buf is not None:
            self._out_buf.write(*signed_to_unit(0.0, 0.0))
        if self._zmq_enabled and self._zmq_pub is not None:
            try:
                self._publish_zmq(0.0, 0.0)
            except Exception:
                pass
            self._zmq_pub.close()
            if self._zmq_context is not None:
                self._zmq_context.term()
            self.logger.info("ZMQ publisher closed")

        if self._pwm is not None and not self._dummy:
            self._write_i2c(*self._pwm.neutral)
        if self._bus is not None:
            self._bus.close()
            self.logger.info("I2C bus closed")
