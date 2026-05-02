"""
The ControlBuffer [0, 1] contract between the controllers, ControlWorker and
ActuatorWorker. 0.5 is neutral; controller output is signed [-1, 1] and gets
mapped by signed_to_unit(). The actuator has to clamp bad values (NaN included), not crash.
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import ControlBuffer
from workers.bev import pose_to_bev
from workers.control_worker import signed_to_unit


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestSignedToUnit:

    @pytest.mark.parametrize(
        "signed, expected",
        # signed +1 = full left -> buffer 0.0 (on the car, PWM below neutral steers left)
        [(1.0, 0.0), (0.0, 0.5), (-1.0, 1.0), (0.5, 0.25), (3.0, 0.0), (-3.0, 1.0)],
    )
    def test_steering_mapping(self, signed, expected):
        steer, _ = signed_to_unit(signed, 0.3)
        assert steer == pytest.approx(expected)

    @pytest.mark.parametrize(
        "throttle, expected",
        [(-1.0, 0.0), (-0.4, 0.3), (0.0, 0.5), (0.3, 0.65), (1.7, 1.0)],
    )
    def test_throttle_mapping_allows_reverse(self, throttle, expected):
        _, thr = signed_to_unit(0.0, throttle)
        assert thr == pytest.approx(expected)


class TestPoseToBev:

    @pytest.mark.parametrize(
        "forward, expected",
        [((1, 0, 0), 0.0), ((0, 0, 1), np.pi / 2), ((-1, 0, 0), np.pi), ((0, 0, -1), -np.pi / 2)],
    )
    def test_heading_matches_bearing_convention(self, forward, expected):
        T = np.eye(4, dtype=np.float32)
        T[:3, 2] = forward
        T[0, 3], T[2, 3] = 1.5, -2.0
        pos, heading = pose_to_bev(T)
        np.testing.assert_allclose(pos, [1.5, -2.0])
        assert np.isclose(np.cos(heading), np.cos(expected), atol=1e-6)
        assert np.isclose(np.sin(heading), np.sin(expected), atol=1e-6)


class TestActuatorClamp:

    @pytest.mark.parametrize(
        "steer, thr, expected",
        [(-0.3, 0.2, (0.0, 0.2)), (1.4, 1.2, (1.0, 1.0)), (math.nan, math.nan, (0.5, 0.5))],
    )
    def test_out_of_range_is_clamped(self, steer, thr, expected):
        from workers import ActuatorWorker

        w = ActuatorWorker(name="actuator", input_buffer=None)
        w._logger = MagicMock()
        assert w._sanitize(steer, thr) == pytest.approx(expected)

    def test_out_of_range_command_does_not_crash_loop(self):
        from workers import ActuatorWorker

        ctrl = ControlBuffer(slots=4, name=f"cc_act_{_uid()}", create=True)
        try:
            w = ActuatorWorker(name="actuator", input_buffer=ctrl, i2c_address=None, i2c_bus=None)
            w._logger = MagicMock()
            w.setup()
            ctrl.write(math.nan, 7.0)
            w.step()  # must not raise
        finally:
            ctrl.unlink()


class TestPWMMapping:
    # ControlBuffer [0, 1] -> PWM (control.pwm.PWMMapper)

    def test_throttle_neutral_backward_forward(self):
        from control.pwm import PWMConfig, PWMMapper

        c = PWMConfig()
        m = PWMMapper(c)
        assert m.to_pwm(0.5, 0.5)[1] == c.throttle_neutral
        assert m.to_pwm(0.5, 0.0)[1] == c.throttle_min      # full backward
        assert m.to_pwm(0.5, 1.0)[1] == c.throttle_max

    def test_steering_full_range(self):
        from control.pwm import PWMConfig, PWMMapper

        c = PWMConfig()
        m = PWMMapper(c)
        assert m.to_pwm(0.0, 0.5)[0] == c.steer_min        # full left
        assert m.to_pwm(1.0, 0.5)[0] == c.steer_max        # full right

    def test_rejects_out_of_range(self):
        from control.pwm import PWMMapper

        with pytest.raises(ValueError):
            PWMMapper().to_pwm(1.2, 0.5)

    def test_values_from_config(self):
        from control.pwm import PWMConfig, PWMMapper

        m = PWMMapper(PWMConfig(throttle_min=1000, throttle_max=2000, throttle_neutral=1500))
        assert m.to_pwm(0.5, 0.5)[1] == 1500
