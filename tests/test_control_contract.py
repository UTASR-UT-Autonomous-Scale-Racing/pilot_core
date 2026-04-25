"""
The ControlBuffer [0, 1] contract between the controllers, ControlWorker and
ActuatorWorker. 0.5 is neutral; PID/FTG output is [-1, 1] and gets mapped by
signed_to_unit(). The actuator has to clamp bad values (NaN included), not crash.
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import ControlBuffer, PoseBuffer, TrackBoundaryBuffer
from workers.control_worker import ControlWorker, pose_to_bev, signed_to_unit


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

    def test_straight_at_target_gives_zero_steer(self):
        from workers.track_pid_controller import TrackPIDController

        pid = TrackPIDController(kp_steer=1.0, ki_steer=0.0, kd_steer=0.0, noise_std=0.0)
        T = np.eye(4, dtype=np.float32)
        T[:3, 2] = (1, 0, 0)
        pos, heading = pose_to_bev(T)
        outer = np.array([[5.0, 1.0]], dtype=np.float32)
        inner = np.array([[5.0, -1.0]], dtype=np.float32)
        steer, _ = pid.step(pos, heading, outer, inner)
        assert steer == pytest.approx(0.0, abs=1e-6)


class TestSteeringDirection:
    """A track bending left has to give a left command (buffer < 0.5), since
    PWM above neutral steers right on the car."""

    @pytest.mark.parametrize("bend, left", [(+0.4, True), (-0.4, False)])
    def test_pid_turns_towards_track(self, bend, left):
        from workers.track_pid_controller import TrackPIDController

        pid = TrackPIDController(kp_steer=1.0, ki_steer=0.0, kd_steer=0.0, noise_std=0.0)
        # car at origin heading +X (camera forward axis Z lined up with world X)
        T = np.eye(4, dtype=np.float32)
        T[:3, 2] = (1, 0, 0)
        pos, heading = pose_to_bev(T)
        # walls either side of a track whose tangent is rotated by `bend` rad,
        # positive is CCW seen from above, i.e. to the car's left
        tangent = np.array([np.cos(bend), np.sin(bend)])
        normal = np.array([-tangent[1], tangent[0]])
        outer = (pos + normal * -1.0)[None].astype(np.float32)
        inner = (pos + normal * +1.0)[None].astype(np.float32)
        steer_signed, _ = pid.step(pos, heading, outer, inner)
        steer_unit, _ = signed_to_unit(steer_signed, 0.0)
        assert (steer_unit < 0.5) == left


class TestControlWorkerPIDContract:

    def test_pid_output_in_unit_range(self, track_boundaries):
        outer, inner = track_boundaries
        ctrl = ControlBuffer(slots=4, name=f"cc_ctrl_{_uid()}", create=True)
        pose = PoseBuffer(slots=4, name=f"cc_pose_{_uid()}", create=True)
        bnd = TrackBoundaryBuffer(slots=2, name=f"cc_bnd_{_uid()}", create=True)
        try:
            bnd.write(outer, inner)
            w = ControlWorker(
                name="control", mode="pid", output_buffer=ctrl,
                pose_buffer=pose, boundary_buffer=bnd,
                pid_config={"kp_steer": 5.0, "noise_std": 0.0},
            )
            w._logger = MagicMock()
            w.setup()

            for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                T = np.eye(4, dtype=np.float32)
                T[0, 3], T[2, 3] = 6.5 * np.cos(theta), 4.0 * np.sin(theta)
                # heading points away from the track on purpose, to force big errors
                T[0, 2], T[2, 2] = np.cos(theta), np.sin(theta)
                pose.write(T)
                w.step()
                _, steer, thr = ctrl.read_latest()
                assert 0.0 <= steer <= 1.0
                assert 0.0 <= thr <= 1.0
        finally:
            for b in (ctrl, pose, bnd):
                b.unlink()


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
