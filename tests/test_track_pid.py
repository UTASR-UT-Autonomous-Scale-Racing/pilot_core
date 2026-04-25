"""TrackPIDController (centre-line PID on BEV boundaries) and _angle_diff."""

from __future__ import annotations

import numpy as np
import pytest

from workers.track_pid_controller import TrackPIDController, _angle_diff


class TestAngleDiff:

    def test_zero_diff(self):
        assert abs(_angle_diff(0.0, 0.0)) < 1e-10

    def test_positive_diff(self):
        d = _angle_diff(1.0, 0.0)
        assert abs(d - 1.0) < 1e-10

    def test_wrap_around_positive(self):
        d = _angle_diff(3.0, -3.0)
        # short way round: 3 - (-3) = 6, wraps to about -0.283
        assert abs(d) < np.pi

    def test_wrap_around_negative(self):
        d = _angle_diff(-3.0, 3.0)
        assert abs(d) < np.pi

    def test_pi_boundary(self):
        d = _angle_diff(np.pi, -np.pi)
        assert abs(d) < 1e-6 or abs(abs(d) - 2 * np.pi) < 1e-6


class TestPIDInit:

    def test_default_init(self):
        pid = TrackPIDController()
        assert pid.kp == 1.0
        assert pid.ki == 0.0
        assert pid.kd == 0.1
        assert pid.target_throttle == 0.3
        assert pid.noise_std == 0.05

    def test_custom_init(self):
        pid = TrackPIDController(
            kp_steer=2.5, ki_steer=0.1, kd_steer=0.5,
            target_throttle=0.8, noise_std=0.0,
        )
        assert pid.kp == 2.5
        assert pid.ki == 0.1
        assert pid.kd == 0.5
        assert pid.target_throttle == 0.8
        assert pid.noise_std == 0.0

    def test_initial_state_zero(self):
        pid = TrackPIDController()
        assert pid._prev_error == 0.0
        assert pid._integral == 0.0

    def test_reset(self):
        pid = TrackPIDController()
        pid._prev_error = 5.0
        pid._integral = 10.0
        pid.reset()
        assert pid._prev_error == 0.0
        assert pid._integral == 0.0


# step()

@pytest.fixture()
def oval_boundaries():
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    outer = np.column_stack([8.0 * np.cos(theta), 5.0 * np.sin(theta)]).astype(np.float32)
    inner = np.column_stack([5.0 * np.cos(theta), 3.0 * np.sin(theta)]).astype(np.float32)
    return outer, inner


class TestPIDStep:

    def test_output_is_tuple_of_floats(self, oval_boundaries):
        pid = TrackPIDController(noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)
        result = pid.step(pos, 0.0, outer, inner)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], float)

    def test_output_clipped(self, oval_boundaries):
        pid = TrackPIDController(kp_steer=100.0, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([7.9, 0.0], dtype=np.float32)  # far from centre-line
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert -1.0 <= steer <= 1.0
        assert -1.0 <= throttle <= 1.0

    def test_deterministic_when_no_noise(self, oval_boundaries):
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)

        pid1 = TrackPIDController(noise_std=0.0)
        s1, t1 = pid1.step(pos, 0.0, outer, inner)

        pid2 = TrackPIDController(noise_std=0.0)
        s2, t2 = pid2.step(pos, 0.0, outer, inner)

        assert abs(s1 - s2) < 1e-10
        assert abs(t1 - t2) < 1e-10

    def test_stochastic_when_noise(self, oval_boundaries):
        pid = TrackPIDController(noise_std=0.2)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)

        steers = set()
        for _ in range(20):
            pid.reset()
            s, _ = pid.step(pos, np.pi / 2, outer, inner)
            steers.add(round(s, 4))

        assert len(steers) > 1

    def test_centre_line_zero_error(self, oval_boundaries):
        pid = TrackPIDController(kp_steer=1.0, noise_std=0.0)
        outer, inner = oval_boundaries
        # midpoint of outer[0]=(8,0) and inner[0]=(5,0)
        pos = np.array([6.5, 0.0], dtype=np.float32)
        heading = np.pi / 2  # along the track tangent (+Z) at theta = 0
        steer, throttle = pid.step(pos, heading, outer, inner)
        assert abs(steer) < 0.05

    def test_throttle_near_target(self, oval_boundaries):
        pid = TrackPIDController(target_throttle=0.4, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)
        _, throttle = pid.step(pos, 0.0, outer, inner)
        assert abs(throttle - 0.4) < 0.01

    def test_integral_accumulation(self, oval_boundaries):
        pid = TrackPIDController(ki_steer=1.0, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([7.5, 0.0], dtype=np.float32)  # off-centre

        for _ in range(5):
            pid.step(pos, 0.0, outer, inner)

        assert pid._integral != 0.0


class TestPIDEdgeCases:

    def test_single_point_boundaries(self):
        pid = TrackPIDController(noise_std=0.0)
        outer = np.array([[10.0, 0.0]], dtype=np.float32)
        inner = np.array([[5.0, 0.0]], dtype=np.float32)
        pos = np.array([7.5, 0.0], dtype=np.float32)
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert np.isfinite(throttle)

    def test_two_point_boundaries(self):
        pid = TrackPIDController(noise_std=0.0)
        outer = np.array([[10.0, -5.0], [10.0, 5.0]], dtype=np.float32)
        inner = np.array([[5.0, -5.0], [5.0, 5.0]], dtype=np.float32)
        pos = np.array([7.5, 0.0], dtype=np.float32)
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert np.isfinite(throttle)

    def test_vehicle_on_outer_boundary(self, oval_boundaries):
        pid = TrackPIDController(noise_std=0.0)
        outer, inner = oval_boundaries
        pos = outer[0].copy()
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert -1.0 <= steer <= 1.0

    def test_vehicle_on_inner_boundary(self, oval_boundaries):
        pid = TrackPIDController(noise_std=0.0)
        outer, inner = oval_boundaries
        pos = inner[0].copy()
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert -1.0 <= steer <= 1.0

    def test_reversed_heading(self, oval_boundaries):
        # facing backwards; for now only checks the output is finite
        pid = TrackPIDController(kp_steer=1.0, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)
        heading = np.pi
        steer, _ = pid.step(pos, heading, outer, inner)
        assert np.isfinite(steer)
