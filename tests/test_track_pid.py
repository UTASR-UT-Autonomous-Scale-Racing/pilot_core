"""
test_track_pid.py — Tests for TrackPIDController and helper functions.

Categories:
    - Smoke: instantiation with defaults and custom gains
    - I/O: step() produces valid (steering, throttle) tuples
    - Null checks: degenerate boundaries, single-point boundary
    - Initialisation: verify internal state, reset()
    - Deterministic vs stochastic: noise_std=0 → consistent output
    - Edge cases: vehicle on boundary, at centre, reversed heading
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.track_pid_controller import TrackPIDController, _angle_diff


#
# --- _angle_diff helper ---
#

class TestAngleDiff:

    def test_zero_diff(self):
        assert abs(_angle_diff(0.0, 0.0)) < 1e-10

    def test_positive_diff(self):
        d = _angle_diff(1.0, 0.0)
        assert abs(d - 1.0) < 1e-10

    def test_wrap_around_positive(self):
        d = _angle_diff(3.0, -3.0)
        # Should be the short way around: 3 - (-3) = 6 → wraps to ~-0.283
        assert abs(d) < np.pi

    def test_wrap_around_negative(self):
        d = _angle_diff(-3.0, 3.0)
        assert abs(d) < np.pi

    def test_pi_boundary(self):
        d = _angle_diff(np.pi, -np.pi)
        assert abs(d) < 1e-6 or abs(abs(d) - 2 * np.pi) < 1e-6


#
# --- TrackPIDController — Smoke & Initialisation ---
#

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


#
# --- PID step() I/O tests ---
#

@pytest.fixture()
def oval_boundaries():
    """Oval outer + inner boundaries centred at origin."""
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
        """Even with extreme gains, output should be clipped to [-1, 1]."""
        pid = TrackPIDController(kp_steer=100.0, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([7.9, 0.0], dtype=np.float32)  # far from centre-line
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert -1.0 <= steer <= 1.0
        assert -1.0 <= throttle <= 1.0

    def test_deterministic_when_no_noise(self, oval_boundaries):
        """With noise_std=0, two calls with same input give same output."""
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)

        pid1 = TrackPIDController(noise_std=0.0)
        s1, t1 = pid1.step(pos, 0.0, outer, inner)

        pid2 = TrackPIDController(noise_std=0.0)
        s2, t2 = pid2.step(pos, 0.0, outer, inner)

        assert abs(s1 - s2) < 1e-10
        assert abs(t1 - t2) < 1e-10

    def test_stochastic_when_noise(self, oval_boundaries):
        """With noise_std > 0, repeated calls produce some variation."""
        pid = TrackPIDController(noise_std=0.2)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)

        steers = set()
        for _ in range(20):
            pid.reset()
            s, _ = pid.step(pos, 0.0, outer, inner)
            steers.add(round(s, 4))

        assert len(steers) > 1  # should not all be identical

    def test_centre_line_zero_error(self, oval_boundaries):
        """Vehicle at midpoint of boundaries with correct heading → small steering."""
        pid = TrackPIDController(kp_steer=1.0, noise_std=0.0)
        outer, inner = oval_boundaries
        # Midpoint between outer[0]=(8,0) and inner[0]=(5,0) is (6.5, 0)
        pos = np.array([6.5, 0.0], dtype=np.float32)
        heading = 0.0  # pointing along X
        steer, throttle = pid.step(pos, heading, outer, inner)
        # Steering should be close to zero (heading towards centre)
        # Depending on exact geometry it may not be exactly 0, but should be small
        assert abs(steer) < 0.5

    def test_throttle_near_target(self, oval_boundaries):
        """Throttle should be close to target_throttle (with no noise)."""
        pid = TrackPIDController(target_throttle=0.4, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)
        _, throttle = pid.step(pos, 0.0, outer, inner)
        assert abs(throttle - 0.4) < 0.01

    def test_integral_accumulation(self, oval_boundaries):
        """Multiple steps should accumulate integral term."""
        pid = TrackPIDController(ki_steer=1.0, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([7.5, 0.0], dtype=np.float32)  # off-centre

        for _ in range(5):
            pid.step(pos, 0.0, outer, inner)

        assert pid._integral != 0.0


#
# --- Edge cases ---
#

class TestPIDEdgeCases:

    def test_single_point_boundaries(self):
        """Degenerate: each boundary is a single point."""
        pid = TrackPIDController(noise_std=0.0)
        outer = np.array([[10.0, 0.0]], dtype=np.float32)
        inner = np.array([[5.0, 0.0]], dtype=np.float32)
        pos = np.array([7.5, 0.0], dtype=np.float32)
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert np.isfinite(throttle)

    def test_two_point_boundaries(self):
        """Two-point line segments."""
        pid = TrackPIDController(noise_std=0.0)
        outer = np.array([[10.0, -5.0], [10.0, 5.0]], dtype=np.float32)
        inner = np.array([[5.0, -5.0], [5.0, 5.0]], dtype=np.float32)
        pos = np.array([7.5, 0.0], dtype=np.float32)
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert np.isfinite(throttle)

    def test_vehicle_on_outer_boundary(self, oval_boundaries):
        """Vehicle sitting exactly on the outer boundary."""
        pid = TrackPIDController(noise_std=0.0)
        outer, inner = oval_boundaries
        pos = outer[0].copy()  # on the boundary
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert -1.0 <= steer <= 1.0

    def test_vehicle_on_inner_boundary(self, oval_boundaries):
        """Vehicle sitting exactly on the inner boundary."""
        pid = TrackPIDController(noise_std=0.0)
        outer, inner = oval_boundaries
        pos = inner[0].copy()
        steer, throttle = pid.step(pos, 0.0, outer, inner)
        assert np.isfinite(steer)
        assert -1.0 <= steer <= 1.0

    def test_reversed_heading(self, oval_boundaries):
        """Vehicle heading backwards → should produce strong correction."""
        pid = TrackPIDController(kp_steer=1.0, noise_std=0.0)
        outer, inner = oval_boundaries
        pos = np.array([6.5, 0.0], dtype=np.float32)
        heading = np.pi  # heading backwards
        steer, _ = pid.step(pos, heading, outer, inner)
        assert np.isfinite(steer)
