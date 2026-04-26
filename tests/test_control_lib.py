"""Unit tests for the pure-Python control/ package."""

from __future__ import annotations

import numpy as np
import pytest

from control import (
    Command, PoseEKF, PurePursuit, SafetyConfig, SafetySupervisor, Stanley,
    TrackModel, VehicleParams, VehicleState, build_track_follower,
)
from control.sim import oval_track, twisty_track
from control.speed import LongitudinalConfig, LongitudinalController, speed_profile

P = VehicleParams.load()


@pytest.fixture(scope="module")
def oval():
    outer, inner, centre = oval_track(a=6.5, b=4.0, width=1.6)
    return TrackModel(outer, inner), centre


def test_vehicle_params_loaded_from_yaml():
    assert P.wheelbase == pytest.approx(0.3302)
    assert P.max_steering_angle == pytest.approx(0.4189)


class TestTrackModel:

    def test_centre_line_and_width(self, oval):
        track, centre = oval
        # Ramanujan approx of the ellipse perimeter, 6.5 x 4.0 centre line
        a, b = 6.5, 4.0
        h = ((a - b) / (a + b)) ** 2
        perimeter = np.pi * (a + b) * (1 + 3 * h / (10 + np.sqrt(4 - 3 * h)))
        assert track.length == pytest.approx(perimeter, rel=0.02)
        assert np.median(track.half_width) == pytest.approx(0.8, abs=0.05)

    def test_curvature_matches_ellipse(self, oval):
        track, _ = oval
        # ellipse curvature peaks at the ends of the major axis (a/b^2), min is b/a^2
        assert np.abs(track.curvature).max() == pytest.approx(6.5 / 4.0 ** 2, rel=0.1)
        assert np.abs(track.curvature).min() == pytest.approx(4.0 / 6.5 ** 2, rel=0.1)

    def test_projection_lateral_sign(self, oval):
        track, _ = oval
        i = 0
        p0, h = track.points[i], track.heading[i]
        left = np.array([-np.sin(h), np.cos(h)])
        pr = track.project(*(p0 + 0.3 * left))
        assert pr.lateral_error == pytest.approx(0.3, abs=0.02)
        pr = track.project(*(p0 - 0.3 * left), hint=pr.idx)
        assert pr.lateral_error == pytest.approx(-0.3, abs=0.02)

    def test_windowed_projection_recovers_from_jump(self, oval):
        track, _ = oval
        far = track.points[len(track.points) // 2]
        pr = track.project(*far, hint=0)
        assert np.linalg.norm(track.points[pr.idx] - far) < 0.1

    def test_align_direction(self, oval):
        track, _ = oval
        x, y = track.points[10]
        backwards = track.heading[10] + np.pi
        assert track.align_direction(x, y, backwards) is True
        assert abs(np.cos(track.project(x, y).path_heading - backwards)) == pytest.approx(1.0, abs=1e-2)
        assert track.align_direction(x, y, backwards) is False

    def test_point_at_wraps(self, oval):
        track, _ = oval
        np.testing.assert_allclose(track.point_at(track.length + 0.2), track.point_at(0.2), atol=1e-9)


class TestSpeed:

    def test_profile_respects_lateral_and_accel_limits(self):
        outer, inner, _ = twisty_track()
        track = TrackModel(outer, inner)
        a_lat, acc, dec = 2.0, 1.0, 1.0
        v = speed_profile(track, v_max=5.0, a_lat_max=a_lat, a_accel=acc, a_decel=dec)
        assert np.all(v <= 5.0 + 1e-9)
        assert np.all(v ** 2 * np.abs(track.curvature) <= a_lat + 1e-6)
        ds = np.diff(np.concatenate([track.s, [track.length]]))
        dv2 = (np.roll(v, -1) ** 2 - v ** 2) / ds
        assert dv2.max() <= 2 * acc + 1e-6 and dv2.min() >= -2 * dec - 1e-6
        assert v.min() < 5.0  # the corners actually bind

    def test_longitudinal_never_reverses_from_standstill(self):
        c = LongitudinalController(LongitudinalConfig())
        assert c.step(v_ref=0.0, v=0.1, dt=0.03) >= 0.0

    def test_longitudinal_uses_backward_input_to_slow_down(self):
        c = LongitudinalController(LongitudinalConfig(max_backward=0.3))
        u = c.step(v_ref=0.2, v=2.0, dt=0.03)
        assert -0.3 <= u < 0.0


# lateral controllers

@pytest.mark.parametrize("lateral_cls", [PurePursuit, Stanley])
class TestLateral:

    def _state_offset(self, track, offset, yaw_err=0.0):
        i = 50
        p, h = track.points[i], track.heading[i]
        left = np.array([-np.sin(h), np.cos(h)])
        x, y = p + offset * left
        return VehicleState(t=0.0, x=x, y=y, yaw=h + yaw_err, v=1.0)

    def test_on_line_and_aligned_is_near_straight(self, oval, lateral_cls):
        track, _ = oval
        st = self._state_offset(track, 0.0)
        delta = lateral_cls(P).steer(st, track, track.project(st.x, st.y))
        # only path curvature left, so expect the Ackermann angle atan(L*kappa)
        kappa = track.curvature[50]
        assert delta == pytest.approx(np.arctan(P.wheelbase * kappa), abs=0.05)

    @pytest.mark.parametrize("offset, left_turn", [(-0.4, True), (0.4, False)])
    def test_steers_back_to_line(self, oval, lateral_cls, offset, left_turn):
        track, _ = oval
        st = self._state_offset(track, offset)
        delta = lateral_cls(P).steer(st, track, track.project(st.x, st.y))
        kappa_delta = np.arctan(P.wheelbase * track.curvature[50])
        assert (delta > kappa_delta) == left_turn   # + = left


def test_build_track_follower_rejects_unknown(oval):
    with pytest.raises(ValueError):
        build_track_follower("nope", oval[0], P)


class TestEstimator:

    def test_infers_speed_and_yaw_rate_from_poses(self):
        ekf = PoseEKF()
        v, w = 1.2, 0.3
        x = y = yaw = 0.0
        for k in range(60):                     # 4 s at 15 Hz
            t = k / 15.0
            ekf.update_pose(t, x, y, yaw)
            x += v * np.cos(yaw) / 15.0
            y += v * np.sin(yaw) / 15.0
            yaw += w / 15.0
        st = ekf.predict_to(59 / 15.0)
        assert st.v == pytest.approx(v, abs=0.05)
        assert st.yaw_rate == pytest.approx(w, abs=0.05)

    def test_goes_stale_without_poses(self):
        ekf = PoseEKF()
        ekf.update_pose(0.0, 0.0, 0.0, 0.0)
        assert ekf.predict_to(0.4) is not None
        assert ekf.predict_to(0.6) is None

    def test_yaw_wraps_across_pi(self):
        ekf = PoseEKF()
        for k, yaw in enumerate([3.10, 3.13, -3.13, -3.10]):
            ekf.update_pose(k * 0.1, 0.0, 0.0, yaw)
        assert ekf.predict_to(0.3).yaw_rate > 0


class TestSafety:

    def test_disarmed_is_neutral(self):
        sup = SafetySupervisor()
        out = sup.step(0.0, Command(0.5, 0.8), 0.0, armed=False)
        assert (out.steering, out.throttle) == (0.0, 0.0)

    def test_disarm_is_immediate(self):
        sup = SafetySupervisor(SafetyConfig(throttle_rate=100, steering_rate=100))
        for k in range(10):
            sup.step(k * 0.02, Command(0.5, 0.4), k * 0.02, armed=True)
        out = sup.step(0.22, Command(0.5, 0.4), 0.22, armed=False)
        assert (out.steering, out.throttle) == (0.0, 0.0)

    def test_watchdog_neutral_throttle_holds_steering(self):
        sup = SafetySupervisor(SafetyConfig(throttle_rate=100, steering_rate=100))
        for k in range(10):
            sup.step(k * 0.02, Command(0.3, 0.4), k * 0.02, armed=True)
        out = sup.step(0.18 + 0.151, Command(0.3, 0.4), 0.18, armed=True)
        assert out.throttle == 0.0
        assert out.steering == pytest.approx(0.3)
        assert sup.status.watchdog_tripped

    def test_speed_cap_cuts_forward_throttle(self):
        sup = SafetySupervisor(SafetyConfig(speed_cap=1.5, throttle_rate=100))
        sup.step(0.0, Command(0.0, 0.5), 0.0, armed=True, speed=1.0)
        out = sup.step(0.02, Command(0.0, 0.5), 0.02, armed=True, speed=1.6)
        assert out.throttle == 0.0 and sup.status.speed_limited
        # backward input (braking) still allowed above the cap
        out = sup.step(0.04, Command(0.0, -0.2), 0.04, armed=True, speed=1.6)
        assert out.throttle < 0.0

    def test_throttle_capped_without_speed_estimate(self):
        sup = SafetySupervisor(SafetyConfig(max_throttle_no_speed=0.4, throttle_rate=100))
        for k in range(5):
            out = sup.step(k * 0.02, Command(0.0, 1.0), k * 0.02, armed=True, speed=None)
        assert out.throttle == pytest.approx(0.4)

    def test_slew_limits(self):
        sup = SafetySupervisor(SafetyConfig(steering_rate=5.0, throttle_rate=2.0))
        sup.step(0.0, Command(0.0, 0.0), 0.0, armed=True)
        out = sup.step(0.02, Command(1.0, 1.0), 0.02, armed=True, speed=0.0)
        assert out.steering == pytest.approx(0.1)
        assert out.throttle == pytest.approx(0.04)

    def test_dropping_to_neutral_is_not_slew_limited(self):
        sup = SafetySupervisor(SafetyConfig(throttle_rate=100))
        for k in range(5):
            sup.step(k * 0.02, Command(0.0, 0.6), k * 0.02, armed=True, speed=0.5)
        sup.cfg.throttle_rate = 0.1
        out = sup.step(0.12, Command(0.0, 0.0), 0.12, armed=True, speed=0.5)
        assert out.throttle == 0.0

    def test_non_finite_speed_counts_as_no_estimate(self):
        sup = SafetySupervisor(SafetyConfig(max_throttle_no_speed=0.4, throttle_rate=100))
        for k in range(5):
            out = sup.step(k * 0.02, Command(0.0, 1.0), k * 0.02, armed=True, speed=float("nan"))
        assert out.throttle == pytest.approx(0.4)

    @pytest.mark.parametrize("command", [Command(float("nan"), 0.3), Command(0.0, float("inf"))])
    def test_non_finite_command_is_stale(self, command):
        sup = SafetySupervisor(SafetyConfig(throttle_rate=100))
        out = sup.step(0.0, command, 0.0, armed=True, speed=0.0)
        assert out.throttle == 0.0 and sup.status.watchdog_tripped

    def test_future_stamped_command_is_stale(self):
        sup = SafetySupervisor(SafetyConfig(throttle_rate=100))
        out = sup.step(0.0, Command(0.0, 0.3), 5.0, armed=True, speed=0.0)
        assert out.throttle == 0.0 and sup.status.watchdog_tripped

    def test_reverse_throttle_is_capped(self):
        sup = SafetySupervisor(SafetyConfig(max_reverse_throttle=0.3, throttle_rate=100))
        for k in range(5):
            out = sup.step(k * 0.02, Command(0.0, -1.0), k * 0.02, armed=True, speed=0.0)
        assert out.throttle == pytest.approx(-0.3) and sup.status.speed_limited
