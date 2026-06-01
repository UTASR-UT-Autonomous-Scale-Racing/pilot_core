"""
Closed-loop acceptance runs. Each track controller drives the kinematic bicycle
bench (control/sim.py) through estimator -> controller -> safety, fed 15 Hz
noisy, delayed poses.
"""

from __future__ import annotations

import numpy as np
import pytest

from control import SafetyConfig, VehicleParams
from control.sim import make_follower_factory, oval_track, run_closed_loop, twisty_track

P = VehicleParams.load()
TRACKS = {"oval": oval_track, "twisty": twisty_track}


@pytest.mark.parametrize("mode", ["pure_pursuit", "stanley"])
@pytest.mark.parametrize("track", sorted(TRACKS))
def test_controller_laps_cleanly(mode, track):
    outer, inner, _ = TRACKS[track]()
    r = run_closed_loop(make_follower_factory(mode, P), outer, inner, P, duration=40.0)
    assert r.laps >= 1.0, r.summary()
    assert r.off_track_fraction == 0.0, r.summary()
    assert r.xtrack_max < 0.25, r.summary()
    assert r.max_speed <= SafetyConfig().speed_cap, r.summary()


def test_speed_cap_holds_with_aggressive_planner():
    # planner asks for 4 m/s, safety cap should still win
    outer, inner, _ = oval_track()
    fast = make_follower_factory("pure_pursuit", P, speed_cfg={"v_max": 4.0})
    r = run_closed_loop(fast, outer, inner, P, duration=20.0)
    # the steady-state throttle bound is what holds the cap here, pose-derived speed lags
    assert r.max_speed <= SafetyConfig().speed_cap, r.summary()


def test_watchdog_stops_car_when_controller_stops():
    """No commands after 5 s: watchdog goes neutral and the car coasts to a stop."""
    base = make_follower_factory("pure_pursuit", P)

    def factory(outer, inner):
        follower = base(outer, inner)
        step = follower.step
        follower.step = lambda state: step(state) if state.t < 5.0 else None
        return follower

    outer, inner, _ = oval_track()
    r = run_closed_loop(factory, outer, inner, P, duration=10.0)
    t, v, throttle = r.log[:, 0], r.log[:, 3], r.log[:, 5]
    assert v[t < 5.0].max() > 0.8                        # it was driving
    assert (throttle[t > 5.0 + 0.2] == 0.0).all()        # watchdog -> neutral
    assert abs(v[-1]) < 0.05                             # coasted to a stop


def test_score_trajectory_on_a_driven_lap():
    """Two laps along the oval's centre-line, started mid-track, speed from positions."""
    from control.sim import score_trajectory

    outer, inner, centre = oval_track()
    pts = np.vstack([np.roll(centre, -100, axis=0)] * 2 + [centre[100:110]])
    seg = np.hypot(*np.diff(np.vstack([pts, pts[:1]]), axis=0).T)[:-1]
    t = np.concatenate([[0.0], np.cumsum(seg / 1.2)])          # 1.2 m/s
    r = score_trajectory(np.column_stack([t, pts]), outer, inner, P)
    assert len(r.lap_times) == 2 and r.laps == pytest.approx(2.0, abs=0.05)
    assert r.lap_times[0] == pytest.approx(t[400], rel=0.02)
    assert r.xtrack_rms < 0.02 and r.off_track_fraction == 0.0
    assert r.mean_speed == pytest.approx(1.2, rel=0.02)
