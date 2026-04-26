"""
LQR, MPC, depth FTG and the ONNX policy controller.

Policy tests skip without onnxruntime. The export test additionally needs
stable_baselines3, gymnasium and onnx.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from control import SafetyConfig, TrackModel, VehicleParams, VehicleState
from control.depth_ftg import DepthFTGConfig, DepthGapFollower, Intrinsics, depth_to_scan
from control.sim import (
    make_follower_factory, oval_track, render_depth, run_closed_loop, run_depth_closed_loop, twisty_track,
)

P = VehicleParams.load()
K_SIM = Intrinsics(99.0, 98.95, 81.75, 47.25)   # OAK-D RGB intrinsics scaled to 160x100
TRACKS = {"oval": oval_track, "twisty": twisty_track}


def _uid() -> str:
    return uuid.uuid4().hex[:8]


# LQR, MPC

@pytest.mark.parametrize("mode", ["lqr", "mpc"])
@pytest.mark.parametrize("track", sorted(TRACKS))
def test_optimal_controllers_lap_cleanly(mode, track):
    outer, inner, _ = TRACKS[track]()
    r = run_closed_loop(make_follower_factory(mode, P), outer, inner, P, duration=40.0)
    assert r.laps >= 1.0, r.summary()
    assert r.off_track_fraction == 0.0, r.summary()
    assert r.xtrack_max < 0.1, r.summary()
    assert r.max_speed <= SafetyConfig().speed_cap, r.summary()


def test_lqr_gain_scheduling_and_sign():
    from control.lqr import LQR

    lqr = LQR(P)
    k_slow, k_fast = lqr.gain(0.5), lqr.gain(1.5)
    assert k_slow.shape == (1, 2)
    assert np.all(k_slow > 0) and np.all(k_fast > 0)      # e>0 (left of path) -> steer right
    assert lqr.gain(0.0) is lqr.gain(0.3)                 # design-speed floor, cached


def test_mpc_respects_steering_limits():
    from control.mpc import MPC

    outer, inner, _ = oval_track()
    track = TrackModel(outer, inner)
    mpc = MPC(P)
    i = 30
    h = track.heading[i]
    x, y = track.points[i] + 0.7 * np.array([-np.sin(h), np.cos(h)])   # far left of the path
    st = VehicleState(t=0.0, x=x, y=y, yaw=h, v=1.0)
    deltas = [mpc.steer(st, track, track.project(x, y)) for _ in range(5)]
    assert mpc.last_solve_ok
    assert all(abs(d) <= P.max_steering_angle + 1e-6 for d in deltas)
    steps = np.abs(np.diff([0.0] + deltas))
    assert np.all(steps <= P.max_steering_vel * mpc.dt + 1e-4)   # rate constraint
    assert deltas[-1] < 0                                        # steering back (right)


# Depth FTG

class TestDepthScan:

    def test_floor_is_not_an_obstacle(self):
        outer, inner, _ = oval_track(width=6.0)
        # walls pushed way past max_range so only floor is visible
        floor = render_depth(0.0, 0.0, 0.0, [outer * 100, inner * 100], K_SIM, 160, 100)
        _, ranges = depth_to_scan(floor, K_SIM, DepthFTGConfig())
        assert np.all(ranges == DepthFTGConfig().max_range)

    def test_wall_range_and_bearing(self):
        # wall 2 m ahead, full width
        wall = [np.array([[2.0, -5.0], [2.0, 5.0], [2.1, 5.0], [2.1, -5.0]])]
        d = render_depth(0.0, 0.0, 0.0, wall, K_SIM, 160, 100)
        bearings, ranges = depth_to_scan(d, K_SIM, DepthFTGConfig())
        centre = np.argmin(np.abs(bearings))
        assert ranges[centre] == pytest.approx(2.0, abs=0.05)
        assert bearings[0] < 0 < bearings[-1]              # ascending, + = left

    def test_intrinsics_from_settings_file(self):
        K = Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
        assert K.fx > 100 and 0 < K.cx < 640


class TestDepthGapFollower:

    @pytest.mark.parametrize("side", [+1, -1])
    def test_steers_into_the_open_side(self, side):
        # box blocks the centre and one side, other side open
        box = np.array([[1.0, -1.5], [1.0, 0.15], [1.2, 0.15], [1.2, -1.5]]) * [1, side]
        cmd = DepthGapFollower(P, K_SIM).step(render_depth(0.0, 0.0, 0.0, [box], K_SIM, 160, 100))
        assert np.sign(cmd.steering) == side               # + = left

    def test_stops_when_blocked(self):
        wall = [np.array([[0.3, -2.0], [0.3, 2.0], [0.4, 2.0], [0.4, -2.0]])]
        cmd = DepthGapFollower(P, K_SIM).step(render_depth(0.0, 0.0, 0.0, wall, K_SIM, 160, 100))
        assert (cmd.steering, cmd.throttle) == (0.0, 0.0)

    @pytest.mark.parametrize("track", sorted(TRACKS))
    def test_map_free_closed_loop(self, track):
        """Depth only, no pose or map. Should keep moving and stay on track."""
        outer, inner, _ = TRACKS[track]()
        r = run_depth_closed_loop(DepthGapFollower(P, K_SIM), outer, inner, P, duration=60.0)
        assert r.off_track_fraction == 0.0, r.summary()
        assert r.laps > 0.6, r.summary()                   # ~0.5 m/s for 60 s, never stalls
        assert r.xtrack_max < 0.3, r.summary()


def _export_test_policy(tmp_path):
    """Tiny but real SB3 PPO + VecNormalize stats, exported through rl/export_onnx"""
    pytest.importorskip("onnxruntime")
    sb3 = pytest.importorskip("stable_baselines3")
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("onnx")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from rl.export_onnx import export

    n_ahead = 10

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, ((n_ahead + 1) * 4 + 1,), np.float32)
        action_space = gym.spaces.Box(np.array([-1, 0], np.float32), np.array([1, 1], np.float32))

        def reset(self, seed=None, options=None):
            return self.observation_space.sample().clip(-3, 3), {}

        def step(self, a):
            return self.observation_space.sample().clip(-3, 3), 0.0, False, False, {}

    vec = VecNormalize(DummyVecEnv([Dummy]), norm_obs=True, clip_obs=10.0)
    model = sb3.PPO("MlpPolicy", vec, n_steps=32, batch_size=16, device="cpu", verbose=0, seed=0)
    model.learn(64)
    env_kwargs = {"n_ahead": n_ahead, "boundary_spacing": 0.4, "boundary_smooth_window": 5}
    path = export(model, vec, env_kwargs, str(tmp_path / "policy.onnx"))
    return path, model, vec


class TestPolicy:

    def test_observation_matches_training_builder(self):
        pytest.importorskip("onnxruntime")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from control import policy as cp
        from workers import boundary_resampler as br

        outer, inner, _ = oval_track()
        o_rs = cp.resample_uniform(cp.smooth_boundary(outer, 5), 0.4)
        np.testing.assert_allclose(o_rs, br.resample_uniform(br.smooth_boundary(outer, 5), 0.4))
        i_rs = cp.resample_uniform(cp.smooth_boundary(inner, 5), 0.4)
        pos, heading = np.array([6.4, 0.3], np.float32), 1.4
        np.testing.assert_allclose(cp.boundary_obs(pos, heading, o_rs, i_rs, 10),
                                   br.boundary_obs(pos, heading, o_rs, i_rs, 10), atol=1e-6)

    def test_export_matches_sb3_and_follower_runs(self, tmp_path):
        import onnxruntime as ort

        path, model, vec = _export_test_policy(tmp_path)
        obs = (np.random.default_rng(0).normal(size=(8, 45)) * 2).astype(np.float32)
        onnx_a = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"obs": obs})[0]
        sb3_a, _ = model.predict(vec.normalize_obs(obs), deterministic=True)
        np.testing.assert_allclose(onnx_a, sb3_a, atol=1e-5)

        from control.policy import PolicyFollower

        outer, inner, centre = oval_track()
        f = PolicyFollower(path, outer, inner)
        st = VehicleState(t=0.0, x=float(centre[0, 0]), y=float(centre[0, 1]), yaw=np.pi / 2, v=0.5)
        cmd = f.step(st)
        assert -1.0 <= cmd.steering <= 1.0 and 0.0 <= cmd.throttle <= 1.0
        assert f.observation(st).shape == (45,)


# worker wiring

def test_control_worker_depth_ftg_mode():
    from shared_buffer import ControlBuffer, FrameBuffer
    from workers import ControlWorker

    fb = FrameBuffer(slots=2, name=f"cm_f_{_uid()}", create=True)
    cb = ControlBuffer(slots=2, name=f"cm_c_{_uid()}", create=True)
    try:
        w = ControlWorker(name="control", mode="depth_ftg", output_buffer=cb, frame_buffer=fb)
        w._logger = MagicMock()
        w.setup()
        K = Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
        outer, inner, centre = oval_track()
        depth = render_depth(float(centre[0, 0]), float(centre[0, 1]), np.pi / 2, [outer, inner], K, 640, 400)
        fb.write(np.zeros((400, 640, 3), np.uint8), depth.astype(np.float32))
        w.step()
        _, steer_u, thr_u = cb.read_latest()
        assert thr_u > 0.5                    # driving forward along the free track
    finally:
        fb.unlink()
        cb.unlink()


def test_track_mode_falls_back_to_depth_when_state_stale():
    from shared_buffer import ControlBuffer, FrameBuffer, StateBuffer, TrackBoundaryBuffer
    from workers import ControlWorker

    outer, inner, centre = oval_track()
    fb = FrameBuffer(slots=2, name=f"cf_f_{_uid()}", create=True)
    cb = ControlBuffer(slots=2, name=f"cf_c_{_uid()}", create=True)
    sb = StateBuffer(slots=2, name=f"cf_s_{_uid()}", create=True)
    bb = TrackBoundaryBuffer(slots=2, name=f"cf_b_{_uid()}", create=True)
    try:
        bb.write(outer.astype(np.float32), inner.astype(np.float32))
        w = ControlWorker(name="control", mode="lqr", output_buffer=cb, state_buffer=sb, boundary_buffer=bb,
                          frame_buffer=fb, track_config={"fallback": "depth_ftg", "fallback_after": 0.05})
        w._logger = MagicMock()
        w.setup()
        time.sleep(0.1)                                   # no state at all -> stale
        K = Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
        depth = render_depth(float(centre[0, 0]), float(centre[0, 1]), np.pi / 2, [outer, inner], K, 640, 400)
        fb.write(np.zeros((400, 640, 3), np.uint8), depth.astype(np.float32))
        w.step()
        assert w._fallback_active
        assert cb.read_latest() is not None
        # fresh state should switch back to the track controller
        sb.write(VehicleState(t=time.monotonic(), x=float(centre[0, 0]), y=float(centre[0, 1]), yaw=np.pi / 2, v=0.5))
        w.step()
        assert not w._fallback_active and w._follower is not None
    finally:
        for b in (fb, cb, sb, bb):
            b.unlink()
