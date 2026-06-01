"""control/boundary.py: closed-loop smoothing/resampling, and observation
parity between training (rl/track_env.TrackEnv) and deployment
(control/policy.PolicyFollower)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from control.boundary import boundary_obs, obs_dim, prepare_boundary, resample_uniform, smooth_boundary
from control.sim import oval_track

A, B = 6.0, 3.0   # ellipse semi-axes, m


def _ellipse(n: int, phase: float = 0.0) -> np.ndarray:
    t = phase + np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return np.stack([A * np.cos(t), B * np.sin(t)], axis=1)


def _dist_to_ellipse(pts: np.ndarray) -> np.ndarray:
    dense = _ellipse(200_000)
    return np.array([np.linalg.norm(dense - p, axis=1).min() for p in pts])


def test_smoothed_closed_ellipse_stays_on_curve_including_seam():
    pts = _ellipse(1000)                               # ~3 cm spacing
    noisy = pts + np.random.default_rng(0).normal(0, 0.01, pts.shape)
    sm = smooth_boundary(noisy, 15)
    assert sm.shape == pts.shape
    d = _dist_to_ellipse(sm)
    assert d.max() < 0.03, d.max()
    # the seam (first/last window) specifically: zero padding pulled these metres off
    seam = np.r_[0:15, len(sm) - 15:len(sm)]
    assert d[seam].max() < 0.03
    np.testing.assert_allclose(sm[0], pts[0], atol=0.03)


def test_even_window_and_duplicate_closing_point():
    pts = _ellipse(600)
    closed = np.vstack([pts, pts[:1]])                  # explicit closing point
    sm = smooth_boundary(closed, 30)                    # even -> 31, centred
    assert len(sm) == len(pts)
    assert _dist_to_ellipse(sm).max() < 0.05


def test_resample_uniform_spacing_across_seam():
    pts = _ellipse(357, phase=0.3)
    rs = resample_uniform(pts, 0.4)
    gaps = np.linalg.norm(np.diff(np.vstack([rs, rs[:1]]), axis=0), axis=1)   # incl. last -> first
    assert abs(gaps.mean() - 0.4) < 0.01
    assert gaps.max() - gaps.min() < 0.01, (gaps.min(), gaps.max())
    np.testing.assert_allclose(rs[0], pts[0], atol=1e-5)
    assert _dist_to_ellipse(rs).max() < 0.01


def test_prepare_boundary_on_track_and_obs_layout():
    outer, inner, _ = oval_track()
    o = prepare_boundary(outer, 30, 0.5)
    i = prepare_boundary(inner, 30, 0.5)
    obs = boundary_obs(np.array([0.0, 0.0], np.float32), 0.0, o, i, 20)
    assert obs.shape == (obs_dim(20) - 1,) and obs.dtype == np.float32


def _tiny_policy(tmp_path, n_ahead, spacing, window):
    gym = pytest.importorskip("gymnasium")
    sb3 = pytest.importorskip("stable_baselines3")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from rl.export_onnx import export

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_dim(n_ahead),), np.float32)
        action_space = gym.spaces.Box(np.array([-1, 0], np.float32), np.array([1, 1], np.float32))

        def reset(self, seed=None, options=None):
            return np.zeros(obs_dim(n_ahead), np.float32), {}

        def step(self, a):
            return np.zeros(obs_dim(n_ahead), np.float32), 0.0, False, False, {}

    model = sb3.PPO("MlpPolicy", Dummy(), n_steps=32, batch_size=16, device="cpu", verbose=0, seed=0)
    kwargs = {"n_ahead": n_ahead, "boundary_spacing": spacing, "boundary_smooth_window": window,
              "policy_hz": 20.0}
    return export(model, None, kwargs, str(tmp_path / "p.onnx"))


def test_observation_parity_training_vs_deployment(tmp_path):
    pytest.importorskip("websockets")           # sim extra (TrackEnv's AutoDRIVEWorker)
    pytest.importorskip("gymnasium")            # rl extra
    from control.policy import PolicyFollower
    from control.types import VehicleState
    from rl.track_env import TrackEnv

    n_ahead, spacing, window = 20, 0.5, 30
    path = _tiny_policy(tmp_path, n_ahead, spacing, window)
    outer, inner, centre = oval_track()

    env = TrackEnv(n_ahead=n_ahead, boundary_spacing=spacing, boundary_smooth_window=window,
                   shm_prefix=f"t_{uuid.uuid4().hex[:6]}")
    try:
        env.load_boundaries(outer, inner)
        follower = PolicyFollower(path, outer, inner)
        rng = np.random.default_rng(1)
        # include points near the start of the boundary arrays (the seam)
        for k in list(range(0, len(centre), max(len(centre) // 12, 1))) + [0, len(centre) - 1]:
            x, y = centre[k] + rng.normal(0, 0.05, 2)
            yaw, v = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0, 2))
            st = VehicleState(t=0.0, x=float(x), y=float(y), yaw=yaw, v=v)
            train_obs = env._build_obs(np.array([x, y], np.float32), yaw, v)
            np.testing.assert_allclose(follower.observation(st), train_obs, atol=1e-5)
        assert train_obs.shape == env.observation_space.shape
    finally:
        env.close()
