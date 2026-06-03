"""rl/bicycle_env.BicycleTrackEnv: gymnasium API, observation parity with the on-car
policy path (control/policy.PolicyFollower) and with TrackEnv's spaces, termination on
leaving the track, lap counting, and a pretrained-style checkpoint that rl/train.py resumes."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from control.boundary import obs_dim  # noqa: E402
from control.safety import SafetyConfig  # noqa: E402
from control.sim import oval_track  # noqa: E402
from rl.bicycle_env import DEFAULT_MAPS, BicycleTrackEnv, generate_track, load_map  # noqa: E402

N_AHEAD, SPACING, WINDOW = 20, 0.5, 30
SAFETY = {"speed_cap": 2.0, "speed_per_throttle": 24.8, "throttle_rate": 0.5,
          "max_throttle_no_speed": 0.04, "max_reverse_throttle": 0.06}


def _env(**kw) -> BicycleTrackEnv:
    base = dict(n_ahead=N_AHEAD, boundary_spacing=SPACING, boundary_smooth_window=WINDOW,
                max_steps=200, policy_hz=20.0, safety=SAFETY)
    base.update(kw)
    return BicycleTrackEnv(**base)


def test_env_checker_passes():
    from gymnasium.utils.env_checker import check_env

    env = _env()
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (obs_dim(N_AHEAD),)
    np.testing.assert_array_equal(env.action_space.low, [-1.0, 0.0])
    np.testing.assert_array_equal(env.action_space.high, [1.0, 1.0])


def test_accepts_train_env_kwargs():
    """rl/train.py's ENV_KWARGS (TrackEnv kwargs, incl. sim-only ones) build the env."""
    pytest.importorskip("stable_baselines3")
    from rl import train

    env = BicycleTrackEnv(**train.resolve_env_kwargs(train.ENV_KWARGS, asdict(SafetyConfig(**SAFETY))))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (obs_dim(train.ENV_KWARGS["n_ahead"]),)
    with pytest.raises(TypeError):
        BicycleTrackEnv(not_a_kwarg=1)


def test_generated_tracks_are_valid():
    rng = np.random.default_rng(3)
    for _ in range(5):
        outer, inner = generate_track(rng)
        from control.track import TrackModel
        tm = TrackModel(outer, inner)
        assert 20 < tm.length < 70
        assert 0.5 < np.median(tm.half_width) < 0.9


def _tiny_policy(tmp_path):
    pytest.importorskip("onnxruntime")
    from tests.test_boundary_obs import _tiny_policy as make

    return make(tmp_path, N_AHEAD, SPACING, WINDOW)


@pytest.mark.parametrize("map_dir", list(DEFAULT_MAPS) + [None])
def test_observation_parity_with_policy_follower(tmp_path, map_dir):
    """The env's observation of a pose == what the on-car PolicyFollower computes for it."""
    from control.policy import PolicyFollower
    from control.types import VehicleState

    path = _tiny_policy(tmp_path)
    if map_dir is None:
        outer, inner, _ = oval_track()
        outer, inner = outer.astype(np.float32), inner.astype(np.float32)
    else:
        outer, inner = load_map(map_dir)
    env = _env(obs_noise=(0.0, 0.0, 0.0))
    env.load_boundaries(outer, inner)
    follower = PolicyFollower(path, outer, inner)
    np.testing.assert_array_equal(follower.outer, env._outer)
    np.testing.assert_array_equal(follower.inner, env._inner)

    rng = np.random.default_rng(1)
    centre = env.track.model.points
    for k in list(range(0, len(centre), max(len(centre) // 15, 1))) + [0, len(centre) - 1]:
        x, y = centre[k] + rng.normal(0, 0.05, 2)
        yaw, v = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(0, 2))
        st = VehicleState(t=0.0, x=float(x), y=float(y), yaw=yaw, v=v)
        np.testing.assert_allclose(follower.observation(st), env._build_obs(np.array([x, y], np.float32), yaw, v),
                                   atol=1e-5)

    # observations the env actually emits (noise off) are the follower's for the true pose
    obs, _ = env.reset(seed=0, options={"start": "random"})
    for _ in range(30):
        x, y, yaw, v = env.true_pose()
        st = VehicleState(t=0.0, x=x, y=y, yaw=yaw, v=v)
        if not follower._aligned:
            follower._align(st)
        np.testing.assert_allclose(obs, follower.observation(st), atol=1e-5)
        obs, *_ = env.step(np.array([0.0, 1.0], np.float32))


def test_terminates_on_leaving_track():
    env = _env()
    outer, inner, _ = oval_track()
    env.load_boundaries(outer, inner)
    env.reset(seed=0)
    for k in range(400):
        obs, r, term, trunc, info = env.step(np.array([-1.0, 1.0], np.float32))   # full right
        if term or trunc:
            break
    assert term and info["collided"] and not info["lap_completed"]
    assert r < -15.0                       # TrackEnv's -20 collision penalty
    assert k < 100


def test_speed_capped_like_actuator_and_stuck():
    env = _env(randomize_dynamics=False, obs_noise=(0, 0, 0))
    env.load_boundaries(*oval_track(a=30.0, b=20.0, width=4.0)[:2])
    env.reset(seed=0)
    speeds = [env.step(np.array([0.0, 1.0], np.float32))[4]["speed"] for _ in range(40)]
    assert 1.8 < max(speeds) <= 2.05      # safety speed_cap 2.0 (throttle bound 2.0 / 24.8)
    # zero throttle: the car stops and is declared stuck after the grace period
    for _ in range(60):
        obs, r, term, trunc, info = env.step(np.array([0.0, 0.0], np.float32))
        if term:
            break
    assert term and info["stuck"]


def test_pure_pursuit_completes_a_lap_on_practice_map():
    from rl.eval_bicycle import make_eval_env, pure_pursuit_actor, run_episodes

    env = make_eval_env(dict(n_ahead=N_AHEAD, boundary_spacing=SPACING, boundary_smooth_window=WINDOW,
                             policy_hz=20.0, safety=SAFETY), max_steps=400)
    act, reset = pure_pursuit_actor(DEFAULT_MAPS[0])
    r = run_episodes(env, act, DEFAULT_MAPS[0], 1, reset_fn=reset)
    assert r["lap_rate"] == 1.0 and r["collisions"] == 0
    assert 14.0 < r["lap_time_mean"] < 20.0    # 31 m at <= 2 m/s


def test_pretrain_checkpoint_resumes_with_train_py(tmp_path, monkeypatch):
    """A run written like rl/pretrain_bicycle.py is loadable by train.py's resume path."""
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from rl import train
    from rl.checkpoints import find_checkpoint

    env_kwargs = dict(n_ahead=5, boundary_spacing=SPACING, boundary_smooth_window=WINDOW,
                      max_steps=50, policy_hz=20.0, sim_host="0.0.0.0", sim_port=4567)
    env_kwargs = train.resolve_env_kwargs(env_kwargs, asdict(SafetyConfig(**SAFETY)))
    vec = VecNormalize(DummyVecEnv([lambda: BicycleTrackEnv(**env_kwargs)] * 2), clip_obs=10.0, clip_reward=10.0)
    model = sb3.PPO(env=vec, **dict(train.PPO_KWARGS, n_steps=32, batch_size=32, verbose=0))
    model.learn(64)
    run = tmp_path / "PPO_1"
    run.mkdir()
    model.save(str(run / "ppo_track_64_steps"))
    vec.save(str(run / "vec_normalize_64.pkl"))
    with open(run / "config.json", "w") as f:
        json.dump({"ENV_KWARGS": env_kwargs, "POLICY_KWARGS": train.POLICY_KWARGS}, f)

    # train.py's resume: TrackEnv(**ENV_KWARGS) is swapped for the bicycle env (no sim)
    saved = train._load_config(str(run))["ENV_KWARGS"]
    path, vec_path = find_checkpoint(str(run), None)
    live = VecNormalize.load(vec_path, DummyVecEnv([lambda: BicycleTrackEnv(**saved)]))
    monkeypatch.setitem(train.PPO_KWARGS, "n_steps", 32)
    monkeypatch.setitem(train.PPO_KWARGS, "batch_size", 16)
    monkeypatch.setitem(train.PPO_KWARGS, "verbose", 0)
    resumed = train.load_resumed_model(path, live)
    assert resumed.num_timesteps == 64
    assert resumed.policy.net_arch == model.policy.net_arch
    resumed.learn(32, reset_num_timesteps=False)
    assert resumed.num_timesteps > 64
