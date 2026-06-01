"""Offline checks of rl/train.py merge/resume, rl/checkpoints.py and
rl/export_onnx.py (no simulator: tiny PPO models on a dummy env)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict

import numpy as np
import pytest
import yaml

gym = pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from control.safety import SafetyConfig  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
N_AHEAD = 5
OBS = (N_AHEAD + 1) * 4 + 1


class Dummy(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS,), np.float32)
    action_space = gym.spaces.Box(np.array([-1, 0], np.float32), np.array([1, 1], np.float32))

    def __init__(self):
        self._rng = np.random.default_rng(0)

    def reset(self, seed=None, options=None):
        return self._rng.normal(size=OBS).astype(np.float32), {}

    def step(self, a):
        return self._rng.normal(size=OBS).astype(np.float32), float(a[1]), False, False, {}


def _venv():
    return DummyVecEnv([Dummy])


def _make_run(root, name, step, seed, ppo_kwargs):
    """A run dir like train.py writes: step checkpoint + vecnormalize + config.json."""
    run = os.path.join(root, name)
    os.makedirs(run)
    vec = VecNormalize(_venv(), norm_obs=True, norm_reward=True)
    model = sb3.PPO("MlpPolicy", vec, n_steps=32, batch_size=16, n_epochs=1, device="cpu",
                    seed=seed, verbose=0, **ppo_kwargs)
    model.learn(32)
    model.save(os.path.join(run, f"ppo_track_{step}_steps"))
    vec.save(os.path.join(run, f"vec_normalize_{step}.pkl"))
    env_kwargs = {"n_ahead": N_AHEAD, "boundary_spacing": 0.5, "boundary_smooth_window": 30,
                  "policy_hz": 20.0, "safety": asdict(SafetyConfig())}
    with open(os.path.join(run, "config.json"), "w") as f:
        json.dump({"ENV_KWARGS": env_kwargs, "PPO_KWARGS": ppo_kwargs}, f)
    return run, model


def test_find_checkpoint(tmp_path):
    from rl.checkpoints import find_checkpoint, step_from_path

    run = tmp_path / "PPO_1"
    run.mkdir()
    with pytest.raises(FileNotFoundError):
        find_checkpoint(str(run), None)
    for s in (10000, 9000, 20000):
        (run / f"ppo_track_{s}_steps.zip").write_bytes(b"")
        (run / f"vec_normalize_{s}.pkl").write_bytes(b"")
    m, v = find_checkpoint(str(run), None)
    assert m.endswith("ppo_track_20000_steps.zip") and v.endswith("vec_normalize_20000.pkl")
    assert step_from_path(m) == 20000
    assert find_checkpoint(str(run), 9000)[0].endswith("ppo_track_9000_steps.zip")
    (run / "vec_normalize_20000.pkl").unlink()
    with pytest.raises(FileNotFoundError, match="VecNormalize"):
        find_checkpoint(str(run), None)
    assert step_from_path("x/ppo_track_final.zip") == 0


def test_merge_path_trains(tmp_path):
    import torch

    from rl import train
    from rl.checkpoints import find_checkpoint

    pol_dir, pol_model = _make_run(tmp_path, "PPO_1", 3000, 0,
                                   dict(learning_rate=1e-3, clip_range=0.3, clip_range_vf=None, vf_coef=0.4))
    val_dir, val_model = _make_run(tmp_path, "PPO_2", 5000, 1,
                                   dict(learning_rate=5e-4, clip_range=0.1, clip_range_vf=0.25, vf_coef=0.7))
    pol_path, pol_vec = find_checkpoint(pol_dir, None)
    val_path, _ = find_checkpoint(val_dir, None)

    env = VecNormalize.load(pol_vec, _venv())
    model = train.load_merged_model(pol_path, val_path, train._load_config(pol_dir),
                                    train._load_config(val_dir), env)
    assert callable(model.clip_range) and model.clip_range(1.0) == pytest.approx(0.3)
    assert callable(model.clip_range_vf) and model.clip_range_vf(1.0) == pytest.approx(0.25)
    assert model.vf_coef == pytest.approx(0.7)
    assert model.lr_schedule(1.0) == pytest.approx(1e-3)
    assert model.num_timesteps == 5000

    sd, psd, vsd = (m.policy.state_dict() for m in (model, pol_model, val_model))
    for k in sd:
        src = vsd if (k.startswith("mlp_extractor.value_net") or k.startswith("value_net.")) else psd
        assert torch.equal(sd[k].cpu(), src[k].cpu()), k

    # this is where SB3 used to crash on a float clip_range
    model.learn(32, reset_num_timesteps=False)
    assert model.num_timesteps > 5000


def test_resume_path_trains(tmp_path, monkeypatch):
    from rl import train
    from rl.checkpoints import find_checkpoint

    run, _ = _make_run(tmp_path, "PPO_1", 4000, 0, dict(learning_rate=1e-3, clip_range=0.3))
    path, vec_path = find_checkpoint(run, None)
    monkeypatch.setitem(train.PPO_KWARGS, "n_steps", 32)
    monkeypatch.setitem(train.PPO_KWARGS, "batch_size", 16)
    monkeypatch.setitem(train.PPO_KWARGS, "n_epochs", 1)
    monkeypatch.setitem(train.PPO_KWARGS, "verbose", 0)
    model = train.load_resumed_model(path, VecNormalize.load(vec_path, _venv()))
    assert model.num_timesteps == 4000
    assert model.clip_range(1.0) == pytest.approx(train.PPO_KWARGS["clip_range"])
    model.learn(32, reset_num_timesteps=False)
    assert model.num_timesteps > 4000


def test_config_and_safety_helpers(tmp_path):
    from rl import train

    safety = train.safety_from_pilot_config(os.path.join(REPO, "config_autodrive.yaml"))
    assert safety == asdict(SafetyConfig(**yaml.safe_load(open(os.path.join(REPO, "config_autodrive.yaml")))["safety"]))
    assert train.safety_from_pilot_config(None) == asdict(SafetyConfig())
    cfg = tmp_path / "c.yaml"
    cfg.write_text("safety: {speed_cap: 2.0}\ntrack_control: {speed: {longitudinal: {speed_per_throttle: 4.0}}}\n")
    s2 = train.safety_from_pilot_config(str(cfg))
    assert s2["speed_cap"] == 2.0 and s2["speed_per_throttle"] == 4.0

    kw = train.resolve_env_kwargs({"n_ahead": 3, "vis_zmq_port": 5558}, safety)
    assert kw == {"n_ahead": 3, "safety": safety}
    train._save_config(str(tmp_path), kw)
    assert train._load_config(str(tmp_path))["ENV_KWARGS"] == kw


def test_export_cli_without_train_import(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    run, model = _make_run(tmp_path, "PPO_1", 2000, 0, {})
    out = tmp_path / "policy.onnx"
    code = (
        "import sys, runpy\n"
        f"sys.argv = ['export_onnx.py', '--model', {os.path.join(run, 'ppo_track_2000_steps.zip')!r}, "
        f"'--vecnormalize', {os.path.join(run, 'vec_normalize_2000.pkl')!r}, "
        f"'--config', {os.path.join(run, 'config.json')!r}, '--out', {str(out)!r}]\n"
        "try:\n"
        "    runpy.run_module('rl.export_onnx', run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    assert not e.code, e.code\n"
        "assert 'rl.train' not in sys.modules and 'rl.track_env' not in sys.modules, 'export imported train'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    meta = json.loads((tmp_path / "policy.json").read_text())
    assert meta["policy_hz"] == 20.0
    assert meta["safety"] == asdict(SafetyConfig())
    assert meta["observation"]["obs_dim"] == OBS
    a = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(
        None, {"obs": np.zeros((1, OBS), np.float32)})[0]
    assert a.shape == (1, 2)
