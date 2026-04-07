"""
rl/train.py — PPO training script for the AutoDRIVE track environment.

Usage
-----
    # Start a fresh run (auto-numbered PPO_1, PPO_2, ...):
    python rl/train.py

    # Resume a run from its latest checkpoint:
    python rl/train.py --resume PPO_1

    # Resume from a specific step:
    python rl/train.py --resume PPO_1 --step 50000

    # Monitor all resumes of a run together:
    tensorboard --logdir rl/logs/PPO_1

    # Start a fresh run but transplant policy weights from PPO_N
    # (keeps learned steering/throttle behavior, resets value function):
    python rl/train.py --policy-from PPO_5

    # Start a fresh run transplanting policy from one run and value from another:
    python rl/train.py --policy-from PPO_5 --value-from PPO_5

    # Start a fresh run with policy from a specific step:
    python rl/train.py --policy-from PPO_5 --step 75000

    # Typical use after reward function change — keep learned behavior, fresh value:
    python rl/train.py --policy-from PPO_5

Requirements
------------
    pip install stable-baselines3 tensorboard

The script expects:
  - autodrive_bridge running in Docker (sensor data on :5555, control on :5556)
  - Simulator open with vehicle set to Autonomous mode
  - Pre-saved boundary files at rl/boundaries/outer_line.npy + inner_line.npy
    (run a mapping lap first with enable_slam: true, then set use_saved_map: true)
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

# Add pilot_core root so shared_buffer, workers etc. are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from track_env import TrackEnv


# ── Feature extractor ─────────────────────────────────────────────────────────

class TrackCNNExtractor(BaseFeaturesExtractor):
    """1D CNN over the ordered boundary-point sequence in the observation.

    Obs layout (from boundary_obs + speed):
        obs[0  : 2*(n_ahead+1)]  — outer boundary, interleaved (x,y) per point
        obs[2*(n_ahead+1) : 4*(n_ahead+1)]  — inner boundary, interleaved (x,y)
        obs[-1]                  — speed

    The 21 points per wall form a spatial sequence (closest → furthest ahead).
    A 1D CNN over this sequence learns curvature patterns with weight sharing,
    unlike MLP which treats all features as unordered.
    """

    def __init__(self, observation_space, n_ahead: int = 20, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        self._seq_len = n_ahead + 1  # e.g. 21

        # 4 channels: outer_x, outer_y, inner_x, inner_y  ×  seq_len points
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        cnn_out_dim = 32 * self._seq_len  # preserves sequence length (same padding)

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, features_dim),  # +1 for speed
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        seq_len = self._seq_len

        # Split boundary points from speed
        boundary = obs[:, :-1]          # (batch, 4*seq_len)
        speed    = obs[:, -1:]          # (batch, 1)

        # Reshape interleaved (x,y) pairs → (batch, seq_len, 2) per wall
        outer = boundary[:, :2*seq_len].reshape(-1, seq_len, 2)
        inner = boundary[:, 2*seq_len:].reshape(-1, seq_len, 2)

        # Stack to (batch, 4_channels, seq_len) for Conv1d
        x = torch.cat([outer, inner], dim=2).permute(0, 2, 1)

        x = self.cnn(x)                             # (batch, 32, seq_len)
        x = x.flatten(1)                            # (batch, 32*seq_len)
        x = self.fc(torch.cat([x, speed], dim=1))   # (batch, features_dim)
        return x


# ── Config ────────────────────────────────────────────────────────────────────

BOUNDARY_PATH   = "rl/boundaries_easy"
LOG_DIR         = "rl/logs"
CHECKPOINT_DIR  = "rl/checkpoints"
TOTAL_TIMESTEPS = 500_000

ENV_KWARGS = dict(
    zmq_host               = "localhost",
    zmq_sensor_port        = 5555,
    zmq_control_port       = 5556,
    boundary_spacing       = 0.5,   # 20pts × 0.5m = 10m lookahead
    boundary_smooth_window = 30,
    n_ahead                = 20,
    max_steps              = 200,
    policy_hz              = 20.0,
)

POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],        # policy network — keep small for fast inference
        vf=[256, 256],        # value network — larger to handle return variance
    ),
    log_std_init=-0.5
)

PPO_KWARGS = dict(
    policy           = "MlpPolicy",
    policy_kwargs    = POLICY_KWARGS,
    device           = "cpu",   # uses CUDA if available, falls back to CPU
    learning_rate    = 3e-4,
    n_steps          = 1024,  # ~34 episodes per rollout vs 17 — stabilizes value function
    batch_size       = 256,
    n_epochs         = 5,
    gamma            = 0.99,
    gae_lambda       = 0.9,
    clip_range       = 0.2,
    clip_range_vf    = 0.2,
    target_kl        = 0.02,
    ent_coef         = 0.0001,
    vf_coef          = 0.5,
    max_grad_norm    = 0.5,
    verbose          = 1,
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self._ep_steps = 0
        self._ep_count = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        done = self.locals["dones"][0]
        self._ep_steps += 1
        if done:
            self._ep_count += 1
            # Use Monitor's raw episode return — rewards[0] is VecNormalize-scaled
            raw_reward = info.get("episode", {}).get("r", float("nan"))
            print(
                f"[ep {self._ep_count:4d}] "
                f"reward={raw_reward:8.1f}  "
                f"steps={self._ep_steps:4d}  "
                f"checkpoints={info.get('checkpoints', 0):2d}  "
                f"collided={info.get('collided', False)}  "
                f"stuck={info.get('stuck', False)}  "
                f"lap={info.get('lap_completed', False)}",
                flush=True,
            )
            self._ep_steps = 0
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize stats alongside each model checkpoint."""

    def __init__(self, save_path: str, save_freq: int):
        super().__init__()
        self._save_path = save_path
        self._save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            path = os.path.join(self._save_path, f"vec_normalize_{self.num_timesteps}.pkl")
            self.training_env.save(path)
        return True


# ── Run management ────────────────────────────────────────────────────────────

def _next_run_number() -> int:
    """Scan CHECKPOINT_DIR for PPO_N dirs and return the next N."""
    if not os.path.exists(CHECKPOINT_DIR):
        return 1
    pattern = re.compile(r"^PPO_(\d+)$")
    nums = [
        int(m.group(1))
        for d in os.listdir(CHECKPOINT_DIR)
        if (m := pattern.match(d)) and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
    ]
    return max(nums) + 1 if nums else 1


def _next_resume_index(run_log_dir: str) -> int:
    """Count existing run_N subdirs to determine the next resume index."""
    if not os.path.exists(run_log_dir):
        return 1
    pattern = re.compile(r"^run_(\d+)$")
    nums = [
        int(m.group(1))
        for d in os.listdir(run_log_dir)
        if (m := pattern.match(d)) and os.path.isdir(os.path.join(run_log_dir, d))
    ]
    return max(nums) + 1 if nums else 1


def _save_config(run_dir: str) -> None:
    """Persist ENV_KWARGS and PPO_KWARGS so resumes can verify compatibility."""
    # Separate out POLICY_KWARGS (nested) for clean serialization
    ppo_serializable = {k: v for k, v in PPO_KWARGS.items() if k != "policy_kwargs"}
    config = {
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
        "ENV_KWARGS": ENV_KWARGS,
        "PPO_KWARGS": ppo_serializable,
        "POLICY_KWARGS": POLICY_KWARGS,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _load_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def _step_from_path(model_path: str) -> int:
    """Extract step number from a checkpoint path, or 0 for 'final'."""
    m = re.search(r"ppo_track_(\d+)_steps\.zip", model_path)
    return int(m.group(1)) if m else 0


def _transplant_value_network(policy_model: PPO, value_model: PPO) -> None:
    """Copy value network weights from value_model into policy_model in-place.

    Transplants:
        mlp_extractor.value_net.*   — value MLP hidden layers
        value_net.*                 — final value output head

    The features_extractor and policy network weights are kept from policy_model.
    The optimizer is rebuilt from scratch after the transplant because the old
    momentum estimates no longer match the new value weights.
    """
    policy_sd = policy_model.policy.state_dict()
    value_sd  = value_model.policy.state_dict()

    value_keys = [
        k for k in value_sd
        if k.startswith("mlp_extractor.value_net") or k.startswith("value_net.")
    ]

    missing = [k for k in value_keys if k not in policy_sd]
    if missing:
        raise ValueError(
            f"Architecture mismatch — these keys exist in the value checkpoint "
            f"but not in the policy model: {missing}"
        )

    for k in value_keys:
        policy_sd[k] = value_sd[k]
    policy_model.policy.load_state_dict(policy_sd)

    # Rebuild optimizer — old momentum is stale after weight transplant
    opt_cls      = type(policy_model.policy.optimizer)
    opt_defaults = dict(policy_model.policy.optimizer.defaults)
    policy_model.policy.optimizer = opt_cls(
        policy_model.policy.parameters(), **opt_defaults
    )
    print(f"Transplanted {len(value_keys)} value-network tensors, optimizer reset.")


def _find_checkpoint(run_dir: str, step: int | None) -> tuple[str, str]:
    """Return (model_zip_path, vec_normalize_pkl_path) for the requested step.

    If step is None, uses the highest-numbered checkpoint available.
    """
    if step is not None:
        model_path = os.path.join(run_dir, f"ppo_track_{step}_steps.zip")
        vec_path   = os.path.join(run_dir, f"vec_normalize_{step}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint at step {step}: {model_path}")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"No VecNormalize stats at step {step}: {vec_path}")
        return model_path, vec_path

    # Find latest by step number
    zips = glob.glob(os.path.join(run_dir, "ppo_track_*_steps.zip"))
    if not zips:
        final_model = os.path.join(run_dir, "ppo_track_final.zip")
        final_vec   = os.path.join(run_dir, "vec_normalize.pkl")
        if os.path.exists(final_model) and os.path.exists(final_vec):
            return final_model, final_vec
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    def _step_num(p: str) -> int:
        m = re.search(r"ppo_track_(\d+)_steps\.zip", p)
        return int(m.group(1)) if m else -1

    latest     = max(zips, key=_step_num)
    latest_step = _step_num(latest)
    vec_path   = os.path.join(run_dir, f"vec_normalize_{latest_step}.pkl")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Model checkpoint found ({latest}) but no matching VecNormalize stats: {vec_path}"
        )
    return latest, vec_path


# ── Environment factory ───────────────────────────────────────────────────────

def make_env(env_kwargs: dict[str, Any] | None = None) -> TrackEnv:
    kwargs = env_kwargs or ENV_KWARGS
    env = TrackEnv(**kwargs)
    env.start()
    env.load_boundaries_from_path(BOUNDARY_PATH)
    return env


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume a PPO agent on the AutoDRIVE track.")
    parser.add_argument("--resume", metavar="PPO_N", help="Resume an existing run by name, e.g. PPO_1")
    parser.add_argument("--step",   type=int, default=None,
                        help="Checkpoint step to resume from (default: latest in the run)")
    # Merge mode: take policy weights from one checkpoint, value weights from another
    parser.add_argument("--policy-from", metavar="PPO_N",  dest="policy_from",
                        help="Run to take policy weights from, e.g. PPO_1")
    parser.add_argument("--policy-step", metavar="N", type=int, dest="policy_step", default=None,
                        help="Step to use from --policy-from (default: latest)")
    parser.add_argument("--value-from",  metavar="PPO_N",  dest="value_from",
                        help="Run to take value network weights from, e.g. PPO_2")
    parser.add_argument("--value-step",  metavar="N", type=int, dest="value_step",  default=None,
                        help="Step to use from --value-from (default: latest)")
    args = parser.parse_args()

    if args.policy_from and args.value_from:
        # ── Merge mode: policy weights from one checkpoint, value from another ─
        pol_dir  = os.path.join(CHECKPOINT_DIR, args.policy_from)
        val_dir  = os.path.join(CHECKPOINT_DIR, args.value_from)
        for d in [pol_dir, val_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Run directory not found: {d}")


        pol_model_path, pol_vec_path = _find_checkpoint(pol_dir, args.policy_step)
        val_model_path, _            = _find_checkpoint(val_dir, args.value_step)

        pol_config = _load_config(pol_dir)
        val_config = _load_config(val_dir)

        # Obs shape must match the policy checkpoint (it defines the network input)
        env_kwargs_to_use = pol_config["ENV_KWARGS"]

        # New run to save results into
        run_num   = _next_run_number()
        run_name  = f"PPO_{run_num}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        log_path  = os.path.join(LOG_DIR, run_name, "run_1")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(log_path,  exist_ok=True)
        _save_config(ckpt_path)

        print(f"Merge → {run_name}")
        print(f"  policy weights : {args.policy_from} / {os.path.basename(pol_model_path)}")
        print(f"  value  weights : {args.value_from}  / {os.path.basename(val_model_path)}")

        print("Starting environment...")
        raw_env = Monitor(make_env(env_kwargs_to_use), filename=os.path.join(log_path, "monitor"))
        env = DummyVecEnv([lambda: raw_env])
        # Use policy checkpoint's VecNormalize — obs stats should match the policy
        env = VecNormalize.load(pol_vec_path, env)
        env.training    = True
        env.norm_reward = True

        # Load both models (env needed only for policy model)
        pol_model = PPO.load(pol_model_path, env=env,
                             tensorboard_log=os.path.join(LOG_DIR, run_name))
        val_model = PPO.load(val_model_path)  # no env needed — weights only

        # Transplant value network from val_model into pol_model
        _transplant_value_network(pol_model, val_model)
        del val_model

        # Apply merged hyperparams:
        #   policy-side: clip_range, learning_rate from policy config
        #   value-side:  clip_range_vf, vf_coef from value config
        pol_ppo = pol_config.get("PPO_KWARGS", {})
        val_ppo = val_config.get("PPO_KWARGS", {})
        if "learning_rate" in pol_ppo:
            pol_model.learning_rate = float(pol_ppo["learning_rate"])
        if "clip_range" in pol_ppo:
            pol_model.clip_range = float(pol_ppo["clip_range"])
        if "clip_range_vf" in val_ppo:
            pol_model.clip_range_vf = float(val_ppo["clip_range_vf"])
        if "vf_coef" in val_ppo:
            pol_model.vf_coef = float(val_ppo["vf_coef"])

        # Timestep counter starts from the value checkpoint's step
        val_step = _step_from_path(val_model_path)
        pol_model.num_timesteps = val_step
        print(f"  starting timestep: {val_step}")

        model = pol_model

    elif args.resume:
        # ── Resume existing run ──────────────────────────────────────────────
        run_name  = args.resume
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Run directory not found: {ckpt_path}")

        saved_config = _load_config(ckpt_path)
        saved_env_kwargs = saved_config["ENV_KWARGS"]

        # Obs shape is determined by n_ahead — must match the saved model
        if saved_env_kwargs.get("n_ahead") != ENV_KWARGS.get("n_ahead"):
            print(
                f"WARNING: saved n_ahead={saved_env_kwargs['n_ahead']} differs from "
                f"current n_ahead={ENV_KWARGS['n_ahead']}. Using saved config to match model."
            )
        # Always use saved ENV_KWARGS on resume (obs shape must match checkpoint)
        env_kwargs_to_use = saved_env_kwargs

        model_path, vec_path = _find_checkpoint(ckpt_path, args.step)
        step_label = f"step {args.step}" if args.step else "latest checkpoint"
        print(f"Resuming {run_name} from {step_label}: {os.path.basename(model_path)}")

        resume_idx = _next_resume_index(os.path.join(LOG_DIR, run_name))
        log_path   = os.path.join(LOG_DIR, run_name, f"run_{resume_idx}")
        os.makedirs(log_path, exist_ok=True)

        print("Starting environment...")
        raw_env = Monitor(make_env(env_kwargs_to_use), filename=os.path.join(log_path, "monitor"))
        env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize.load(vec_path, env)
        env.training   = True
        env.norm_reward = True

        print(f"Loading model from {model_path} ...")
        # custom_objects overrides saved hyperparams with current PPO_KWARGS values
        overrides = {k: v for k, v in PPO_KWARGS.items()
                     if k not in ("policy", "policy_kwargs")}
        model = PPO.load(model_path, env=env,
                         tensorboard_log=os.path.join(LOG_DIR, run_name),
                         custom_objects=overrides)
        # PPO.load() doesn't reliably restore num_timesteps — set it from the filename
        resume_step = _step_from_path(model_path)
        if resume_step > 0:
            model.num_timesteps = resume_step
            print(f"Restored num_timesteps = {resume_step}")

    else:
        # ── New run ──────────────────────────────────────────────────────────
        run_num  = _next_run_number()
        run_name = f"PPO_{run_num}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        log_path  = os.path.join(LOG_DIR, run_name, "run_1")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(log_path,  exist_ok=True)

        _save_config(ckpt_path)
        print(f"Starting new run: {run_name}")

        print("Starting environment...")
        raw_env = Monitor(make_env(), filename=os.path.join(log_path, "monitor"))
        env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=0.99,
        )

        print("Building PPO model...")
        model = PPO(env=env, tensorboard_log=os.path.join(LOG_DIR, run_name), **PPO_KWARGS)

        env_kwargs_to_use = ENV_KWARGS  # used only for the new-run path

    print(f"Run: {run_name}  |  checkpoints → {ckpt_path}")
    print(f"TensorBoard:  tensorboard --logdir {os.path.join(LOG_DIR, run_name)}")

    callbacks = [
        EpisodeLogger(),
        CheckpointCallback(
            save_freq   = 10_000,
            save_path   = ckpt_path,
            name_prefix = "ppo_track",
        ),
        SaveVecNormalizeCallback(
            save_path = ckpt_path,
            save_freq = 10_000,
        ),
    ]

    model.learn(
        total_timesteps      = TOTAL_TIMESTEPS,
        callback             = callbacks,
        progress_bar         = True,
        reset_num_timesteps  = (args.resume is None),
    )

    model.save(os.path.join(ckpt_path, "ppo_track_final"))
    env.save(os.path.join(ckpt_path, "vec_normalize.pkl"))
    print(f"Saved final model → {os.path.join(ckpt_path, 'ppo_track_final.zip')}")

    env.close()


if __name__ == "__main__":
    main()
