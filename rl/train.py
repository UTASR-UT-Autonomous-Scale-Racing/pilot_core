"""
PPO training on the AutoDRIVE track env.

    python rl/train.py                            # new run, auto-numbered PPO_N
    python rl/train.py --resume PPO_1             # continue from latest checkpoint
    python rl/train.py --resume PPO_1 --step 50000
    tensorboard --logdir rl/logs/PPO_1            # shows every resume of the run

Merge mode starts a new run using the policy net (plus its VecNormalize stats)
from one run and the value net from another. --policy-from and --value-from
have to be given together; the -step flags are optional:

    python rl/train.py --policy-from PPO_5 --value-from PPO_3
    python rl/train.py --policy-from PPO_5 --policy-step 75000 \
                       --value-from PPO_3 --value-step 50000

Checkpoints (zips, VecNormalize pkls, config.json) land in
rl/checkpoints/PPO_N/, logs in rl/logs/PPO_N/run_K/. Use rl/export_onnx.py to
get something the car can run.

Needs `pip install stable-baselines3 tensorboard`, plus: autodrive_bridge
up in Docker (sensors :5555, control :5556), the sim open with the vehicle in
Autonomous mode, and BOUNDARY_PATH/{outer,inner}_line.npy. Those come from a
mapping run with config_zmq.yaml (see mapping.boundary_save_path);
default path is rl/boundaries_easy.
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Any

# repo root, for shared_buffer / workers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from track_env import TrackEnv


class TrackCNNExtractor(BaseFeaturesExtractor):
    """1D CNN over the boundary points in the obs.

    Not used by default, PPO_KWARGS is a plain MlpPolicy. To try it, put
    features_extractor_class=TrackCNNExtractor and
    features_extractor_kwargs=dict(n_ahead=...) into POLICY_KWARGS.

    Obs is outer (x, y) pairs, then inner (x, y) pairs, then speed last. Each
    wall's n_ahead+1 points are ordered closest -> furthest, so a conv with
    shared weights can pick up curvature, where an MLP just sees a flat
    unordered vector.
    """

    def __init__(self, observation_space, n_ahead: int = 20, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        self._seq_len = n_ahead + 1

        # channels: outer_x, outer_y, inner_x, inner_y
        self.cnn = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        cnn_out_dim = 32 * self._seq_len  # same padding, length unchanged

        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 1, features_dim),  # +1 for speed
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        seq_len = self._seq_len

        boundary = obs[:, :-1]          # (batch, 4*seq_len)
        speed    = obs[:, -1:]

        # interleaved xy -> (batch, seq_len, 2) per wall
        outer = boundary[:, :2*seq_len].reshape(-1, seq_len, 2)
        inner = boundary[:, 2*seq_len:].reshape(-1, seq_len, 2)

        # Conv1d wants (batch, channels, seq_len)
        x = torch.cat([outer, inner], dim=2).permute(0, 2, 1)

        x = self.cnn(x)  # (batch, 32, seq_len)
        x = x.flatten(1)
        x = self.fc(torch.cat([x, speed], dim=1))
        return x


# config

BOUNDARY_PATH   = "rl/boundaries_easy"
LOG_DIR         = "rl/logs"
CHECKPOINT_DIR  = "rl/checkpoints"
TOTAL_TIMESTEPS = 500_000

ENV_KWARGS = dict(
    zmq_host               = "localhost",
    zmq_sensor_port        = 5555,
    zmq_control_port       = 5556,
    boundary_spacing       = 0.5,   # 20 pts x 0.5 m = 10 m lookahead
    boundary_smooth_window = 30,
    n_ahead                = 20,
    max_steps              = 200,
    policy_hz              = 20.0,
)

POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],        # this is what gets exported to ONNX
        vf=[256, 256],        # training only
    ),
    log_std_init=-0.5,
)

PPO_KWARGS = dict(
    policy           = "MlpPolicy",
    policy_kwargs    = POLICY_KWARGS,
    device           = "cpu",   # MLP this small isn't any faster on GPU
    learning_rate    = 3e-4,
    n_steps          = 1024,  # per update
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
            # rewards[0] is VecNormalize-scaled, so use Monitor's raw return instead
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
    """VecNormalize stats need saving next to every model checkpoint, or
    resumes won't match."""

    def __init__(self, save_path: str, save_freq: int):
        super().__init__()
        self._save_path = save_path
        self._save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self._save_freq == 0:
            path = os.path.join(self._save_path, f"vec_normalize_{self.num_timesteps}.pkl")
            self.training_env.save(path)
        return True


# run bookkeeping

def _next_run_number() -> int:
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
    """Written so a later resume can check it's compatible."""
    # policy_kwargs is nested; it's saved separately as POLICY_KWARGS
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
    """Step number from a checkpoint filename (0 for the final one)."""
    m = re.search(r"ppo_track_(\d+)_steps\.zip", model_path)
    return int(m.group(1)) if m else 0


def _transplant_value_network(policy_model: PPO, value_model: PPO) -> None:
    """Copy the value net (mlp_extractor.value_net.* and the value_net.*
    head) from value_model into policy_model, in place.

    Feature extractor and policy weights stay as they were. The optimizer
    gets rebuilt, since its momentum was accumulated for the old value weights.
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

    opt_cls      = type(policy_model.policy.optimizer)
    opt_defaults = dict(policy_model.policy.optimizer.defaults)
    policy_model.policy.optimizer = opt_cls(
        policy_model.policy.parameters(), **opt_defaults
    )
    print(f"Transplanted {len(value_keys)} value-network tensors, optimizer reset.")


def _find_checkpoint(run_dir: str, step: int | None) -> tuple[str, str]:
    """(model zip, vecnormalize pkl) for a step; step=None means latest."""
    if step is not None:
        model_path = os.path.join(run_dir, f"ppo_track_{step}_steps.zip")
        vec_path   = os.path.join(run_dir, f"vec_normalize_{step}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint at step {step}: {model_path}")
        if not os.path.exists(vec_path):
            raise FileNotFoundError(f"No VecNormalize stats at step {step}: {vec_path}")
        return model_path, vec_path

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


def make_env(env_kwargs: dict[str, Any] | None = None) -> TrackEnv:
    kwargs = env_kwargs or ENV_KWARGS
    env = TrackEnv(**kwargs)
    env.start()
    env.load_boundaries_from_path(BOUNDARY_PATH)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or resume a PPO agent on the AutoDRIVE track.")
    parser.add_argument("--resume", metavar="PPO_N", help="Resume an existing run by name, e.g. PPO_1")
    parser.add_argument("--step",   type=int, default=None,
                        help="Checkpoint step to resume from (default: latest in the run)")
    # merge mode flags
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
        # merge
        pol_dir  = os.path.join(CHECKPOINT_DIR, args.policy_from)
        val_dir  = os.path.join(CHECKPOINT_DIR, args.value_from)
        for d in [pol_dir, val_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Run directory not found: {d}")

        pol_model_path, pol_vec_path = _find_checkpoint(pol_dir, args.policy_step)
        val_model_path, _            = _find_checkpoint(val_dir, args.value_step)

        pol_config = _load_config(pol_dir)
        val_config = _load_config(val_dir)

        # obs shape has to match the policy checkpoint since that's the net input
        env_kwargs_to_use = pol_config["ENV_KWARGS"]

        # results go into a fresh run dir
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
        # VecNormalize comes from the policy side, obs stats belong with the policy
        env = VecNormalize.load(pol_vec_path, env)
        env.training    = True
        env.norm_reward = True

        pol_model = PPO.load(pol_model_path, env=env,
                             tensorboard_log=os.path.join(LOG_DIR, run_name))
        val_model = PPO.load(val_model_path)  # weights only, no env

        _transplant_value_network(pol_model, val_model)
        del val_model

        # hyperparams: lr + clip_range from the policy run,
        # clip_range_vf + vf_coef from the value run
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

        # timestep counter continues from the value checkpoint
        val_step = _step_from_path(val_model_path)
        pol_model.num_timesteps = val_step
        print(f"  starting timestep: {val_step}")

        model = pol_model

    elif args.resume:
        # resume
        run_name  = args.resume
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Run directory not found: {ckpt_path}")

        saved_config = _load_config(ckpt_path)
        saved_env_kwargs = saved_config["ENV_KWARGS"]

        # n_ahead sets the obs shape, so it has to match the saved model
        if saved_env_kwargs.get("n_ahead") != ENV_KWARGS.get("n_ahead"):
            print(
                f"WARNING: saved n_ahead={saved_env_kwargs['n_ahead']} differs from "
                f"current n_ahead={ENV_KWARGS['n_ahead']}. Using saved config to match model."
            )
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
        # custom_objects lets the current PPO_KWARGS override what was saved
        overrides = {k: v for k, v in PPO_KWARGS.items()
                     if k not in ("policy", "policy_kwargs")}
        model = PPO.load(model_path, env=env,
                         tensorboard_log=os.path.join(LOG_DIR, run_name),
                         custom_objects=overrides)
        # hack: PPO.load() doesn't reliably restore num_timesteps, take it from the filename
        resume_step = _step_from_path(model_path)
        if resume_step > 0:
            model.num_timesteps = resume_step
            print(f"Restored num_timesteps = {resume_step}")

    else:
        # new run
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

        env_kwargs_to_use = ENV_KWARGS

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
