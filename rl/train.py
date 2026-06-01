"""PPO training on the AutoDRIVE track env (needs stable-baselines3, autodrive_bridge, sim running).

    python rl/train.py                            # new run, auto-numbered PPO_N
    python rl/train.py --resume PPO_1 --step 50000
    python rl/train.py --policy-from PPO_5 --value-from PPO_3   # merge nets
"""

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
from dataclasses import asdict
from typing import Any

# repo root, for shared_buffer / workers / rl.*
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.checkpoints import find_checkpoint, step_from_path


# config

BOUNDARY_PATH   = "rl/boundaries_easy"
LOG_DIR         = "rl/logs"
CHECKPOINT_DIR  = "rl/checkpoints"
TOTAL_TIMESTEPS = 500_000

ENV_KWARGS = dict(
    sim_host               = "0.0.0.0",   # the sim connects to the env (AutoDRIVEWorker)
    sim_port               = 4567,
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


def _save_config(run_dir: str, env_kwargs: dict[str, Any]) -> None:
    """Written so a later resume/merge/export uses the same env settings.
    env_kwargs is what the env was actually built with."""
    # policy_kwargs is nested; it's saved separately as POLICY_KWARGS
    ppo_serializable = {k: v for k, v in PPO_KWARGS.items() if k != "policy_kwargs"}
    config = {
        "BOUNDARY_PATH": BOUNDARY_PATH,
        "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
        "ENV_KWARGS": env_kwargs,
        "PPO_KWARGS": ppo_serializable,
        "POLICY_KWARGS": POLICY_KWARGS,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _load_config(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def safety_from_pilot_config(path: str | None) -> dict:
    """SafetyConfig fields the car would run with for this pilot config.

    Mirrors start.py: the ``safety`` section, with speed_per_throttle
    defaulting to track_control.speed.longitudinal.speed_per_throttle (3.0).
    A missing file gives the SafetyConfig defaults.
    """
    from control.safety import SafetyConfig

    cfg: dict = {}
    if path and os.path.exists(path):
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    safety = dict(cfg.get("safety") or {})
    spt = (cfg.get("track_control") or {}).get("speed", {}).get("longitudinal", {}).get("speed_per_throttle", 3.0)
    safety.setdefault("speed_per_throttle", spt)
    return asdict(SafetyConfig(**safety))


def resolve_env_kwargs(base: dict[str, Any], safety: dict) -> dict[str, Any]:
    """Env kwargs to build (and save) the env with, safety filled in if the
    saved config predates it (older runs also saved vis_zmq_port and the ZMQ bridge's
    zmq_* ports, which the direct sim link replaced)."""
    legacy = ("vis_zmq_port", "zmq_host", "zmq_sensor_port", "zmq_control_port")
    kwargs = {k: v for k, v in base.items() if k not in legacy}
    if kwargs.get("safety") is None:
        kwargs["safety"] = dict(safety)
    elif kwargs["safety"] != safety:
        print(f"NOTE: using the run's saved safety settings {kwargs['safety']} "
              f"instead of the pilot config's {safety}")
    return kwargs


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


def merge_overrides(pol_config: dict, val_config: dict) -> dict[str, Any]:
    """Hyperparameters for a merged model: lr + clip_range from the policy
    run, clip_range_vf + vf_coef from the value run.

    They go into PPO.load(custom_objects=...) so _setup_model() turns them
    into proper schedules (setting e.g. model.clip_range = 0.2 after loading
    leaves a float where SB3 expects a callable)."""
    pol_ppo = pol_config.get("PPO_KWARGS", {})
    val_ppo = val_config.get("PPO_KWARGS", {})
    overrides: dict[str, Any] = {}
    for key, src in (("learning_rate", pol_ppo), ("clip_range", pol_ppo),
                     ("clip_range_vf", val_ppo), ("vf_coef", val_ppo)):
        if key in src:
            overrides[key] = src[key]
    return overrides


def load_merged_model(pol_model_path: str, val_model_path: str, pol_config: dict, val_config: dict,
                      env, tensorboard_log: str | None = None) -> PPO:
    """Policy net (+ features) from one checkpoint, value net from another."""
    pol_model = PPO.load(pol_model_path, env=env, device="cpu", tensorboard_log=tensorboard_log,
                         custom_objects=merge_overrides(pol_config, val_config))
    val_model = PPO.load(val_model_path, device="cpu")  # weights only, no env
    _transplant_value_network(pol_model, val_model)
    del val_model
    # timestep counter continues from the value checkpoint
    pol_model.num_timesteps = step_from_path(val_model_path)
    return pol_model


def load_resumed_model(model_path: str, env, tensorboard_log: str | None = None) -> PPO:
    """Reload a checkpoint; the current PPO_KWARGS override what was saved."""
    overrides = {k: v for k, v in PPO_KWARGS.items() if k not in ("policy", "policy_kwargs", "device")}
    model = PPO.load(model_path, env=env, device=PPO_KWARGS.get("device", "cpu"),
                     tensorboard_log=tensorboard_log, custom_objects=overrides)
    # PPO.load() doesn't reliably restore num_timesteps, take it from the filename
    resume_step = step_from_path(model_path)
    if resume_step > 0:
        model.num_timesteps = resume_step
    return model


def make_env(env_kwargs: dict[str, Any]):
    from rl.track_env import TrackEnv

    env = TrackEnv(**env_kwargs)
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
    parser.add_argument("--pilot-config", default=os.path.join(REPO_ROOT, "config_autodrive.yaml"),
                        dest="pilot_config",
                        help="pilot config whose `safety` section the env's actuator uses (default: config_autodrive.yaml)")
    args = parser.parse_args()
    if bool(args.policy_from) != bool(args.value_from):
        parser.error("--policy-from and --value-from have to be given together")

    # workers are spawned from the env; buffers are created in this process
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # already set

    safety = safety_from_pilot_config(args.pilot_config)
    tb_root = None

    if args.policy_from and args.value_from:
        mode = "merge"
        pol_dir  = os.path.join(CHECKPOINT_DIR, args.policy_from)
        val_dir  = os.path.join(CHECKPOINT_DIR, args.value_from)
        for d in [pol_dir, val_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"Run directory not found: {d}")

        pol_model_path, pol_vec_path = find_checkpoint(pol_dir, args.policy_step)
        val_model_path, _            = find_checkpoint(val_dir, args.value_step)

        pol_config = _load_config(pol_dir)
        val_config = _load_config(val_dir)

        # obs shape has to match the policy checkpoint since that's the net input
        env_kwargs_to_use = resolve_env_kwargs(pol_config["ENV_KWARGS"], safety)

        # results go into a fresh run dir
        run_num   = _next_run_number()
        run_name  = f"PPO_{run_num}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        log_path  = os.path.join(LOG_DIR, run_name, "run_1")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(log_path,  exist_ok=True)
        _save_config(ckpt_path, env_kwargs_to_use)
        tb_root = os.path.join(LOG_DIR, run_name)

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

        model = load_merged_model(pol_model_path, val_model_path, pol_config, val_config, env, tb_root)
        print(f"  starting timestep: {model.num_timesteps}")

    elif args.resume:
        mode = "resume"
        run_name  = args.resume
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"Run directory not found: {ckpt_path}")

        saved_env_kwargs = _load_config(ckpt_path)["ENV_KWARGS"]

        # n_ahead sets the obs shape, so it has to match the saved model
        if saved_env_kwargs.get("n_ahead") != ENV_KWARGS.get("n_ahead"):
            print(
                f"WARNING: saved n_ahead={saved_env_kwargs['n_ahead']} differs from "
                f"current n_ahead={ENV_KWARGS['n_ahead']}. Using saved config to match model."
            )
        env_kwargs_to_use = resolve_env_kwargs(saved_env_kwargs, safety)
        _save_config(ckpt_path, env_kwargs_to_use)
        tb_root = os.path.join(LOG_DIR, run_name)

        model_path, vec_path = find_checkpoint(ckpt_path, args.step)
        step_label = f"step {args.step}" if args.step else "latest checkpoint"
        print(f"Resuming {run_name} from {step_label}: {os.path.basename(model_path)}")

        resume_idx = _next_resume_index(tb_root)
        log_path   = os.path.join(tb_root, f"run_{resume_idx}")
        os.makedirs(log_path, exist_ok=True)

        print("Starting environment...")
        raw_env = Monitor(make_env(env_kwargs_to_use), filename=os.path.join(log_path, "monitor"))
        env = DummyVecEnv([lambda: raw_env])
        env = VecNormalize.load(vec_path, env)
        env.training   = True
        env.norm_reward = True

        print(f"Loading model from {model_path} ...")
        model = load_resumed_model(model_path, env, tb_root)
        print(f"num_timesteps = {model.num_timesteps}")

    else:
        mode = "new"
        run_num  = _next_run_number()
        run_name = f"PPO_{run_num}"
        ckpt_path = os.path.join(CHECKPOINT_DIR, run_name)
        log_path  = os.path.join(LOG_DIR, run_name, "run_1")
        os.makedirs(ckpt_path, exist_ok=True)
        os.makedirs(log_path,  exist_ok=True)

        env_kwargs_to_use = resolve_env_kwargs(ENV_KWARGS, safety)
        _save_config(ckpt_path, env_kwargs_to_use)
        tb_root = os.path.join(LOG_DIR, run_name)
        print(f"Starting new run: {run_name}")

        print("Starting environment...")
        raw_env = Monitor(make_env(env_kwargs_to_use), filename=os.path.join(log_path, "monitor"))
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
        model = PPO(env=env, tensorboard_log=tb_root, **PPO_KWARGS)

    print(f"Run: {run_name}  |  checkpoints → {ckpt_path}")
    print(f"TensorBoard:  tensorboard --logdir {tb_root}")

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

    try:
        model.learn(
            total_timesteps      = TOTAL_TIMESTEPS,
            callback             = callbacks,
            progress_bar         = True,
            # merge and resume both continue the step counter
            reset_num_timesteps  = (mode == "new"),
        )
        model.save(os.path.join(ckpt_path, "ppo_track_final"))
        env.save(os.path.join(ckpt_path, "vec_normalize.pkl"))
        print(f"Saved final model → {os.path.join(ckpt_path, 'ppo_track_final.zip')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
