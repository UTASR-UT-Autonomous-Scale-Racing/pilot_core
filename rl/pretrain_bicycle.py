#!/usr/bin/env python3
"""PPO pretraining in rl/bicycle_env.BicycleTrackEnv (no simulator), written as an
rl/train.py run so it can be fine-tuned on the live sim with ``rl/train.py --resume``.

    python rl/pretrain_bicycle.py --n-envs 16 --timesteps 5000000
    python rl/pretrain_bicycle.py --pilot-config config_autodrive.yaml --speed-cap 4.5

Run layout (rl/checkpoints/PPO_N) is train.py's: ppo_track_<step>_steps.zip +
vec_normalize_<step>.pkl every --ckpt-every steps, ppo_track_final.zip + vec_normalize.pkl,
config.json with ENV_KWARGS = train.ENV_KWARGS (TrackEnv kwargs, safety filled in) and
train.py's POLICY_KWARGS. Bicycle-only settings and the eval history go to
bicycle_pretrain.json / eval_log.jsonl next to them. Each checkpoint is evaluated
deterministically on the saved AutoDRIVE maps from the sim's spawn pose; --patience stops
the run once that stops improving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize  # noqa: E402

from rl import train  # noqa: E402
from rl.bicycle_env import DEFAULT_MAPS, BicycleTrackEnv  # noqa: E402
from rl.eval_bicycle import make_eval_env, run_episodes, sb3_actor  # noqa: E402

# bicycle-only env settings (BicycleTrackEnv defaults unless overridden here)
BICYCLE_KWARGS: dict = dict(
    map_dirs=[os.path.relpath(m, REPO_ROOT) for m in DEFAULT_MAPS],
)

# rollout size suits 16 parallel envs; train.PPO_KWARGS override these on a live resume
PRETRAIN_PPO_OVERRIDES = dict(n_steps=512, batch_size=1024, n_epochs=5)


def env_factory(env_kwargs: dict, bicycle_kwargs: dict, seed: int, monitor_path: str | None):
    def make():
        env = BicycleTrackEnv(**env_kwargs, **bicycle_kwargs)
        env.reset(seed=seed)
        return Monitor(env, filename=monitor_path,
                       info_keywords=("lap_completed", "collided", "stuck", "progress"))
    return make


class StatsLogger(BaseCallback):
    """Episode stats over all envs, printed once per rollout."""

    def __init__(self, window: int = 200):
        super().__init__()
        self._eps: deque = deque(maxlen=window)
        self._t0 = time.time()

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and "episode" in info:
                self._eps.append((info["episode"]["r"], info["episode"]["l"], info.get("lap_completed", False),
                                  info.get("collided", False), info.get("stuck", False), info.get("progress", 0.0)))
        return True

    def _on_rollout_end(self) -> None:
        if not self._eps:
            return
        r, ln, lap, col, stk, prog = (np.array(x, dtype=float) for x in zip(*self._eps))
        stats = {"ep_return": r.mean(), "ep_len": ln.mean(), "lap_rate": lap.mean(),
                 "collision_rate": col.mean(), "stuck_rate": stk.mean(), "progress_m": prog.mean()}
        for k, v in stats.items():
            self.logger.record(f"bicycle/{k}", float(v))
        el = time.time() - self._t0
        print(f"[{self.num_timesteps:>9d} steps {el / 60:6.1f} min {self.num_timesteps / max(el, 1e-6):6.0f} sps] "
              + "  ".join(f"{k}={v:.3f}" for k, v in stats.items()), flush=True)


class MapEval(BaseCallback):
    """Deterministic laps on the saved maps every eval_freq steps; early stop on a plateau."""

    def __init__(self, run_dir: str, env_kwargs: dict, bicycle_kwargs: dict, maps, eval_freq: int,
                 episodes: int, patience: int):
        super().__init__()
        self._run_dir = run_dir
        self._env = make_eval_env(env_kwargs, bicycle_kwargs)
        self._maps = list(maps)
        self._freq = eval_freq
        self._episodes = episodes
        self._patience = patience
        self._next = eval_freq
        self._best = -np.inf
        self._since_best = 0
        self._log = os.path.join(run_dir, "eval_log.jsonl")

    def _score(self, res: dict) -> float:
        # completion first, then lap time (s) as a tie-breaker
        s = 0.0
        for r in res.values():
            s += 100.0 * r["lap_rate"] - (r["lap_time_mean"] if r["lap_time_mean"] else 100.0) * r["lap_rate"]
        return s / len(res)

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next:
            return True
        self._next += self._freq
        act = sb3_actor(self.model, self.training_env)   # VecNormalize.normalize_obs, current stats
        res = {}
        for m in self._maps:
            res[os.path.basename(os.path.normpath(m))] = run_episodes(self._env, act, m, self._episodes, seed=1000)
        score = self._score(res)
        for name, r in res.items():
            self.logger.record(f"eval/{name}_lap_rate", r["lap_rate"])
            if r["lap_time_mean"]:
                self.logger.record(f"eval/{name}_lap_time", r["lap_time_mean"])
            self.logger.record(f"eval/{name}_xtrack_rms", r["xtrack_rms"])
        with open(self._log, "a") as f:
            f.write(json.dumps({"step": int(self.num_timesteps), "score": score, **res}) + "\n")
        print(f"[eval {self.num_timesteps}] score={score:.2f} " + "  ".join(
            f"{n}: laps={100 * r['lap_rate']:.0f}% t={r['lap_time_mean'] or float('nan'):.2f}s "
            f"coll={r['collisions']} xtrack={r['xtrack_rms']:.3f}" for n, r in res.items()), flush=True)
        if score > self._best + 0.1:
            self._best, self._since_best = score, 0
            self.model.save(os.path.join(self._run_dir, "best_model"))
            self.training_env.save(os.path.join(self._run_dir, "best_vec_normalize.pkl"))
            with open(os.path.join(self._run_dir, "best_step.txt"), "w") as f:
                f.write(f"{self.num_timesteps}\n")
        else:
            self._since_best += 1
        if self._patience and self._since_best >= self._patience:
            print(f"Eval score has not improved for {self._patience} evals, stopping.", flush=True)
            return False
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--ckpt-every", type=int, default=250_000, help="checkpoint + eval period (steps)")
    ap.add_argument("--eval-episodes", type=int, default=5, help="per map, per eval")
    ap.add_argument("--patience", type=int, default=6, help="evals without improvement before stopping (0 = never)")
    ap.add_argument("--pilot-config", default=os.path.join(REPO_ROOT, "config_autodrive.yaml"),
                    help="pilot config whose `safety` section the env's supervisor uses")
    ap.add_argument("--speed-cap", type=float, default=None, help="override safety.speed_cap (m/s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dummy", action="store_true", help="DummyVecEnv (debugging)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))

    safety = train.safety_from_pilot_config(args.pilot_config)
    if args.speed_cap is not None:
        safety["speed_cap"] = float(args.speed_cap)
    env_kwargs = train.resolve_env_kwargs(train.ENV_KWARGS, safety)
    bicycle_kwargs = {k: (list(v) if isinstance(v, tuple) else v) for k, v in BICYCLE_KWARGS.items()}
    sim_env_kwargs = {k: v for k, v in env_kwargs.items()}

    os.chdir(REPO_ROOT)   # train.py's CHECKPOINT_DIR / LOG_DIR are repo-relative
    run_name = f"PPO_{train._next_run_number()}"
    ckpt_path = os.path.join(train.CHECKPOINT_DIR, run_name)
    log_path = os.path.join(train.LOG_DIR, run_name, "run_1")
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    train._save_config(ckpt_path, env_kwargs)
    ppo_kwargs = dict(train.PPO_KWARGS, **PRETRAIN_PPO_OVERRIDES, seed=args.seed)
    with open(os.path.join(ckpt_path, "bicycle_pretrain.json"), "w") as f:
        json.dump({"BICYCLE_KWARGS": bicycle_kwargs, "N_ENVS": args.n_envs, "TIMESTEPS": args.timesteps,
                   "PPO_KWARGS": {k: v for k, v in ppo_kwargs.items() if k != "policy_kwargs"},
                   "argv": sys.argv}, f, indent=2)
    print(f"Bicycle pretraining run {run_name} → {ckpt_path}  (safety speed_cap={safety['speed_cap']})", flush=True)

    fns = [env_factory(sim_env_kwargs, bicycle_kwargs, args.seed * 1000 + i,
                       os.path.join(log_path, f"monitor_{i}")) for i in range(args.n_envs)]
    venv = DummyVecEnv(fns) if args.dummy else SubprocVecEnv(fns, start_method="fork")
    env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    model = PPO(env=env, tensorboard_log=os.path.join(train.LOG_DIR, run_name), **ppo_kwargs)

    save_freq = max(args.ckpt_every // args.n_envs, 1)   # callbacks count vec steps
    callbacks = [
        StatsLogger(),
        CheckpointCallback(save_freq=save_freq, save_path=ckpt_path, name_prefix="ppo_track"),
        train.SaveVecNormalizeCallback(save_path=ckpt_path, save_freq=save_freq),
        MapEval(ckpt_path, sim_env_kwargs, bicycle_kwargs, BICYCLE_KWARGS["map_dirs"], args.ckpt_every,
                args.eval_episodes, args.patience),
    ]
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
        model.save(os.path.join(ckpt_path, "ppo_track_final"))
        env.save(os.path.join(ckpt_path, "vec_normalize.pkl"))
        print(f"Saved final model → {os.path.join(ckpt_path, 'ppo_track_final.zip')} "
              f"({model.num_timesteps} steps, {(time.time() - t0) / 60:.1f} min)", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
