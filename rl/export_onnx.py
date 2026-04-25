#!/usr/bin/env python3
"""
Export a PPO policy from rl/train.py to ONNX for running on the car
(control_mode: policy, see control/policy.py).

The graph bakes in the VecNormalize step, so it's the full deterministic path:
raw obs -> (obs - mean) / sqrt(var + eps), clipped -> mean action -> clip to
the action space. A <out>.json sidecar next to it records the obs settings
(n_ahead, boundary spacing/smoothing) the policy was trained with.

Usage:
    python rl/export_onnx.py --run PPO_3   # latest checkpoint of that run
    python rl/export_onnx.py --model path/model.zip --vecnormalize path/vec.pkl \\
        --config path/config.json --out policy.onnx
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class DeterministicPolicy(torch.nn.Module):
    """VecNormalize + PPO mean action + clipping, as a single module."""

    def __init__(self, policy, obs_mean, obs_var, clip_obs, eps, act_low, act_high) -> None:
        super().__init__()
        self.policy = policy
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("std", torch.sqrt(torch.as_tensor(obs_var, dtype=torch.float32) + eps))
        self.register_buffer("low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(act_high, dtype=torch.float32))
        self.clip_obs = float(clip_obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.mean) / self.std, -self.clip_obs, self.clip_obs)
        features = self.policy.extract_features(x, self.policy.pi_features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        action = self.policy.action_net(latent_pi)
        return torch.max(torch.min(action, self.high), self.low)


def export(model, vec_normalize, env_kwargs: dict, out_path: str) -> str:
    """SB3 PPO model + its VecNormalize stats -> ONNX at out_path."""
    obs_dim = int(model.observation_space.shape[0])
    rms = vec_normalize.obs_rms if vec_normalize is not None and vec_normalize.norm_obs else None
    mean = rms.mean if rms is not None else np.zeros(obs_dim)
    var = rms.var if rms is not None else np.ones(obs_dim) - 1e-8
    clip_obs = vec_normalize.clip_obs if rms is not None else 1e9
    eps = vec_normalize.epsilon if rms is not None else 1e-8

    module = DeterministicPolicy(model.policy, mean, var, clip_obs, eps,
                                 model.action_space.low, model.action_space.high).eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    torch.onnx.export(module, dummy, out_path, input_names=["obs"], output_names=["action"],
                      dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
                      opset_version=17, dynamo=False)

    meta = {
        "observation": {
            "n_ahead": int(env_kwargs["n_ahead"]),
            "boundary_spacing": float(env_kwargs["boundary_spacing"]),
            "boundary_smooth_window": int(env_kwargs["boundary_smooth_window"]),
            "obs_dim": obs_dim,
        },
        "action": {"low": model.action_space.low.tolist(), "high": model.action_space.high.tolist(),
                   "semantics": "[steering (+1 = left), throttle (0..1, forward)]"},
    }
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", help="run name under rl/checkpoints/ (latest checkpoint unless --step)")
    ap.add_argument("--step", type=int, help="checkpoint step to export with --run")
    ap.add_argument("--model", help="PPO .zip")
    ap.add_argument("--vecnormalize", help="VecNormalize .pkl")
    ap.add_argument("--config", help="run config.json (ENV_KWARGS)")
    ap.add_argument("--out", default="policy.onnx", help="output ONNX path (default: policy.onnx)")
    args = ap.parse_args()

    # train.py expects to be run from rl/ (it does `from track_env import ...`).
    # We import it anyway because that registers TrackCNNExtractor, which the
    # unpickler needs
    sys.path.insert(0, os.path.join(REPO_ROOT, "rl"))
    import importlib
    train = importlib.import_module("train")

    if args.run:
        run_dir = os.path.join(REPO_ROOT, "rl", "checkpoints", args.run)
        # model + vecnormalize must come from the same step
        model_path, vec_path = train._find_checkpoint(run_dir, args.step)
        args.model = args.model or model_path
        args.vecnormalize = args.vecnormalize or vec_path
        args.config = args.config or os.path.join(run_dir, "config.json")
    if not (args.model and args.config):
        ap.error("--run or --model/--config required")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    with open(args.config) as f:
        env_kwargs = json.load(f)["ENV_KWARGS"]
    model = PPO.load(args.model, device="cpu")
    vec = None
    if args.vecnormalize:
        import pickle
        with open(args.vecnormalize, "rb") as f:
            vec = pickle.load(f)
        if not isinstance(vec, VecNormalize):
            raise TypeError(f"{args.vecnormalize} is not a VecNormalize pickle")
    out = export(model, vec, env_kwargs, args.out)
    print(f"Exported {args.model} → {out} (+ {os.path.splitext(out)[0]}.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
