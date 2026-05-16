"""Checkpoint lookup for rl/checkpoints/PPO_N run dirs (shared by train.py and export_onnx.py).
"""

from __future__ import annotations

import glob
import os
import re

_STEP_RE = re.compile(r"ppo_track_(\d+)_steps\.zip")


def step_from_path(model_path: str) -> int:
    """Step number from a checkpoint filename (0 for the final one)."""
    m = _STEP_RE.search(os.path.basename(model_path))
    return int(m.group(1)) if m else 0


def find_checkpoint(run_dir: str, step: int | None) -> tuple[str, str]:
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

    latest      = max(zips, key=step_from_path)
    latest_step = step_from_path(latest)
    vec_path   = os.path.join(run_dir, f"vec_normalize_{latest_step}.pkl")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            f"Model checkpoint found ({latest}) but no matching VecNormalize stats: {vec_path}"
        )
    return latest, vec_path
