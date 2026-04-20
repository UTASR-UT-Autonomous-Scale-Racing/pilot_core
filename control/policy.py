"""
Learned-policy controller: runs a PPO policy exported to ONNX
(``rl/export_onnx.py``) on the CPU with onnxruntime.

Reference: J. Schulman et al., "Proximal Policy Optimization Algorithms",
arXiv:1707.06347, 2017.

The observation is built exactly as in training (``rl/track_env.py``):
boundaries smoothed and resampled with the training settings, then the
closest + ``n_ahead`` points of each boundary in the vehicle frame
(forward = +x, left = +y), followed by the speed.  Observation
normalisation (VecNormalize) and action clipping are baked into the ONNX
graph by the exporter, so this module feeds raw observations.

Action: ``[steering, throttle]`` with steering +1 = left and throttle in
``[0, 1]`` (forward only, as trained).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from .types import Command, VehicleState, wrap_angle


# --- Observation helpers (must stay identical to workers/boundary_resampler.py) ---

def smooth_boundary(points: np.ndarray, window: int) -> np.ndarray:
    if len(points) < window or window <= 1:
        return points.astype(np.float32)
    kernel = np.ones(window) / window
    px = np.convolve(points[:, 0], kernel, mode="same")
    py = np.convolve(points[:, 1], kernel, mode="same")
    return np.stack([px, py], axis=1).astype(np.float32)


def resample_uniform(points: np.ndarray, spacing: float) -> np.ndarray:
    if len(points) < 2:
        return points.astype(np.float32)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if arc[-1] == 0.0:
        return points[:1].astype(np.float32)
    targets = np.arange(0.0, arc[-1], spacing)
    return np.stack([np.interp(targets, arc, points[:, 0]), np.interp(targets, arc, points[:, 1])],
                    axis=1).astype(np.float32)


def boundary_obs(pos, heading, outer, inner, n_ahead) -> np.ndarray:
    idx_o = int(np.linalg.norm(outer - pos, axis=1).argmin())
    idx_i = int(np.linalg.norm(inner - pos, axis=1).argmin())
    o_pts = outer[np.arange(idx_o, idx_o + n_ahead + 1) % len(outer)]
    i_pts = inner[np.arange(idx_i, idx_i + n_ahead + 1) % len(inner)]
    c, s = np.cos(heading), np.sin(heading)
    R_inv = np.array([[c, s], [-s, c]], dtype=np.float32)
    return np.concatenate([((o_pts - pos) @ R_inv.T).ravel(), ((i_pts - pos) @ R_inv.T).ravel()]).astype(np.float32)


# --- Policy ---

@dataclass
class PolicyMeta:
    n_ahead: int
    boundary_spacing: float
    boundary_smooth_window: int
    obs_dim: int

    @classmethod
    def load(cls, onnx_path: str) -> "PolicyMeta":
        """Read the ``<model>.json`` sidecar written by rl/export_onnx.py."""
        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            return cls(**json.load(f)["observation"])


class PolicyFollower:
    """ONNX PPO policy on mapped track boundaries."""

    def __init__(self, onnx_path: str, outer: np.ndarray, inner: np.ndarray) -> None:
        import onnxruntime as ort

        self.meta = PolicyMeta.load(onnx_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1            # tiny network; don't compete with other workers
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self._input = self.session.get_inputs()[0].name
        m = self.meta
        self.outer = resample_uniform(smooth_boundary(np.asarray(outer)[:, :2], m.boundary_smooth_window), m.boundary_spacing)
        self.inner = resample_uniform(smooth_boundary(np.asarray(inner)[:, :2], m.boundary_smooth_window), m.boundary_spacing)
        self._aligned = False

    def reset(self) -> None:
        self._aligned = False

    def _align(self, state: VehicleState) -> None:
        """Boundaries must be ordered along the driving direction (as in training)."""
        pos = state.position.astype(np.float32)
        i = int(np.linalg.norm(self.outer - pos, axis=1).argmin())
        d = self.outer[(i + 1) % len(self.outer)] - self.outer[i]
        if abs(wrap_angle(np.arctan2(d[1], d[0]) - state.yaw)) > np.pi / 2:
            self.outer, self.inner = self.outer[::-1].copy(), self.inner[::-1].copy()
        self._aligned = True

    def observation(self, state: VehicleState) -> np.ndarray:
        pos = state.position.astype(np.float32)
        bnd = boundary_obs(pos, state.yaw, self.outer, self.inner, self.meta.n_ahead)
        return np.append(bnd, state.v).astype(np.float32)

    def step(self, state: VehicleState) -> Command:
        if not self._aligned:
            self._align(state)
        obs = self.observation(state)[None]
        action = self.session.run(None, {self._input: obs})[0][0]
        return Command(float(action[0]), float(action[1])).clipped()
