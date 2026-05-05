"""
Runs a PPO policy exported to ONNX (by rl/export_onnx.py) on the CPU through
onnxruntime.

PPO: J. Schulman et al., "Proximal Policy Optimization Algorithms",
arXiv:1707.06347, 2017.

Observation has to match training (rl/track_env.py) exactly; both build it
with control/boundary.py. Boundaries are smoothed and resampled (as closed
loops) with the training settings, then we take the closest
point plus n_ahead more from each boundary, in the vehicle frame (+x
forward, +y left), and append the speed. VecNormalize and action clipping
are baked into the graph by the exporter, so raw observations go in here.

Action is [steering, throttle]. Steering +1 = left, throttle in [0, 1]
(forward only, that's how it was trained).

The sidecar also records the policy rate (policy_hz) and the SafetyConfig
the training env ran with; the caller should step the policy at policy_hz.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .boundary import obs_dim as expected_obs_dim
from .boundary import prepare_boundary, track_observation
from .types import Command, VehicleState, wrap_angle


@dataclass
class PolicyMeta:
    n_ahead: int
    boundary_spacing: float
    boundary_smooth_window: int
    obs_dim: int
    # rate the policy was trained at (env steps per second); None for
    # sidecars exported before it was recorded
    policy_hz: Optional[float] = None
    # SafetyConfig fields the training env ran with (None if not recorded)
    safety: Optional[dict] = field(default=None)

    @classmethod
    def load(cls, onnx_path: str) -> "PolicyMeta":
        """Loads the <model>.json sidecar that rl/export_onnx.py writes."""
        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            data = json.load(f)
        meta = cls(**data["observation"], policy_hz=data.get("policy_hz"), safety=data.get("safety"))
        want = expected_obs_dim(meta.n_ahead)
        if meta.obs_dim != want:
            raise ValueError(
                f"{onnx_path}: sidecar obs_dim={meta.obs_dim} but n_ahead={meta.n_ahead} "
                f"implies (n_ahead+1)*4+1 = {want}; the model does not match this observation"
            )
        return meta


class PolicyFollower:
    """ONNX PPO policy on mapped track boundaries."""

    def __init__(self, onnx_path: str, outer: np.ndarray, inner: np.ndarray) -> None:
        import onnxruntime as ort

        self.meta = PolicyMeta.load(onnx_path)
        opts = ort.SessionOptions()
        # tiny network, no point competing with the other workers for cores
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self._input = inp.name
        m = self.meta
        width = inp.shape[-1] if inp.shape else None
        if isinstance(width, int) and width != m.obs_dim:
            raise ValueError(f"{onnx_path}: ONNX input width {width} != sidecar obs_dim {m.obs_dim}")
        self.outer = prepare_boundary(np.asarray(outer)[:, :2], m.boundary_smooth_window, m.boundary_spacing)
        self.inner = prepare_boundary(np.asarray(inner)[:, :2], m.boundary_smooth_window, m.boundary_spacing)
        self._aligned = False

    @property
    def policy_hz(self) -> Optional[float]:
        """Rate the policy was trained at; step() should be called at this rate."""
        return self.meta.policy_hz

    @property
    def safety(self) -> Optional[dict]:
        """SafetyConfig fields used during training, for comparing with the runtime config."""
        return self.meta.safety

    def reset(self) -> None:
        self._aligned = False

    def _align(self, state: VehicleState) -> None:
        # boundaries have to run along the driving direction, like in training
        pos = state.position.astype(np.float32)
        i = int(np.linalg.norm(self.outer - pos, axis=1).argmin())
        d = self.outer[(i + 1) % len(self.outer)] - self.outer[i]
        if abs(wrap_angle(np.arctan2(d[1], d[0]) - state.yaw)) > np.pi / 2:
            self.outer, self.inner = self.outer[::-1].copy(), self.inner[::-1].copy()
        self._aligned = True

    def observation(self, state: VehicleState) -> np.ndarray:
        pos = state.position.astype(np.float32)
        return track_observation(pos, state.yaw, state.v, self.outer, self.inner, self.meta.n_ahead)

    def step(self, state: VehicleState) -> Command:
        if not self._aligned:
            self._align(state)
        obs = self.observation(state)[None]
        action = self.session.run(None, {self._input: obs})[0][0]
        return Command(float(action[0]), float(action[1])).clipped()
