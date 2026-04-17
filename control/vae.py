"""
VAE control model: a variational auto-encoder compresses each depth frame
into a small latent vector, and a separate controller head maps the latent
to a steering command ("World Models" V + C decomposition).

References: D. Ha, J. Schmidhuber, "World Models", arXiv:1803.10122, 2018;
D. P. Kingma, M. Welling, "Auto-Encoding Variational Bayes", ICLR 2014
(β-weighted KL: I. Higgins et al., "β-VAE", ICLR 2017).  The head is trained
by behaviour cloning with noise-injected demonstrations: M. Laskey et al.,
"DART: Noise Injection for Robust Imitation Learning", CoRL 2017.

On the car only :class:`VAEPolicyFollower` is used: it runs the exported
ONNX graph (encoder mean → head) with onnxruntime, so torch is only needed
for training (``scripts/train_vae_policy.py``).

Input: a depth frame (metres) resized to ``INPUT_H × INPUT_W`` and scaled
to ``[0, 1]`` by ``max_range`` (invalid pixels = 0).  Output: signed
steering (+1 = left); throttle follows a cruise speed that drops with
steering, like ``depth_ftg``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .types import Command

INPUT_H, INPUT_W = 40, 64


def preprocess_depth(depth: np.ndarray, max_range: float) -> np.ndarray:
    """Depth (H, W) metres → ``(INPUT_H, INPUT_W)`` float32 in ``[0, 1]``."""
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0)
    d = np.clip(d, 0.0, max_range) / max_range
    return cv2.resize(d, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------------ training

def build_models(latent_dim: int = 16):
    """Return ``(vae, head)`` torch modules (torch imported lazily)."""
    import torch
    from torch import nn

    class ConvVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),     # 40×64 → 20×32
                nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),    # → 10×16
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),    # → 5×8
                nn.Flatten(),
            )
            self.fc_mu = nn.Linear(64 * 5 * 8, latent_dim)
            self.fc_logvar = nn.Linear(64 * 5 * 8, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, 64 * 5 * 8)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
                nn.ConvTranspose2d(16, 1, 4, 2, 1), nn.Sigmoid(),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def decode(self, z):
            return self.dec(self.fc_dec(z).view(-1, 64, 5, 8))

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decode(z), mu, logvar

    head = nn.Sequential(
        nn.Linear(latent_dim, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1), nn.Tanh(),                      # steering in [-1, 1]
    )
    return ConvVAE(), head


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    """Reconstruction (sum of squared errors) + β·KL, averaged per sample."""
    import torch

    rec = torch.sum((recon - x) ** 2) / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return rec + beta * kl, rec, kl


def export_onnx(vae, head, path: str, meta: dict) -> str:
    """Export encoder-mean → head as one graph ``depth (1,1,H,W) → steering (1,1)``."""
    import torch
    from torch import nn

    class Inference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae, self.head = vae, head

        def forward(self, x):
            mu, _ = self.vae.encode(x)
            return self.head(mu)

    model = Inference().eval()
    torch.onnx.export(model, torch.zeros(1, 1, INPUT_H, INPUT_W), path,
                      input_names=["depth"], output_names=["steering"],
                      dynamic_axes={"depth": {0: "batch"}, "steering": {0: "batch"}},
                      opset_version=17, dynamo=False)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    return path


# ----------------------------------------------------------------- inference

@dataclass
class VAEPolicyConfig:
    cruise_speed: float = 0.8          # m/s
    speed_per_throttle: float = 3.0    # m/s per unit throttle (shared calibration)
    steer_slowdown: float = 0.5        # speed *= 1 - steer_slowdown * |steering|


class VAEPolicyFollower:
    """Map-free controller: depth frame → VAE latent → steering (ONNX)."""

    def __init__(self, onnx_path: str, cfg: VAEPolicyConfig | None = None) -> None:
        import onnxruntime as ort

        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            self.meta = json.load(f)
        self.max_range = float(self.meta["max_range"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.cfg = cfg or VAEPolicyConfig()

    def reset(self) -> None:
        pass

    def steer(self, depth: np.ndarray) -> float:
        x = preprocess_depth(depth, self.max_range)[None, None]
        return float(self.session.run(None, {"depth": x})[0][0, 0])

    def step(self, depth: np.ndarray) -> Command:
        steering = float(np.clip(self.steer(depth), -1.0, 1.0))
        c = self.cfg
        speed = c.cruise_speed * (1.0 - c.steer_slowdown * abs(steering))
        return Command(steering, speed / c.speed_per_throttle).clipped()
