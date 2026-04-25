"""
VAE control model. A variational autoencoder squeezes each depth frame into a
small latent, and a separate head maps latent -> steering. This is the
"World Models" V + C split.

D. Ha, J. Schmidhuber, "World Models", arXiv:1803.10122, 2018;
D. P. Kingma, M. Welling, "Auto-Encoding Variational Bayes", ICLR 2014
(beta-weighted KL from I. Higgins et al., "β-VAE", ICLR 2017). Head is trained
by behaviour cloning on noise-injected demos, see M. Laskey et al., "DART:
Noise Injection for Robust Imitation Learning", CoRL 2017.

On the car, the exported ONNX graph (encoder mean -> head) runs through
control.imitation.DepthPolicyFollower, same as BC. So torch is only needed
for training (scripts/train_vae_policy.py). Preprocessing, demos and the
follower all live in control.imitation.
"""

from __future__ import annotations

from .imitation import (  # re-exported for callers of control.vae
    INPUT_H,
    INPUT_W,
    DepthPolicyConfig as VAEPolicyConfig,
    DepthPolicyFollower as VAEPolicyFollower,
    export_steering_onnx,
    preprocess_depth,
)

__all__ = [
    "INPUT_H", "INPUT_W", "VAEPolicyConfig", "VAEPolicyFollower", "build_models",
    "export_onnx", "preprocess_depth", "vae_loss",
]


def build_models(latent_dim: int = 16):
    """(vae, head) torch modules. torch is imported lazily."""
    import torch
    from torch import nn

    class ConvVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.latent_dim = latent_dim
            self.enc = nn.Sequential(
                nn.Conv2d(1, 16, 4, 2, 1), nn.ReLU(),     # 40x64 -> 20x32
                nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),    # -> 10x16
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),    # -> 5x8
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
    """SSE reconstruction + beta * KL, averaged per sample."""
    import torch

    rec = torch.sum((recon - x) ** 2) / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return rec + beta * kl, rec, kl


def export_onnx(vae, head, path: str, meta: dict) -> str:
    """Exports encoder-mean -> head as a single depth -> steering graph."""
    from torch import nn

    class Inference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae, self.head = vae, head

        def forward(self, x):
            mu, _ = self.vae.encode(x)
            return self.head(mu)

    return export_steering_onnx(Inference(), path, meta)
