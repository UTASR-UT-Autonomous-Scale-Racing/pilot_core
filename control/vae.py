"""
VAE control model: a variational auto-encoder compresses each depth frame
into a small latent vector, and a separate controller head maps the latent
to a steering command ("World Models" V + C decomposition).

References: D. Ha, J. Schmidhuber, "World Models", arXiv:1803.10122, 2018;
D. P. Kingma, M. Welling, "Auto-Encoding Variational Bayes", ICLR 2014
(β-weighted KL: I. Higgins et al., "β-VAE", ICLR 2017).  The head is trained
by behaviour cloning with noise-injected demonstrations: M. Laskey et al.,
"DART: Noise Injection for Robust Imitation Learning", CoRL 2017.

On the car the exported ONNX graph (encoder mean → head) runs through the
shared :class:`control.imitation.DepthPolicyFollower`, so torch is only
needed for training (``scripts/train_vae_policy.py``).  Preprocessing,
demonstrations and the follower are shared with the BC model
(:mod:`control.imitation`).
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
    """Export encoder-mean → head as one graph ``depth (N,1,H,W) → steering (N,1)``."""
    from torch import nn

    class Inference(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae, self.head = vae, head

        def forward(self, x):
            mu, _ = self.vae.encode(x)
            return self.head(mu)

    return export_steering_onnx(Inference(), path, meta)
