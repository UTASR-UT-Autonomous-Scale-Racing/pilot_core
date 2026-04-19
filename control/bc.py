"""
Behaviour-cloning (BC) control model: an end-to-end CNN maps a depth frame
directly to a steering command, trained by supervised regression on expert
demonstrations.

References: M. Bojarski et al., "End to End Learning for Self-Driving Cars"
(PilotNet), arXiv:1604.07316, 2016; D. A. Pomerleau, "ALVINN: An
Autonomous Land Vehicle in a Neural Network", NIPS 1989.  Demonstrations
use DART noise injection (M. Laskey et al., CoRL 2017), see
:mod:`control.imitation`.

The network follows PilotNet's layout (strided 5×5 convolutions, then 3×3
convolutions, then a narrowing fully-connected stack), scaled down to the
40×64 depth input.  Unlike the VAE model there is no separately learned
latent: the whole network is trained on the steering loss.

On the car the exported ONNX graph runs through
:class:`control.imitation.DepthPolicyFollower` (``control_mode: bc``).
Train with ``scripts/train_bc_policy.py``.
"""

from __future__ import annotations

from .imitation import INPUT_H, INPUT_W, export_steering_onnx


def build_model():
    """PilotNet-style CNN ``(N, 1, 40, 64) → (N, 1)`` steering in [-1, 1]."""
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 24, 5, 2, 2), nn.ELU(),     # 40×64 → 20×32
        nn.Conv2d(24, 36, 5, 2, 2), nn.ELU(),    # → 10×16
        nn.Conv2d(36, 48, 5, 2, 2), nn.ELU(),    # → 5×8
        nn.Conv2d(48, 64, 3, 1, 1), nn.ELU(),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ELU(),
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(64 * 5 * 8, 100), nn.ELU(),
        nn.Linear(100, 50), nn.ELU(),
        nn.Linear(50, 10), nn.ELU(),
        nn.Linear(10, 1), nn.Tanh(),
    )


def export_onnx(model, path: str, meta: dict) -> str:
    """Export the CNN as ``depth (N,1,H,W) → steering (N,1)``."""
    return export_steering_onnx(model, path, meta)


__all__ = ["INPUT_H", "INPUT_W", "build_model", "export_onnx"]
