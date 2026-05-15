"""
Behaviour cloning: end-to-end PilotNet-style CNN, depth frame -> steering.

Ref: M. Bojarski et al., "End to End Learning for Self-Driving Cars", arXiv:1604.07316, 2016.
"""

from __future__ import annotations

from .imitation import INPUT_H, INPUT_W, export_steering_onnx


def build_model():
    """PilotNet-ish CNN, (N, 1, 40, 64) -> (N, 1) steering in [-1, 1]."""
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(1, 24, 5, 2, 2), nn.ELU(),     # 40x64 -> 20x32
        nn.Conv2d(24, 36, 5, 2, 2), nn.ELU(),    # -> 10x16
        nn.Conv2d(36, 48, 5, 2, 2), nn.ELU(),    # -> 5x8
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
    return export_steering_onnx(model, path, meta)


__all__ = ["INPUT_H", "INPUT_W", "build_model", "export_onnx"]
