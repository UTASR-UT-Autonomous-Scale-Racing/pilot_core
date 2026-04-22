"""
Training loop and ONNX export for the drivable-track networks.

:class:`TrackSegNet` wraps a :func:`vision.models.build_model` network with
ImageNet normalisation, so training and the exported graph both take RGB
in ``[0, 1]``.  ``scripts/train_vision.py`` is the command-line front end.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from .data import augment, bce_dice_loss, load_pair, to_tensor_batch, track_iou
from .models import build_model

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TrackSegNet(nn.Module):
    """Normalisation + network: RGB ``(N, 3, H, W)`` in [0, 1] → logits ``(N, 1, H, W)``."""

    def __init__(self, name: str, pretrained_backbone: bool = False) -> None:
        super().__init__()
        self.name = name
        self.net = build_model(name, pretrained_backbone)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


class _ProbabilityGraph(nn.Module):
    def __init__(self, model: TrackSegNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return torch.sigmoid(self.model(x))


def evaluate(model: TrackSegNet, pairs: List[Tuple[str, str]], size, threshold: float = 0.5) -> float:
    """Mean track IoU over ``pairs``."""
    model.eval()
    ious = []
    with torch.no_grad():
        for img, msk in pairs:
            rgb, mask = load_pair(img, msk, size)
            x, _ = to_tensor_batch([rgb], [mask])
            pred = (torch.sigmoid(model(x))[0, 0].numpy() > threshold)
            ious.append(track_iou(pred, mask))
    return float(np.mean(ious)) if ious else float("nan")


def train(model: TrackSegNet, train_pairs, val_pairs, size, epochs: int = 40, batch_size: int = 8,
          lr: float = 1e-3, seed: int = 0, log=print) -> TrackSegNet:
    """Train with BCE + Dice, AdamW and a cosine learning-rate schedule.

    Keeps the weights with the best validation IoU.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_iou, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        order = rng.permutation(len(train_pairs))
        losses = []
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            if len(idx) < 2:          # BatchNorm needs more than one sample per batch
                continue
            batch = [augment(*load_pair(*train_pairs[i], size), rng) for i in idx]
            x, y = to_tensor_batch([r for r, _ in batch], [m for _, m in batch])
            loss = bce_dice_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        val_iou = evaluate(model, val_pairs, size) if val_pairs else float("nan")
        if not val_pairs or val_iou > best_iou:
            best_iou, best_state = val_iou, {k: v.detach().clone() for k, v in model.state_dict().items()}
        log(f"  epoch {ep + 1:3d}/{epochs}: loss={np.mean(losses):.4f} val IoU={val_iou:.3f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return model.eval()


def export_onnx(model: TrackSegNet, path: str, size, meta: dict | None = None) -> str:
    """Export ``RGB (N, 3, H, W) in [0, 1] → track probability (N, 1, H, W)`` + JSON sidecar."""
    h, w = size
    graph = _ProbabilityGraph(model).eval()      # eval on the wrapper: export traces in its mode
    torch.onnx.export(graph, torch.zeros(1, 3, h, w), path,
                      input_names=["rgb"], output_names=["track_prob"],
                      dynamic_axes={"rgb": {0: "batch"}, "track_prob": {0: "batch"}},
                      opset_version=17, dynamo=False)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({"model": model.name, "input_size": [h, w], "threshold": 0.5, **(meta or {})}, f, indent=2)
    return path
