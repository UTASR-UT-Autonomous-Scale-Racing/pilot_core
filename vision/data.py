"""
Dataset, loss and metrics for training the drivable-track networks.

Dataset layout (one mask per image, same file stem):

    <root>/images/<name>.png|.jpg     RGB camera frames
    <root>/masks/<name>.png           drivable track = non-zero, everything else = 0

Masks from the SAM2 auto-labelling tools in the pilot_vision repository fit this
layout once exported as single-channel PNGs; scripts/recording_to_dataset.py
exports recorded frames into it.

Loss: binary cross-entropy + soft Dice (F. Milletari, N. Navab, S.-A. Ahmadi,
"V-Net", 3DV 2016), which keeps small or thin track regions from being
ignored.  Metric: intersection-over-union (IoU) of the track class.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def find_pairs(root: str) -> List[Tuple[str, str]]:
    """``(image_path, mask_path)`` pairs under ``root/images`` and ``root/masks``."""
    img_dir, mask_dir = os.path.join(root, "images"), os.path.join(root, "masks")
    pairs = []
    for name in sorted(os.listdir(img_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        mask = os.path.join(mask_dir, stem + ".png")
        if os.path.exists(mask):
            pairs.append((os.path.join(img_dir, name), mask))
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}/images and {root}/masks")
    return pairs


def load_pair(img_path: str, mask_path: str, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load and resize to ``size = (H, W)``: RGB uint8 (H, W, 3), mask uint8 {0, 1} (H, W)."""
    h, w = size
    rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return rgb, (mask > 0).astype(np.uint8)


def augment(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Horizontal flip plus brightness / contrast / saturation jitter."""
    if rng.random() < 0.5:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return rgb, np.ascontiguousarray(mask)


def to_tensor_batch(rgbs: List[np.ndarray], masks: List[np.ndarray]):
    """RGB uint8 → float [0, 1] ``(N, 3, H, W)``; masks → float ``(N, 1, H, W)``."""
    import torch

    x = torch.from_numpy(np.stack(rgbs)).permute(0, 3, 1, 2).float() / 255.0
    y = torch.from_numpy(np.stack(masks)).unsqueeze(1).float()
    return x, y


def bce_dice_loss(logits, target, eps: float = 1.0):
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)
    return bce + dice.mean()


def track_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """IoU of the drivable-track class (1.0 if both masks are empty)."""
    pred, target = pred.astype(bool), target.astype(bool)
    union = np.logical_or(pred, target).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, target).sum() / union)
