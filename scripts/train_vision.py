#!/usr/bin/env python3
"""Train a drivable-track segmentation network and export ONNX for VisionWorker.

Data: <data>/images/*.png|jpg and <data>/masks/*.png (non-zero = track). Needs torch, torchvision.

    python scripts/train_vision.py --data datasets/track --model fast_scnn
    python scripts/train_vision.py --data datasets/track --model lraspp --pretrained --epochs 60
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.data import find_pairs  # noqa: E402
from vision.models import MODELS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", required=True, help="dataset root with images/ and masks/")
    ap.add_argument("--model", choices=MODELS, default="fast_scnn", help="network (default: fast_scnn)")
    ap.add_argument("--pretrained", action="store_true",
                    help="start lraspp/deeplabv3 from an ImageNet backbone (downloads weights)")
    ap.add_argument("--size", type=int, nargs=2, default=[192, 320], metavar=("H", "W"),
                    help="network input size (default: 192 320)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val", type=float, default=0.15, help="validation fraction (default: 0.15)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="track_seg.onnx", help="output ONNX path (+ .json sidecar)")
    args = ap.parse_args()

    import torch

    from vision.training import TrackSegNet, evaluate, export_onnx, train

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    pairs = find_pairs(args.data)
    order = np.random.default_rng(args.seed).permutation(len(pairs))
    n_val = int(round(len(pairs) * args.val)) if len(pairs) > 1 else 0
    val = [pairs[i] for i in order[:n_val]]
    tr = [pairs[i] for i in order[n_val:]]
    print(f"{len(tr)} training / {len(val)} validation images, input {args.size[0]}x{args.size[1]}, model {args.model}")

    model = TrackSegNet(args.model, pretrained_backbone=args.pretrained)
    model = train(model, tr, val, tuple(args.size), args.epochs, args.batch_size, args.lr, args.seed)
    val_iou = evaluate(model, val, tuple(args.size)) if val else float("nan")

    export_onnx(model, args.out, tuple(args.size),
                {"val_iou": val_iou, "train_images": len(tr), "pretrained_backbone": args.pretrained})
    print(f"Exported {args.out} (validation IoU {val_iou:.3f}). Use it with vision.model: onnx, "
          f"vision.onnx_model: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
