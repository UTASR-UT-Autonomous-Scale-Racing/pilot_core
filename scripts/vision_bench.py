#!/usr/bin/env python3
"""
Compare the drivable-track segmentation candidates on size and CPU latency.

Each network is exported to ONNX (untrained weights — latency does not depend
on training) and timed with onnxruntime, the way VisionWorker runs it.  The
training-free ground_plane segmenter is timed on a depth frame of a flat
floor.  Accuracy needs labelled data: see scripts/train_vision.py.

Usage (needs torch and torchvision; run from the repo root):
    python scripts/vision_bench.py
    python scripts/vision_bench.py --size 360 640 --threads 4
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from vision.models import MODELS  # noqa: E402


def time_call(fn, repeats: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--size", type=int, nargs=2, default=[192, 320], metavar=("H", "W"),
                    help="network input size (default: 192 320)")
    ap.add_argument("--frame", type=int, nargs=2, default=[400, 640], metavar=("H", "W"),
                    help="camera frame size (default: 400 640)")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads (default: 1, as in the pipeline)")
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import torch

    from vision.segmenters import GroundPlaneSegmenter, OnnxTrackSegmenter
    from vision.training import TrackSegNet, export_onnx

    h, w = args.frame
    rgb = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    print(f"frame {w}x{h}, network input {args.size[1]}x{args.size[0]}, {args.threads} thread(s)\n")
    print(f"{'candidate':12s} {'params':>9s} {'onnx MB':>8s} {'latency':>10s}")
    with tempfile.TemporaryDirectory() as tmp:
        for name in MODELS:
            torch.manual_seed(0)
            model = TrackSegNet(name)
            path = export_onnx(model, os.path.join(tmp, f"{name}.onnx"), tuple(args.size))
            seg = OnnxTrackSegmenter(path)
            ms = time_call(lambda: seg.get_mask(None, rgb), args.repeats)
            params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"{name:12s} {params:8.2f}M {os.path.getsize(path) / 1e6:8.1f} {ms:8.1f} ms")

    from control.depth_ftg import Intrinsics

    K = Intrinsics.from_orbslam_settings(os.path.join(REPO_ROOT, "orbslam_oakd_rgbd.yaml"))
    K = Intrinsics(K.fx * w / 640, K.fy * h / 400, K.cx * w / 640, K.cy * h / 400)
    v = np.arange(h)[:, None].repeat(w, axis=1).astype(np.float32)
    ray_y = (v - K.cy) / K.fy
    depth = np.where(ray_y > 1e-3, 0.12 / np.maximum(ray_y, 1e-3), 0.0).astype(np.float32)   # flat floor, camera 12 cm up
    seg = GroundPlaneSegmenter(K)
    ms = time_call(lambda: seg.get_mask(depth, rgb), args.repeats)
    print(f"{'ground_plane':12s} {'-':>9s} {'-':>8s} {ms:8.1f} ms   (no training)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
