#!/usr/bin/env python3
"""
Train the behaviour-cloning model (control/bc.py) on the bench and export it
to ONNX for control_mode: bc.

Collects expert demos with DART noise via control/imitation.py (or loads a
cached --data file, same format train_vae_policy.py uses), trains the
PilotNet-style CNN on the steering labels, exports ONNX + a JSON sidecar, then
runs closed loop, map-free, on the held-out oval and twisty tracks.

Needs torch and onnxruntime.
    python scripts/train_bc_policy.py                          # -> bc_policy.onnx
    python scripts/train_bc_policy.py --data demos.npz --epochs 40

NB: a bench-trained model has only ever seen rendered depth. For the car it
needs retraining on recorded real depth frames with expert labels.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import VehicleParams  # noqa: E402
from control.imitation import (  # noqa: E402
    CAM_HEIGHT, DepthPolicyConfig, DepthPolicyFollower, bench_intrinsics, collect_depth_demos,
)
from control.sim import oval_track, run_depth_closed_loop, twisty_track  # noqa: E402


def load_or_collect(args, params, K):
    if args.data and os.path.exists(args.data):
        d = np.load(args.data)
        print(f"      loaded {len(d['x'])} frames from {args.data}")
        return d["x"], d["y"]
    x, y = collect_depth_demos(params, K, args.tracks, args.seconds, args.speed, args.max_range, args.seed)
    if args.data:
        np.savez_compressed(args.data, x=x, y=y)
        print(f"      saved dataset → {args.data}")
    return x, y


def train(x, y, epochs, seed):
    import torch

    from control.bc import build_model

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    model = build_model()
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    idx = torch.randperm(len(X))
    n_val = len(X) // 10
    val, tr = idx[:n_val], idx[n_val:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), 128):
            i = perm[b:b + 128]
            loss = torch.mean((model(X[i]) - Y[i]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                mae = float(torch.mean(torch.abs(model(X[val]) - Y[val])))
            print(f"  epoch {ep + 1:3d}: val steering MAE={mae:.3f}")
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", type=int, default=16, help="random training tracks to drive (default: 16)")
    ap.add_argument("--seconds", type=float, default=45.0, help="expert driving per track")
    ap.add_argument("--speed", type=float, default=0.8, help="expert / cruise speed (m/s)")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs (default: 30)")
    ap.add_argument("--max-range", type=float, default=4.0, help="depth clip / normalisation range, m (default: 4.0)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for tracks, noise and training")
    ap.add_argument("--data", help="dataset cache (.npz): loaded if present (collection flags are then ignored), else written")
    ap.add_argument("--out", default="bc_policy.onnx", help="output ONNX path (+ .json sidecar)")
    ap.add_argument("--eval-seconds", type=float, default=90.0, help="simulated seconds per evaluation run")
    args = ap.parse_args()

    params = VehicleParams.load()
    K = bench_intrinsics()
    t0 = time.time()
    print("[1/4] Demonstrations …")
    x, y = load_or_collect(args, params, K)

    print(f"[2/4] Training BC CNN on {len(x)} frames …")
    model = train(x, y, args.epochs, args.seed)

    print(f"[3/4] Exporting → {args.out}")
    from control.bc import export_onnx

    export_onnx(model, args.out, {"max_range": args.max_range, "camera_height": CAM_HEIGHT,
                                  "model": "bc (PilotNet-style CNN)", "trained_on": "bench-rendered depth",
                                  "cruise_speed": args.speed})

    print("[4/4] Closed-loop evaluation (map-free, held-out tracks) …")
    follower = DepthPolicyFollower(args.out, DepthPolicyConfig(cruise_speed=args.speed))
    for name, (outer, inner, _) in [("oval", oval_track()), ("twisty", twisty_track())]:
        r = run_depth_closed_loop(follower, outer, inner, params, duration=args.eval_seconds,
                                  K=K, camera_height=CAM_HEIGHT)
        print(f"      {name:6s} {r.summary()}")
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
