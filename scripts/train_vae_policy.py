#!/usr/bin/env python3
"""
Train the VAE control model (control/vae.py) on the closed-loop bench and
export it to ONNX for ``control_mode: vae``.

    1. Collect expert demonstrations with DART noise (control/imitation.py),
       or load a cached dataset (--data; shared with train_bc_policy.py).
    2. Train the VAE (unsupervised) on the frames.
    3. Train the controller head on the frozen latent means (behaviour cloning).
    4. Export encoder-mean → head to ONNX (+ JSON sidecar).
    5. Evaluate closed loop, map-free, on the held-out oval and twisty tracks.

Usage (needs torch and onnxruntime):
    python scripts/train_vae_policy.py                    # → vae_policy.onnx
    python scripts/train_vae_policy.py --tracks 24 --epochs 40 --out models/vae.onnx

Bench-trained models learn the *rendered* depth; for the car, retrain the
same pipeline on recorded real depth frames with expert labels.
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


def train(x, y, latent_dim, epochs, head_epochs, beta, seed):
    import torch

    from control.vae import build_models, vae_loss

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    vae, head = build_models(latent_dim)
    X, Y = torch.from_numpy(x), torch.from_numpy(y)
    n = len(X)
    idx = torch.randperm(n)
    n_val = n // 10
    val, tr = idx[:n_val], idx[n_val:]

    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for ep in range(epochs):
        vae.train()
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), 128):
            xb = X[perm[b:b + 128]]
            recon, mu, logvar = vae(xb)
            loss, _, _ = vae_loss(recon, xb, mu, logvar, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            vae.eval()
            with torch.no_grad():
                recon, mu, logvar = vae(X[val])
                _, rec, kl = vae_loss(recon, X[val], mu, logvar, beta)
            rmse = float(torch.sqrt(torch.mean((recon - X[val]) ** 2)))
            print(f"  VAE epoch {ep + 1:3d}: val recon RMSE={rmse:.4f} (normalised depth), KL={float(kl):.1f}")

    # Controller head on frozen latent means
    vae.eval()
    with torch.no_grad():
        MU = torch.cat([vae.encode(X[b:b + 1024])[0] for b in range(0, n, 1024)])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    for ep in range(head_epochs):
        head.train()
        perm = tr[torch.randperm(len(tr))]
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]
            loss = torch.mean((head(MU[i]) - Y[i]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 20 == 19 or ep == head_epochs - 1:
            head.eval()
            with torch.no_grad():
                val_mae = float(torch.mean(torch.abs(head(MU[val]) - Y[val])))
            print(f"  head epoch {ep + 1:3d}: val steering MAE={val_mae:.3f}")
    return vae.eval(), head.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tracks", type=int, default=16, help="random training tracks to drive (default: 16)")
    ap.add_argument("--seconds", type=float, default=45.0, help="expert driving per track")
    ap.add_argument("--speed", type=float, default=0.8, help="expert / cruise speed (m/s)")
    ap.add_argument("--latent", type=int, default=16, help="VAE latent dimension (default: 16)")
    ap.add_argument("--epochs", type=int, default=30, help="training epochs (default: 30)")
    ap.add_argument("--head-epochs", type=int, default=100, help="controller-head epochs (default: 100)")
    ap.add_argument("--beta", type=float, default=1.0, help="KL weight in the VAE loss (default: 1.0)")
    ap.add_argument("--max-range", type=float, default=4.0, help="depth clip / normalisation range, m (default: 4.0)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for tracks, noise and training")
    ap.add_argument("--data", help="dataset cache (.npz): loaded if present (collection flags are then ignored), else written")
    ap.add_argument("--out", default="vae_policy.onnx", help="output ONNX path (+ .json sidecar)")
    ap.add_argument("--eval-seconds", type=float, default=90.0, help="simulated seconds per evaluation run")
    args = ap.parse_args()

    params = VehicleParams.load()
    K = bench_intrinsics()

    t0 = time.time()
    print("[1/4] Demonstrations …")
    x, y = load_or_collect(args, params, K)
    print(f"      {len(x)} frames (with mirroring) in {time.time() - t0:.0f}s")

    print("[2/4] Training VAE + controller head …")
    vae, head = train(x, y, args.latent, args.epochs, args.head_epochs, args.beta, args.seed)

    print(f"[3/4] Exporting → {args.out}")
    from control.vae import INPUT_H, INPUT_W, export_onnx

    export_onnx(vae, head, args.out, {
        "input": [INPUT_H, INPUT_W], "max_range": args.max_range, "latent_dim": args.latent,
        "camera_height": CAM_HEIGHT, "trained_on": "bench-rendered depth", "cruise_speed": args.speed,
    })

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
