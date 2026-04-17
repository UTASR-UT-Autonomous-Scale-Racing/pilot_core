#!/usr/bin/env python3
"""
Train the VAE control model (control/vae.py) on the closed-loop bench and
export it to ONNX for ``control_mode: vae``.

    1. Collect: an LQR expert drives random tracks from the true state while
       depth frames are rendered.  Steering noise is injected into the
       executed command (DART) so the data contains recoveries; the label is
       always the expert's clean command.  Frames are mirrored (steering
       negated) to balance left and right turns.
    2. Train the VAE (unsupervised) on the frames.
    3. Train the controller head on the frozen latent means (behaviour cloning).
    4. Export encoder-mean → head to ONNX (+ JSON sidecar).
    5. Evaluate closed loop, map-free, on the held-out oval and twisty tracks.

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

from control import TrackModel, VehicleParams, VehicleState  # noqa: E402
from control.controllers import build_track_follower  # noqa: E402
from control.depth_ftg import Intrinsics  # noqa: E402
from control.sim import (  # noqa: E402
    BicyclePlant, PlantConfig, oval_track, random_track, render_depth, run_depth_closed_loop, twisty_track,
)
from control.vae import VAEPolicyConfig, VAEPolicyFollower, preprocess_depth  # noqa: E402

W, H = 160, 100
CAM_HEIGHT = 0.12


def camera_intrinsics() -> Intrinsics:
    """OAK-D RGB intrinsics scaled from 640×400 to the rendered 160×100."""
    K = Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
    return Intrinsics(K.fx * W / 640, K.fy * H / 400, K.cx * W / 640, K.cy * H / 400)


def collect(params, K, n_tracks, seconds, speed, max_range, seed, noise_std=0.3, dt=0.01, rate=15.0):
    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for k in range(n_tracks):
        outer, inner, _ = random_track(rng)
        expert = build_track_follower("lqr", TrackModel(outer, inner), params, speed_cfg={"v_max": speed})
        tr = expert.track
        car = BicyclePlant(params, PlantConfig(), *tr.points[0], tr.heading[0])
        noise, t, next_frame = 0.0, 0.0, 0.0
        while t < seconds:
            cmd = expert.step(VehicleState(t, car.x, car.y, car.yaw, car.v))
            if t >= next_frame:
                d = render_depth(car.x, car.y, car.yaw, [outer, inner], K, W, H, cam_height=CAM_HEIGHT,
                                 max_range=max_range)
                frames.append(preprocess_depth(d, max_range))
                labels.append(cmd.steering)
                next_frame += 1.0 / rate
            # DART: correlated steering noise on the executed command only
            noise = 0.95 * noise + rng.normal(0, noise_std * np.sqrt(1 - 0.95 ** 2))
            cmd.steering = float(np.clip(cmd.steering + noise, -1, 1))
            car.step(cmd, t, dt)
            t += dt
        print(f"  track {k + 1}/{n_tracks}: {len(frames)} frames")
    x = np.stack(frames)[:, None]
    y = np.array(labels, dtype=np.float32)[:, None]
    # Mirror augmentation: flip left↔right, negate steering
    return np.concatenate([x, x[..., ::-1]]).astype(np.float32), np.concatenate([y, -y])


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
    ap.add_argument("--tracks", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=45.0, help="expert driving per track")
    ap.add_argument("--speed", type=float, default=0.8, help="expert / cruise speed (m/s)")
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--head-epochs", type=int, default=100)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--max-range", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="vae_policy.onnx")
    ap.add_argument("--eval-seconds", type=float, default=90.0)
    args = ap.parse_args()

    params = VehicleParams.load()
    K = camera_intrinsics()

    t0 = time.time()
    print(f"[1/4] Collecting expert demonstrations on {args.tracks} random tracks …")
    x, y = collect(params, K, args.tracks, args.seconds, args.speed, args.max_range, args.seed)
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
    follower = VAEPolicyFollower(args.out, VAEPolicyConfig(cruise_speed=args.speed))
    for name, (outer, inner, _) in [("oval", oval_track()), ("twisty", twisty_track())]:
        r = run_depth_closed_loop(follower, outer, inner, params, duration=args.eval_seconds,
                                  K=K, camera_height=CAM_HEIGHT)
        print(f"      {name:6s} {r.summary()}")
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
