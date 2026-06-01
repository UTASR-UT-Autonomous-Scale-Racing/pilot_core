#!/usr/bin/env python3
"""Fit the stock AutoDRIVE sim camera for the LiDAR depth adapter (workers/lidar_depth.py)
from a start.py --recording run (frame + lidar buffers, the car moving):

1. car-body mask: the car's own chassis fills the bottom of the front camera's view and
   is static in the image while the world moves; pixels with low temporal variance that
   are connected to the bottom edge (the sky is static too, but at the top).
2. vertical FOV, pitch, camera height, wall height: the depth image predicted from each
   frame's scan has floor -> wall and wall -> nothing boundaries; the parameters that put
   those boundaries on the strongest vertical intensity edges of the real images win
   (coarse-to-fine grid search, mean |dI/dv| along the predicted boundaries).

    python scripts/autodrive_camera_fit.py <recording dir> [<more>] --out sim_camera/

Writes sim_camera/car_mask.png, sim_camera/camera.yaml (CameraModel fields),
sim_camera/orbslam_settings.yaml (ORB-SLAM3 RGB-D, images upscaled --scale) and
fit.png (the fitted boundaries over a few frames).
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import cv2  # noqa: E402
import yaml  # noqa: E402

from workers.lidar_depth import CameraModel, LidarDepth  # noqa: E402


def load_pairs(recs: list[str], max_pairs: int = 400):
    """(gray float image, rgb, ranges) for frames with a scan from the same sim message."""
    from read_recording import load_buffer

    out = []
    for rec in recs:
        frames = load_buffer(rec, "frame")
        scans = load_buffer(rec, "lidar")
        st = np.array([s.produced_ts for s in scans])
        for f in frames:
            k = int(np.argmin(np.abs(st - f.produced_ts)))
            if abs(st[k] - f.produced_ts) < 0.02:
                rgb = f.value[0]
                out.append((cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32), rgb, scans[k].value))
    if len(out) > max_pairs:
        out = [out[i] for i in np.linspace(0, len(out) - 1, max_pairs).astype(int)]
    return out


def car_mask(rgbs: list[np.ndarray], percentile: float = 85, dilate: int = 3) -> np.ndarray:
    """The car's own body. Per-pixel 85th-percentile frame (the asphalt bright, the dark
    chassis in front of it), GrabCut seeded with the top rows as scene and the bottom
    centre as car, then every column filled from the car's top contour down (the body
    occludes what is below it) and dilated."""
    ref = np.percentile(np.stack(rgbs).astype(np.float32), percentile, axis=0).astype(np.uint8)
    h, w = ref.shape[:2]
    up = 4
    big = cv2.resize(cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), (w * up, h * up), interpolation=cv2.INTER_CUBIC)
    m = np.full(big.shape[:2], cv2.GC_PR_BGD, np.uint8)
    m[: int(0.35 * h * up)] = cv2.GC_BGD
    m[int(0.80 * h * up):, int(0.45 * w * up): int(0.55 * w * up)] = cv2.GC_FGD
    cv2.grabCut(big, m, None, np.zeros((1, 65)), np.zeros((1, 65)), 8, cv2.GC_INIT_WITH_MASK)
    fg = cv2.resize(((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_AREA) > 0
    mask = column_fill(fg)
    # parts of the car that GrabCut calls scene (the bright lights on the chassis): grow
    # the mask into static pixels (never change in the image) just above it
    gray = np.stack([cv2.cvtColor(r, cv2.COLOR_RGB2GRAY) for r in rgbs]).astype(np.float32)
    # never changes, or always bright (the lights: their bloom flickers but never goes dark)
    static = (gray.std(axis=0) < 8.0) | (np.percentile(gray, 10, axis=0) > 120)
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0     # bridge to the parts
    for _ in range(8):
        grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        grown &= static | mask
        if (grown == mask).all():
            break
        mask = grown
    mask = column_fill(mask)
    if dilate:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8)) > 0
    return mask


def column_fill(m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(m)
    for u in range(m.shape[1]):
        rows = np.flatnonzero(m[:, u])
        if len(rows):
            out[rows[0]:, u] = True
    return out


def classes(depth_fn: LidarDepth, ranges: np.ndarray) -> np.ndarray:
    """0 nothing, 1 wall, 2 floor (the mask is applied by LidarDepth)."""
    cam = depth_fn.cam
    D = np.interp(depth_fn.bearing, depth_fn.grid, depth_fn.wall_distance(ranges))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = D / depth_fn.d_h
        z_wall = cam.cam_height + t_wall * depth_fn.d_up
        t_floor = np.where(depth_fn.d_up < -1e-6, cam.cam_height / -depth_fn.d_up, np.inf)
    c = np.zeros(D.shape, np.int8)
    floor = t_floor < t_wall
    c[floor] = 2
    c[~floor & np.isfinite(t_wall) & (z_wall >= 0) & (z_wall <= cam.wall_height)] = 1
    return c


def score(params: dict, pairs, grads, mask, which=("base", "top")) -> float:
    cam = CameraModel(**params)
    fn = LidarDepth(cam, bearing_step_deg=0.25)
    fn.mask = None
    vals = []
    for (g, _, ranges), G in zip(pairs, grads):
        c = classes(fn, ranges)
        a, b = c[:-1], c[1:]
        sel = np.zeros(a.shape, bool)
        if "base" in which:
            sel |= (a == 1) & (b == 2)          # wall above floor
        if "top" in which:
            sel |= (a == 0) & (b == 1)          # nothing above wall
        sel &= ~mask[:-1] & ~mask[1:]
        if sel.sum() > 5:
            vals.append(float(G[:-1][sel].mean()))
    return float(np.mean(vals)) if vals else 0.0


def fit(pairs, mask) -> tuple[dict, dict]:
    grads = [np.abs(np.diff(cv2.GaussianBlur(g, (3, 3), 0), axis=0, append=g[-1:])) for g, _, _ in pairs]
    sub = pairs[:: max(1, len(pairs) // 40)]
    gsub = grads[:: max(1, len(pairs) // 40)]
    best, log = {"vfov_deg": 60.0, "pitch_deg": 10.0, "cam_height": 0.21, "wall_height": 5.0}, {}
    # 1. floor/wall boundary: fov x pitch at the devkit camera height, walls "infinitely" tall
    grid = list(itertools.product(range(20, 110, 3), np.arange(-4, 26, 1.5)))
    res = [(score({**best, "vfov_deg": f, "pitch_deg": p}, sub, gsub, mask, ("base",)), f, p) for f, p in grid]
    s, f, p = max(res)
    best.update(vfov_deg=float(f), pitch_deg=float(p))
    log["stage1"] = {"score": s, **best}
    # 2. refine fov, pitch, height together
    grid = list(itertools.product(np.arange(f - 3, f + 3.1, 1.0), np.arange(p - 1.5, p + 1.6, 0.5),
                                  np.arange(0.15, 0.31, 0.02)))
    res = [(score({**best, "vfov_deg": a, "pitch_deg": b, "cam_height": h}, sub, gsub, mask, ("base",)), a, b, h)
           for a, b, h in grid]
    s, a, b, h = max(res)
    best.update(vfov_deg=float(a), pitch_deg=float(b), cam_height=float(h))
    log["stage2"] = {"score": s, **best}
    # 3. wall height from the wall top edge
    res = [(score({**best, "wall_height": w}, sub, gsub, mask, ("top",)), w) for w in np.arange(0.08, 0.81, 0.02)]
    s, w = max(res)
    best["wall_height"] = float(w)
    log["stage3"] = {"score": s, **best}
    log["full_score_fitted"] = score(best, pairs, grads, mask)
    log["full_score_devkit_guess"] = score({"vfov_deg": 60.0, "pitch_deg": 10.0, "cam_height": 0.21,
                                            "wall_height": 0.3}, pairs, grads, mask)
    return best, log


def overlay(pairs, mask, params, path: str) -> None:
    cam = CameraModel(**params)
    fn = LidarDepth(cam)
    fn.mask = None
    tiles = []
    for g, rgb, ranges in pairs[:: max(1, len(pairs) // 6)][:6]:
        c = classes(fn, ranges)
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        img[mask] = (img[mask] * 0.4 + np.array([0, 0, 120]) * 0.6).astype(np.uint8)
        base = (c[:-1] == 1) & (c[1:] == 2)
        top = (c[:-1] == 0) & (c[1:] == 1)
        img[:-1][base] = (0, 255, 0)
        img[:-1][top] = (255, 0, 255)
        tiles.append(cv2.resize(img, (384, 216), interpolation=cv2.INTER_NEAREST))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    cv2.imwrite(path, np.vstack(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "sim_camera"))
    ap.add_argument("--scale", type=float, default=2.0, help="upscale for ORB-SLAM3 (settings file)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pairs = load_pairs(a.recordings)
    print(f"{len(pairs)} frame + scan pairs")
    mask = car_mask([rgb for _, rgb, _ in pairs])
    cv2.imwrite(os.path.join(a.out, "car_mask.png"), mask.astype(np.uint8) * 255)
    print(f"car mask: {mask.mean() * 100:.1f}% of the image")
    best, log = fit(pairs, mask)
    best.update(width=int(pairs[0][0].shape[1]), height=int(pairs[0][0].shape[0]),
                mask_path=os.path.relpath(os.path.join(a.out, "car_mask.png"), REPO_ROOT))
    with open(os.path.join(a.out, "camera.yaml"), "w") as f:
        yaml.safe_dump({"camera_model": best, "fit": log}, f, sort_keys=False)
    overlay(pairs, mask, {k: v for k, v in best.items() if k != "mask_path"}, os.path.join(a.out, "fit.png"))
    cam = CameraModel(**{**best, "scale": a.scale})
    with open(os.path.join(a.out, "orbslam_settings.yaml"), "w") as f:
        f.write(cam.orbslam_settings())
    print(yaml.safe_dump(log, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
