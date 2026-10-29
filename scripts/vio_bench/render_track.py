#!/usr/bin/env python3
"""Render a geometrically exact RGB-D-inertial sequence of a sim run's track and trajectory.

    python scripts/vio_bench/render_track.py <sim seq dir> <run dir> --out <seq dir> \
        [--blackout 4.0:0.3] [--workers 16]

The sim sequences (sim_to_rgbdi.py) take their depth from the 2-D LiDAR and their images
from the sim's renderer, and the two do not agree well enough for a tracker that builds
its map from depth (ORB-SLAM3): only about half of its matches between consecutive frames
fit the depth. This renders the same track (the run's boundary.npz: floor plus walls
along the inner and outer boundary, a far backdrop) from the same ground-truth camera
poses, with a non-repeating procedural texture and exact depth, and keeps the sim
sequence's IMU, calibration and ground truth: image, depth, IMU and ground truth are
exactly consistent. What is left to differ is the trackers themselves.

--blackout START:DUR (s from the first frame, repeatable) replaces those frames with a
featureless grey image: a forced tracking loss, e.g. before the IMU is initialised.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")      # one thread per worker: BLAS pools x workers exhaust the host's PIDs

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402

WALL_HEIGHT = 0.30       # m
BACKDROP_HEIGHT = 4.0    # m
MAX_DEPTH = 10.0         # m, farther depth is written as 0 (no depth), like the LiDAR depth
SUPERSAMPLE = 2


def _hash(i: np.ndarray, j: np.ndarray, seed: float) -> np.ndarray:
    return np.modf(np.abs(np.sin(i * 127.1 + j * 311.7 + seed * 74.7) * 43758.5453))[0]


def value_noise(x: np.ndarray, y: np.ndarray, seed: float) -> np.ndarray:
    """Smooth lattice noise in [0, 1] with unit cell size."""
    i, j = np.floor(x), np.floor(y)
    fx, fy = x - i, y - j
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a, b = _hash(i, j, seed), _hash(i + 1, j, seed)
    c, d = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def texture(u: np.ndarray, v: np.ndarray, footprint: np.ndarray, seed: float, scales) -> np.ndarray:
    """Multi-octave noise in [0, 255] at surface coordinates (u, v) m; octaves finer than
    ~2 footprints (m per rendered pixel) fade out instead of aliasing."""
    acc = np.zeros_like(u)
    wsum = np.zeros_like(u)
    for k, s in enumerate(scales):
        w = np.clip((s / np.maximum(footprint, 1e-9) - 2.0) / 2.0, 0.0, 1.0)
        acc += w * (value_noise(u / s, v / s, seed + 13 * k) - 0.5)
        wsum += w
    n = 0.5 + 1.8 * acc / np.maximum(np.sqrt(np.maximum(wsum, 1e-9)), 1.0)
    return np.clip(255 * n, 0, 255)


class Scene:
    def __init__(self, inner: np.ndarray, outer: np.ndarray) -> None:
        segs, arcs, seeds = [], [], []
        for k, poly in enumerate((inner, outer)):
            p = np.asarray(poly, np.float64)
            if np.linalg.norm(p[0] - p[-1]) > 1e-6:
                p = np.vstack([p, p[:1]])                # closed loop
            e = np.diff(p, axis=0)
            L = np.linalg.norm(e, axis=1)
            segs.append(np.hstack([p[:-1], e]))
            arcs.append(np.concatenate([[0.0], np.cumsum(L)[:-1]]))
            seeds.append(np.full(len(e), 100.0 + 50 * k))
        self.seg = np.vstack(segs)                      # (S, 4): p0x, p0y, ex, ey
        self.arc = np.concatenate(arcs)                 # arc length at p0
        self.len = np.linalg.norm(self.seg[:, 2:], axis=1)
        self.seed = np.concatenate(seeds)
        allp = np.vstack([inner, outer])
        self.centre = allp.mean(axis=0)
        self.r_backdrop = float(np.linalg.norm(allp - self.centre, axis=1).max()) + 8.0

    def cast(self, T_wc: np.ndarray, K: tuple, w: int, h: int, colour: bool) -> np.ndarray:
        """Optical depth (h, w) of the first hit, or grey image if colour."""
        fx, fy, cx, cy = K
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        rc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], -1).reshape(-1, 3)
        R, o = T_wc[:3, :3], T_wc[:3, 3]
        d = rc @ R.T                                    # world ray per pixel, optical z = 1
        n = len(d)
        t = np.full(n, np.inf)
        kind = np.zeros(n, np.int8)                     # 0 sky, 1 floor, 2 wall, 3 backdrop
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.where(d[:, 2] < -1e-9, -o[2] / d[:, 2], np.inf)
        t, kind = np.where(tf < t, tf, t), np.where(tf < t, 1, kind)
        # walls: segments within reach, vertical from 0 to WALL_HEIGHT
        rel = self.seg[:, :2] - o[:2]
        near = np.minimum(np.linalg.norm(rel, axis=1), np.linalg.norm(rel + self.seg[:, 2:], axis=1)) < MAX_DEPTH * 1.5
        ws, wsg = np.full(n, 0.0), np.full(n, -1)
        for k in np.nonzero(near)[0]:
            p0x, p0y, ex, ey = self.seg[k]
            den = d[:, 0] * ey - d[:, 1] * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                tk = ((p0x - o[0]) * ey - (p0y - o[1]) * ex) / den
                sk = ((p0x - o[0]) * d[:, 1] - (p0y - o[1]) * d[:, 0]) / den
            z = o[2] + tk * d[:, 2]
            hit = (tk > 1e-6) & (sk >= 0) & (sk <= 1) & (z >= 0) & (z <= WALL_HEIGHT) & (tk < t)
            t[hit], kind[hit], ws[hit], wsg[hit] = tk[hit], 2, sk[hit], k
        # backdrop: vertical cylinder around the track
        a = d[:, 0] ** 2 + d[:, 1] ** 2
        oc = o[:2] - self.centre
        b = 2 * (oc[0] * d[:, 0] + oc[1] * d[:, 1])
        c = oc @ oc - self.r_backdrop ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            tb = (-b + np.sqrt(np.maximum(b * b - 4 * a * c, 0))) / (2 * a)
        zb = o[2] + tb * d[:, 2]
        hit = (tb > 0) & (zb <= BACKDROP_HEIGHT) & (tb < t)
        t, kind = np.where(hit, tb, t), np.where(hit, 3, kind)
        if not colour:
            depth = np.where((kind > 0) & (kind < 3) & (t <= MAX_DEPTH), t, 0.0)
            return depth.reshape(h, w)

        P = o + d * np.where(np.isfinite(t), t, 0.0)[:, None]
        foot = np.where(np.isfinite(t), t * np.linalg.norm(d, axis=1), 1e3) / fx
        img = np.full(n, 190.0)
        m = kind == 1
        img[m] = texture(P[m, 0], P[m, 1], foot[m], 1.0, (0.6, 0.2, 0.07, 0.03))
        m = kind == 2
        g = wsg[m]
        img[m] = 0.8 * texture(self.arc[g] + ws[m] * self.len[g], P[m, 2], foot[m], 7.0, (0.3, 0.1, 0.04)) \
            + 0.2 * np.where(self.seed[g] > 120, 40.0, 210.0)
        m = kind == 3
        ang = np.arctan2(P[m, 1] - self.centre[1], P[m, 0] - self.centre[0]) * self.r_backdrop
        img[m] = 0.6 * texture(ang, P[m, 2], foot[m], 31.0, (2.0, 0.7, 0.25)) + 0.4 * 120
        return img.reshape(h, w)


_SCENE: Scene | None = None


def _init(inner, outer):
    global _SCENE
    _SCENE = Scene(inner, outer)


def _render(job):
    t_ns, T, K, w, h, out, blank = job
    if blank:
        img = np.full((h, w), 128, np.uint8)
    else:
        s = SUPERSAMPLE
        fx, fy, cx, cy = K
        hi = _SCENE.cast(T, (fx * s, fy * s, (cx + 0.5) * s - 0.5, (cy + 0.5) * s - 0.5), w * s, h * s, True)
        img = cv2.resize(hi.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA).round().astype(np.uint8)
    depth = _SCENE.cast(T, K, w, h, False)
    cv2.imwrite(os.path.join(out, "rgb", f"{t_ns}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    cv2.imwrite(os.path.join(out, "depth", f"{t_ns}.png"), np.round(depth * 1000).astype(np.uint16))
    return t_ns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq", help="sim rgbdi sequence (sim_to_rgbdi.py): poses, IMU, calibration")
    ap.add_argument("run", help="the sim run it came from (export/boundary.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blackout", action="append", default=[], help="START:DUR s, featureless frames")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    seq = rgbdi.load(args.seq)
    b = np.load(os.path.join(args.run, "export", "boundary.npz"))
    inner, outer = b["inner_0"], b["outer_0"]
    G, inside = rgbdi.interpolate_poses(seq.gt_t_ns, seq.gt_T, seq.t_ns)
    t_ns = seq.t_ns[inside]
    t0 = t_ns[0]
    blank = np.zeros(len(t_ns), bool)
    for spec in args.blackout:
        a, d = (float(x) for x in spec.split(":"))
        blank |= ((t_ns - t0) * 1e-9 >= a) & ((t_ns - t0) * 1e-9 < a + d)

    os.makedirs(os.path.join(args.out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "depth"), exist_ok=True)
    w, h = int(seq.calib["width"]), int(seq.calib["height"])
    jobs = [(int(t), T, tuple(seq.K), w, h, args.out, bool(bl)) for t, T, bl in zip(t_ns, G[inside], blank)]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(inner, outer)) as ex:
        for k, _ in enumerate(ex.map(_render, jobs, chunksize=4)):
            if k % 200 == 0:
                print(f"{k}/{len(jobs)}", flush=True)

    with open(os.path.join(args.out, "frames.csv"), "w") as f:
        f.write("t_ns,rgb,depth\n")
        for t in t_ns:
            f.write(f"{t},rgb/{t}.png,depth/{t}.png\n")
    for name in ("imu.csv", "groundtruth.csv"):
        shutil.copyfile(os.path.join(args.seq, name), os.path.join(args.out, name))
    calib = dict(seq.calib)
    calib["source"] = f"render_track.py of {os.path.abspath(args.seq)}"
    calib["notes"] = ("rendered: floor + walls from the run's track boundary, procedural texture, exact depth; "
                      "IMU and ground truth from the sim sequence" +
                      (f"; featureless frames at {', '.join(args.blackout)} s" if args.blackout else ""))
    with open(os.path.join(args.out, "calib.yaml"), "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    print(f"{len(t_ns)} frames ({int(blank.sum())} blanked) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
