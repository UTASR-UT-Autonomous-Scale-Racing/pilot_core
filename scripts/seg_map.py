#!/usr/bin/env python3
"""Track map from recorded segmentation masks, no LiDAR: for a mapping lap whose wall hits or
long-range edges spoil MappingWorker's accumulated cloud. Replays each recording's masks
through control/seg_scan.py onto its sim poses and builds an occupancy grid: a ray frees the
floor it crosses and an edge only counts near the car, so an edge placed short from afar
(it gets worse with range, most in dark corners) is cleared by the floor seen up close. The
drivable region is the floor seen free that is connected to the driven path; its outer
contour and its hole are the walls. Works from sim_messages.csv, so a hit's respawn does not matter.

    python scripts/seg_map.py runs/x/map/rec/* --out maps/autodrive_compete_seg --config config_autodrive_compete.yaml
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from control.depth_ftg import Intrinsics  # noqa: E402
from control.lidar_gap import ANGLE_INCREMENT, ANGLE_MIN  # noqa: E402
from control.seg_scan import SegScanConfig, seg_edge_scan  # noqa: E402
from control.track import TrackModel, _resample_closed, _smooth_closed  # noqa: E402
from read_recording import _decoder, iter_file, list_files  # noqa: E402

RES = 0.05                  # m, grid cell
BEAM_STEP = 6               # every n-th beam
PAD = 3.0                   # m of grid around the driven path
HIT_BEFORE, HIT_AFTER = 1.5, 2.5    # s around a wall hit whose masks are dropped
MIN_MOVE = 0.03             # m driven between masks used
CAR_RADIUS = 3              # cells around the driven path that can't be wall
CLOSE = 10                  # cells; notches in the outer wall narrower than twice this are filled
GROW = 8                    # cells the region grows to reach the walls
TRIM = 6                    # cells; narrower necks than twice this are cut


def load_poses(rec: str) -> dict:
    with open(os.path.join(rec, "sim_messages.csv")) as f:
        rows = list(csv.DictReader(f))
    p = {k: np.array([float(r[k]) for r in rows]) for k in ("t_rx", "x", "y", "yaw", "collisions")}
    p["yaw"] = np.unwrap(p["yaw"])
    return p


def replay(rec: str, poses: dict, K: Intrinsics, cfg: SegScanConfig, scan_x: float, max_range: float) -> np.ndarray:
    """(N, 5) [edge x, edge y, range, origin x, origin y], world frame."""
    ang = ANGLE_MIN + np.arange(1080) * ANGLE_INCREMENT
    t = poses["t_rx"]
    rf = list_files(rec, "segmentation")[0]
    dec = _decoder(rf)
    hits = t[np.flatnonzero(np.diff(poses["collisions"]) > 0)]
    out, last = [], None
    for r in iter_file(rf, decode=False):
        i = min(int(np.searchsorted(t, r.produced_ts)), len(t) - 1)
        if abs(t[i] - r.produced_ts) > 0.1:
            continue
        # a car against a wall, or standing, would repeat one bad view hundreds of times
        if np.any((r.produced_ts > hits - HIT_BEFORE) & (r.produced_ts < hits + HIT_AFTER)):
            continue
        here = np.array([poses["x"][i], poses["y"][i]])
        if last is not None and np.linalg.norm(here - last) < MIN_MOVE:
            continue
        last = here
        m = dec(r.payload)
        m = np.asarray(m[1] if isinstance(m, tuple) else m)
        s = seg_edge_scan(m, np.zeros(m.shape, np.float32), K, cfg)
        ok = np.isfinite(s) & (s > 0.05) & (s < max_range)
        ok[np.arange(len(s)) % BEAM_STEP != 0] = False
        c, sn = np.cos(poses["yaw"][i]), np.sin(poses["yaw"][i])
        ox, oy = poses["x"][i] + scan_x * c, poses["y"][i] + scan_x * sn
        fx, fy = s[ok] * np.cos(ang[ok]), s[ok] * np.sin(ang[ok])
        out.append(np.column_stack([ox + fx * c - fy * sn, oy + fx * sn + fy * c, s[ok],
                                    np.full(ok.sum(), ox), np.full(ok.sum(), oy)]))
    return np.vstack(out)


def occupancy(scans: np.ndarray, x0: float, y0: float, shape: tuple, edge_range: float, free_range: float):
    free, occ = np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def add(grid, x, y):
        np.add.at(grid, (((y - y0) / RES).astype(int), ((x - x0) / RES).astype(int)), 1)

    ex, ey, r, ox, oy = scans.T
    for k in np.linspace(0.2, 0.92, 10):        # along the ray, stopping short of the edge
        m = r * k < free_range
        add(free, ox[m] + (ex[m] - ox[m]) * k, oy[m] + (ey[m] - oy[m]) * k)
    m = r < edge_range
    add(occ, ex[m], ey[m])
    return cv2.GaussianBlur(free, (0, 0), 1.5), cv2.GaussianBlur(occ, (0, 0), 1.0)


def signed_area(p: np.ndarray) -> float:
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def walls_from_grid(free, occ, path_cells: np.ndarray, ratio: float, min_occ: float, seal: int, min_free: float, reach: float):
    """(outer, inner) contours in cells (x, y) and the drivable region: floor seen free, not
    behind a wall, connected to the driven path. Rays stop at walls, so a gap in a wall
    doesn't let the region out."""
    wall = (occ / (occ + free + 1e-6) > ratio) & (occ > min_occ)
    # no wall stands where the car drove: a lighting edge across the floor looks like one, and
    # as the camera only looks ahead nothing else clears it
    driven = np.ones(free.shape, np.uint8)
    driven[path_cells[:, 1], path_cells[:, 0]] = 0
    distance = cv2.distanceTransform(driven, cv2.DIST_L2, 5)
    wall &= distance > CAR_RADIUS
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * seal + 1, 2 * seal + 1))
    seen = ((cv2.dilate(wall.astype(np.uint8), disc) == 0) & (free > min_free)).astype(np.uint8)
    n, labels = cv2.connectedComponents(seen)
    on_path = labels[path_cells[:, 1], path_cells[:, 0]]
    region = (labels == np.bincount(on_path[on_path > 0], minlength=n).argmax()).astype(np.uint8)
    # out to the walls (rays stop short of them), no further than reach from the driven path: a
    # side opening in a wall is floor too. Then cut off what got out through a wall's gap
    solid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(GROW):
        region = cv2.dilate(region, solid) & (~wall).astype(np.uint8)
    region &= (distance < reach).astype(np.uint8)
    trim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TRIM + 1, 2 * TRIM + 1))
    n, labels = cv2.connectedComponents(cv2.morphologyEx(region, cv2.MORPH_OPEN, trim))
    on_path = labels[path_cells[:, 1], path_cells[:, 0]]
    region = (labels == np.bincount(on_path[on_path > 0], minlength=n).argmax()).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    by_area = sorted(contours, key=lambda c: -abs(cv2.contourArea(c)))
    # the outer wall has no thin spikes into the track, the inner one has (the hairpin tips):
    # only the outer contour is closed
    outside = cv2.drawContours(np.zeros_like(region), by_area[:1], -1, 1, cv2.FILLED)
    fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CLOSE + 1, 2 * CLOSE + 1))
    outside, _ = cv2.findContours(cv2.morphologyEx(outside, cv2.MORPH_CLOSE, fill),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(outside, key=lambda c: abs(cv2.contourArea(c)))
    return outer[:, 0].astype(np.float64), by_area[1][:, 0].astype(np.float64), region


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", help="recording directories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config_autodrive_compete.yaml")
    ap.add_argument("--edge-range", type=float, default=2.5, help="m, edges further than this are ignored")
    ap.add_argument("--free-range", type=float, default=1.6, help="m of each ray that frees the floor")
    ap.add_argument("--ratio", type=float, default=0.35, help="occupied / (occupied + free) of a wall cell")
    ap.add_argument("--min-occ", type=float, default=0.6, help="smoothed edge count of a wall cell")
    ap.add_argument("--min-free", type=float, default=2.0, help="smoothed ray count of a free cell")
    ap.add_argument("--reach", type=float, default=0.9, help="m from the driven path the track can extend")
    ap.add_argument("--seal", type=int, default=3, help="cells; wall gaps up to twice this are closed")
    ap.add_argument("--reference", help="another map's directory: only reports the distance to its walls")
    a = ap.parse_args()

    with open(a.config) as f:
        c = yaml.safe_load(f)
    s = c["seg_scan"]
    cfg = SegScanConfig(camera_pitch=s["camera_pitch"], camera_height=s["camera_height"],
                        origin_forward=s["origin_forward"])
    settings = c["vision"].get("camera_settings", "sim_camera/orbslam_settings.yaml")
    K = Intrinsics.from_orbslam_settings(settings)
    scan_x = float(c["mapping"]["scan_offset"][0])

    poses, scans = [], []
    for rec in a.recordings:
        p = load_poses(rec)
        poses.append(p)
        scans.append(replay(rec, p, K, cfg, scan_x, a.edge_range + 2.5))
    scans = np.vstack(scans)
    path = np.vstack([np.column_stack([p["x"], p["y"]]) for p in poses])

    x0, y0 = path.min(axis=0) - PAD
    w, h = np.ceil((path.max(axis=0) + PAD - [x0, y0]) / RES).astype(int)
    keep = (scans[:, 0] > x0) & (scans[:, 1] > y0) & (scans[:, 0] < x0 + w * RES) & (scans[:, 1] < y0 + h * RES)
    free, occ = occupancy(scans[keep], x0, y0, (h, w), a.edge_range, a.free_range)
    cells = np.unique(((path - [x0, y0]) / RES).astype(int), axis=0)
    outer, inner, region = walls_from_grid(free, occ, cells, a.ratio, a.min_occ, seal=a.seal, min_free=a.min_free,
                                           reach=a.reach / RES)

    # the lap's direction is the driven path's: the map's files are ordered along it
    t = poses[0]
    turn = signed_area(np.column_stack([t["x"], t["y"]])[::10])
    lines = []
    for b in (outer, inner):
        b = (b + 0.5) * RES + [x0, y0]
        b = _resample_closed(_smooth_closed(b, 11), 0.1)
        lines.append(b if signed_area(b) * turn > 0 else b[::-1])
    outer, inner = lines
    track = TrackModel(outer, inner)
    print(f"{len(a.recordings)} recording(s), {len(scans)} edge samples, outer {len(outer)} pts, inner {len(inner)} pts")
    print(f"centre-line {track.length:.1f} m, half-width min {track.half_width.min():.2f} "
          f"median {np.median(track.half_width):.2f} m")
    os.makedirs(a.out, exist_ok=True)
    np.save(os.path.join(a.out, "outer_line.npy"), outer.astype(np.float32))
    np.save(os.path.join(a.out, "inner_line.npy"), inner.astype(np.float32))

    ref = None
    if a.reference:
        from scipy.spatial import cKDTree

        ref = [np.load(os.path.join(a.reference, n))[:, :2] for n in ("outer_line.npy", "inner_line.npy")]
        for name, line in (("outer", outer), ("inner", inner)):
            d = cKDTree(np.vstack(ref)).query(line)[0]
            print(f"{name} to the reference walls: median {np.median(d):.2f} p90 {np.percentile(d, 90):.2f} "
                  f"max {d.max():.2f} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 12))
    ax.imshow(region, origin="lower", cmap="Blues", alpha=0.3, extent=(x0, x0 + w * RES, y0, y0 + h * RES))
    for r in ref or []:
        ax.plot(r[:, 0], r[:, 1], "r-", lw=0.5)
    for b in (outer, inner):
        ax.plot(*np.vstack([b, b[:1]]).T, "k-", lw=1)
    ax.plot(*track.points.T, "g--", lw=0.6)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(a.out, "map.png"), dpi=80, bbox_inches="tight")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
