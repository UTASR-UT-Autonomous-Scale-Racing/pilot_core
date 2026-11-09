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

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from control.depth_ftg import Intrinsics  # noqa: E402
from control.lidar_gap import ANGLE_INCREMENT, ANGLE_MIN  # noqa: E402
from control.seg_scan import SegScanConfig, seg_edge_scan  # noqa: E402
from control.occupancy_map import (OccupancyConfig, occupancy, scan_to_samples, signed_area,  # noqa: E402
                                   walls_from_grid)
from control.track import TrackModel, _resample_closed, _smooth_closed  # noqa: E402
from read_recording import _decoder, iter_file, list_files  # noqa: E402

def load_poses(rec: str) -> dict:
    with open(os.path.join(rec, "sim_messages.csv")) as f:
        rows = list(csv.DictReader(f))
    p = {k: np.array([float(r[k]) for r in rows]) for k in ("t_rx", "x", "y", "yaw", "collisions")}
    p["yaw"] = np.unwrap(p["yaw"])
    return p


def replay(rec: str, poses: dict, K: Intrinsics, cfg: SegScanConfig, scan_x: float, occ: OccupancyConfig) -> np.ndarray:
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
        if np.any((r.produced_ts > hits - occ.drop_before) & (r.produced_ts < hits + occ.drop_after)):
            continue
        here = np.array([poses["x"][i], poses["y"][i]])
        if last is not None and np.linalg.norm(here - last) < occ.min_move:
            continue
        last = here
        m = dec(r.payload)
        m = np.asarray(m[1] if isinstance(m, tuple) else m)
        s = seg_edge_scan(m, np.zeros(m.shape, np.float32), K, cfg)
        yaw = poses["yaw"][i]
        origin = here + scan_x * np.array([np.cos(yaw), np.sin(yaw)])
        out.append(scan_to_samples(s, ang, origin, yaw, occ))
    return np.vstack(out)


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

    occ_cfg = OccupancyConfig(edge_range=a.edge_range, free_range=a.free_range, ratio=a.ratio, min_occ=a.min_occ,
                              min_free=a.min_free, reach=a.reach, seal=a.seal)
    poses, scans = [], []
    for rec in a.recordings:
        p = load_poses(rec)
        poses.append(p)
        scans.append(replay(rec, p, K, cfg, scan_x, occ_cfg))
    scans = np.vstack(scans)
    path = np.vstack([np.column_stack([p["x"], p["y"]]) for p in poses])

    res = occ_cfg.res
    x0, y0 = path.min(axis=0) - occ_cfg.pad
    w, h = np.ceil((path.max(axis=0) + occ_cfg.pad - [x0, y0]) / res).astype(int)
    free, occ = occupancy(scans, x0, y0, (h, w), occ_cfg)
    cells = np.unique(((path - [x0, y0]) / res).astype(int), axis=0)
    outer, inner, region = walls_from_grid(free, occ, cells, occ_cfg)

    # the lap's direction is the driven path's: the map's files are ordered along it
    t = poses[0]
    turn = signed_area(np.column_stack([t["x"], t["y"]])[::10])
    lines = []
    for b in (outer, inner):
        b = (b + 0.5) * res + [x0, y0]
        b = _resample_closed(_smooth_closed(b, occ_cfg.smooth), occ_cfg.spacing)
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
    ax.imshow(region, origin="lower", cmap="Blues", alpha=0.3, extent=(x0, x0 + w * res, y0, y0 + h * res))
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
