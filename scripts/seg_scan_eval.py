#!/usr/bin/env python3
"""Score the segmentation-edge scan (control/seg_scan.py) against the sim's real LiDAR on an
exported run (scripts/export_recording.py), frame by frame.

    python scripts/seg_scan_eval.py <export dir> [--camera sim_camera/camera.yaml] [--stride 3]

The export must have camera/ (RGB + depth), segmentation/ and lidar.npz from the same run.
Only beams inside the camera's view are compared. In the stock sim the depth image itself
comes from the LiDAR (sim_depth), so this checks the mask edges and the geometry, not the
depth sensor.

Reports, over the compared beams:
  range error     |seg scan - LiDAR| where both see a wall within --near m
  missed walls    LiDAR return within --near m, seg scan free or unknown
  phantom walls   seg scan return, LiDAR more than --tol m farther (or free)
  gap target      LidarGapFollower's target angle on the seg scan vs on the LiDAR scan
                  (same FOV limit for both), and on the full 270 deg LiDAR scan
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    import cv2
    import yaml

    from control.depth_ftg import Intrinsics
    from control.lidar_gap import LidarGapConfig, LidarGapFollower
    from control.seg_scan import SegScanConfig, seg_edge_scan
    from control.types import VehicleParams, resolve_path
    from workers.lidar_depth import CameraModel

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="export directory")
    ap.add_argument("--camera", default="sim_camera/camera.yaml", help="sim camera model (mount + intrinsics)")
    ap.add_argument("--scale", type=float, default=2.0, help="sim_depth.scale of the run (default 2)")
    ap.add_argument("--config", default="config_autodrive.yaml", help="for track_control.mapping_drive")
    ap.add_argument("--stride", type=int, default=3, help="every n-th frame (default 3)")
    ap.add_argument("--near", type=float, default=3.0, help="m, walls that matter for driving (default 3)")
    ap.add_argument("--tol", type=float, default=0.5, help="m, phantom-wall tolerance (default 0.5)")
    ap.add_argument("--no-floor-fallback", action="store_true", help="unknown instead of the flat-floor estimate")
    a = ap.parse_args()

    cam = CameraModel.from_yaml(a.camera, scale=a.scale)
    fx, fy, cx, cy = cam.intrinsics()
    K = Intrinsics(fx, fy, cx, cy)
    cfg = SegScanConfig(camera_pitch=np.radians(cam.pitch_deg), camera_height=cam.cam_height,
                        origin_forward=cam.cam_x - cam.lidar_x, max_range=cam.max_range,
                        floor_fallback=not a.no_floor_fallback)
    with open(resolve_path(a.config)) as f:
        drive = dict((yaml.safe_load(f) or {}).get("track_control", {}).get("mapping_drive", {}))
    gap = LidarGapFollower(VehicleParams.load("vehicle_params_roboracer.yaml"), LidarGapConfig(**drive))

    E = a.export
    with open(os.path.join(E, "camera.csv")) as f:
        cams = list(csv.DictReader(f))
    with open(os.path.join(E, "segmentation.csv")) as f:
        segs = list(csv.DictReader(f))
    lid = np.load(os.path.join(E, "lidar.npz"))
    t_seg = np.array([float(s["t_mono"]) for s in segs])

    err, missed, phantom, n_near, n_cmp, d_fov, d_full = [], 0, 0, 0, 0, [], []
    kinds: dict = {}
    for i in range(0, len(cams), a.stride):
        tc = float(cams[i]["t_mono"])
        j = int(np.searchsorted(t_seg, tc))            # the mask computed from this frame
        nxt = float(cams[i + 1]["t_mono"]) if i + 1 < len(cams) else np.inf
        if j >= len(segs) or t_seg[j] >= nxt:
            continue
        k = int(np.searchsorted(lid["t_mono"], tc + 0.005)) - 1    # same sim message
        if k < 0:
            continue
        depth = cv2.imread(os.path.join(E, cams[i]["depth_png_mm"]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        mask = cv2.imread(os.path.join(E, segs[j]["mask_png"]), cv2.IMREAD_GRAYSCALE)
        scan, dbg = seg_edge_scan(mask, depth, K, cfg, return_debug=True)
        for kk, v in dbg["counts"].items():
            kinds[kk] = kinds.get(kk, 0) + v
        true = lid["ranges"][k].astype(np.float64)
        view = ~np.isnan(scan)
        n_cmp += int(view.sum())
        near = view & np.isfinite(true) & (true < a.near)
        n_near += int(near.sum())
        both = near & np.isfinite(scan)
        err.extend(np.abs(scan[both] - true[both]))
        missed += int(np.sum(near & ~np.isfinite(scan)))
        ph = view & np.isfinite(scan) & (scan < a.near) & (~np.isfinite(true) | (true > scan + a.tol))
        phantom += int(ph.sum())
        # same FOV for both: what the camera can see
        lim = np.where(view, true, np.nan)
        d_fov.append(np.degrees(gap.target_angle(scan) - gap.target_angle(lim)))
        d_full.append(np.degrees(gap.target_angle(scan) - gap.target_angle(true)))

    err = np.array(err)
    d_fov, d_full = np.abs(d_fov), np.abs(d_full)
    tot = sum(kinds.values())
    print(f"frames: {len(d_fov)}  beams in view: {n_cmp}  of which a LiDAR wall within {a.near} m: {n_near}")
    print("columns: " + ", ".join(f"{k} {100 * v / tot:.1f} %" for k, v in kinds.items()))
    print(f"range error: median {100 * np.median(err):.1f} cm, p90 {100 * np.percentile(err, 90):.1f} cm, "
          f"p99 {100 * np.percentile(err, 99):.1f} cm  (n={len(err)})")
    print(f"missed walls: {100 * missed / max(n_near, 1):.2f} % of near LiDAR returns")
    print(f"phantom walls: {100 * phantom / max(n_cmp, 1):.2f} % of beams in view")
    print(f"gap target vs LiDAR, same FOV: median {np.median(d_fov):.1f} deg, p90 {np.percentile(d_fov, 90):.1f} deg; "
          f"vs full 270 deg LiDAR: median {np.median(d_full):.1f} deg, p90 {np.percentile(d_full, 90):.1f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
