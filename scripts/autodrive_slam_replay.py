#!/usr/bin/env python3
"""Replay an AutoDRIVE sim recording through ORB-SLAM3 RGB-D with the LiDAR depth
adapter (workers/lidar_depth.py) and score it against the sim's ground-truth pose.

    python scripts/autodrive_slam_replay.py <recording dir> [--camera sim_camera/camera.yaml]
           [--settings sim_camera/orbslam_settings.yaml] [--cpu] [--out dir]

The frames go in as SLAMWorker would get them live from AutoDRIVEWorker with sim_depth
on: upscaled, the car body painted flat, depth from that message's LiDAR scan. The SLAM
camera trajectory is rigidly aligned (Umeyama, no scale: RGB-D is metric) to the true
camera trajectory (sim pose + camera mount); reported: tracked share, ATE RMS / max,
the Sim(3) scale as a check, tracking ms per frame. --cpu sets ORB_SLAM3_CUDA=0.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """R, t, s minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / xs.var(0).sum()) if with_scale else 1.0
    return R, mu_d - s * R @ mu_s, s


def frames_from_recording(rec: str, cam):
    """[(ts, rgb_in, depth, true camera position)] as the live adapter would produce them."""
    import cv2

    from read_recording import load_buffer
    from workers.lidar_depth import LidarDepth, mask_rgb

    fn = LidarDepth(cam)
    frames = load_buffer(rec, "frame")
    scans = load_buffer(rec, "lidar")
    poses = load_buffer(rec, "pose")
    st = np.array([s.produced_ts for s in scans])
    pt = np.array([p.produced_ts for p in poses])
    w, h = cam.size
    out = []
    for f in frames:
        k = int(np.argmin(np.abs(st - f.produced_ts)))
        j = int(np.argmin(np.abs(pt - f.produced_ts)))
        if abs(st[k] - f.produced_ts) > 0.02 or abs(pt[j] - f.produced_ts) > 0.02:
            continue
        rgb = cv2.resize(f.value[0], (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = mask_rgb(rgb, fn.mask)
        depth = fn(scans[k].value)
        Twv = poses[j].value[0].astype(np.float64)
        cam_world = Twv[:3, :3] @ np.array([cam.cam_x, 0.0, 0.15]) + Twv[:3, 3]
        out.append((f.produced_ts, np.ascontiguousarray(rgb), depth.astype(np.float32), cam_world))
    return out


def replay(rec: str, camera: str, settings: str, scale: float, cpu: bool, max_frames: int = 0) -> dict:
    if cpu:
        os.environ["ORB_SLAM3_CUDA"] = "0"
    import importlib
    importlib.import_module("ssl")       # see SLAMWorker: libcrypto load order
    import orbslam3_python as slam_mod

    from control.types import resolve_path
    from workers.lidar_depth import CameraModel

    cam = CameraModel.from_yaml(camera, scale=scale)
    data = frames_from_recording(rec, cam)
    if max_frames:
        data = data[:max_frames]
    slam = slam_mod.System(resolve_path("ORB_SLAM3/Vocabulary/ORBvoc.txt"), resolve_path(settings), slam_mod.RGBD)
    est, gt, ms, states, ts_ok, keys = [], [], [], [], [], []
    try:
        for i, (ts, rgb, depth, cw) in enumerate(data):
            if i % 100 == 0:
                print(f"frame {i}/{len(data)}: tracked {len(est)}, "
                      f"{np.mean(ms[-100:]) if ms else 0:.1f} ms/frame", file=sys.stderr, flush=True)
            t0 = time.perf_counter()
            Tcw = np.asarray(slam.track_rgbd(rgb, depth, ts), dtype=np.float64)
            ms.append(1e3 * (time.perf_counter() - t0))
            state = slam.get_tracking_state()
            states.append(int(state))
            if state == 2 and np.isfinite(Tcw).all() and abs(np.linalg.det(Tcw[:3, :3]) - 1) < 1e-3:
                R, t = Tcw[:3, :3], Tcw[:3, 3]
                mid, kf0, _ = slam.map_info()
                est.append(-R.T @ t)
                gt.append(cw)
                ts_ok.append(ts)
                keys.append((int(mid), int(kf0)))     # a new key = a new world frame
    finally:
        slam.shutdown()
    est, gt = np.array(est), np.array(gt)
    res = {"recording": rec, "frames": len(data), "tracked": len(est),
           "tracked_pct": 100.0 * len(est) / max(len(data), 1),
           "track_ms_mean": float(np.mean(ms)) if ms else math.nan,
           "track_ms_p95": float(np.percentile(ms, 95)) if ms else math.nan,
           "cuda": not cpu, "camera": camera, "settings": settings, "scale": scale,
           "state_counts": {str(k): states.count(k) for k in sorted(set(states))}}
    # each ORB-SLAM3 map has its own world frame: align and score each one separately
    maps = []
    for key in dict.fromkeys(keys):
        sel = np.array([k == key for k in keys])
        if sel.sum() < 10:
            maps.append({"map": list(key), "tracked": int(sel.sum())})
            continue
        e, g = est[sel], gt[sel]
        R, t, _ = umeyama(e, g, with_scale=False)
        err = np.linalg.norm((e @ R.T + t) - g, axis=1)
        _, _, s = umeyama(e, g, with_scale=True)
        path = float(np.sum(np.linalg.norm(np.diff(g, axis=0), axis=1)))
        maps.append({"map": list(key), "tracked": int(sel.sum()), "path_m": path,
                     "ate_rms_m": float(np.sqrt(np.mean(err ** 2))), "ate_max_m": float(err.max()),
                     "sim3_scale": s, "_e": (e @ R.T + t).tolist(), "_g": g.tolist()})
    res["maps"] = [{k: v for k, v in m.items() if not k.startswith("_")} for m in maps]
    res["n_maps"] = len(maps)
    scored = [m for m in maps if "ate_rms_m" in m]
    if scored:
        main_ = max(scored, key=lambda m: m["path_m"])
        res.update(ate_rms_m=main_["ate_rms_m"], ate_max_m=main_["ate_max_m"], sim3_scale=main_["sim3_scale"],
                   main_map_path_m=main_["path_m"], main_map_tracked=main_["tracked"])
        res["_traj"] = {"est_aligned": main_["_e"], "gt": main_["_g"]}
    return res


def plot(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr = res.get("_traj")
    if not tr:
        return
    e, g = np.array(tr["est_aligned"]), np.array(tr["gt"])
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(g[:, 0], g[:, 1], color="#52514e", lw=1.2, label="true camera path (sim pose)")
    ax.plot(e[:, 0], e[:, 1], color="#2a78d6", lw=1.2, label="ORB-SLAM3 RGB-D, LiDAR depth (aligned)")
    ax.set_aspect("equal")
    ax.set_title(f"ATE RMS {100 * res['ate_rms_m']:.1f} cm, tracked {res['tracked_pct']:.0f}% "
                 f"({'CUDA' if res['cuda'] else 'CPU'}, {res['track_ms_mean']:.1f} ms)", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording")
    ap.add_argument("--camera", default="sim_camera/camera.yaml")
    ap.add_argument("--settings", default="sim_camera/orbslam_settings.yaml")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--cpu", action="store_true", help="ORB_SLAM3_CUDA=0")
    ap.add_argument("--out", help="write replay.json and replay.png here")
    ap.add_argument("--max-frames", type=int, default=0)
    a = ap.parse_args()
    res = replay(a.recording, a.camera, a.settings, a.scale, a.cpu, a.max_frames)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        tag = "cpu" if a.cpu else "cuda"
        with open(os.path.join(a.out, f"replay_{tag}.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
        plot(res, os.path.join(a.out, f"replay_{tag}.png"))
    print(json.dumps({k: v for k, v in res.items() if k != "_traj"}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
