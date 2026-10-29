#!/usr/bin/env python3
"""Score RGB-D-inertial benchmark results against an rgbdi sequence's ground truth.

    python scripts/vio_bench/score.py <seq dir> <result dir> [<result dir> ...] --out <dir>

Per result (run_orbslam3.py / run_cuvslam.py output):
  accuracy  ATE: RMSE / median / max position error after one rigid (SE3, no scale: depth
            fixes it) alignment of the whole run; drift: RPE over 1 s windows,
            translation (m and % of distance) and rotation (deg)
  coverage  tracked frames / all frames; time from the first frame to the first pose
  latency   per-frame tracking call, ms: mean, p50, p95, p99, max (the first frame, which
            builds the map, is left out)
Without a groundtruth.csv (the real track) only coverage and latency are scored; the
accuracy comes from markers.py. Writes summary.md, summary.json and trajectories.png.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402


def umeyama_se3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """4x4 rigid T minimising |T src - dst|^2 (no scale)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mu_d - R @ mu_s
    return T


def rot_angle_deg(R: np.ndarray) -> np.ndarray:
    c = (np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def score(seq: rgbdi.Sequence, res_dir: str, window_s: float = 1.0) -> dict:
    with open(os.path.join(res_dir, "meta.json")) as f:
        meta = json.load(f)
    t_est, T_est = rgbdi.read_poses(os.path.join(res_dir, "poses.csv"))
    with open(os.path.join(res_dir, "frames.csv")) as f:
        fr = list(csv.DictReader(f))
    ms = np.array([float(r["track_ms"]) for r in fr])[1:]
    out = {"name": os.path.basename(os.path.normpath(res_dir)), **meta,
           "frames": len(seq.t_ns), "tracked_frac": len(t_est) / max(len(seq.t_ns), 1),
           "first_pose_s": float((t_est[0] - seq.t_ns[0]) * 1e-9) if len(t_est) else None,
           "latency_ms": {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)),
                          "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)),
                          "max": float(ms.max())}}
    if "imu_initialized" in fr[0]:
        init = [int(r["t_ns"]) for r in fr if r["imu_initialized"] == "1"]
        out["imu_init_s"] = float((init[0] - seq.t_ns[0]) * 1e-9) if init else None
    if "map_id" in fr[0]:
        out["maps"] = len({r["map_id"] for r in fr if int(r["state"]) == 2})
    if len(t_est) < 3 or len(seq.gt_t_ns) == 0:
        return out

    G, inside = rgbdi.interpolate_poses(seq.gt_t_ns, seq.gt_T, t_est)
    t_est, T_est, G = t_est[inside], T_est[inside], G[inside]
    A = umeyama_se3(T_est[:, :3, 3], G[:, :3, 3])
    P = A @ T_est
    err = np.linalg.norm(P[:, :3, 3] - G[:, :3, 3], axis=1)
    out["ate_m"] = {"rmse": float(np.sqrt(np.mean(err ** 2))), "median": float(np.median(err)),
                    "max": float(err.max())}
    # RPE over window_s
    ts = t_est * 1e-9
    j = np.searchsorted(ts, ts + window_s)
    ok = j < len(ts)
    i, j = np.nonzero(ok)[0], j[ok]
    if len(i):
        dG = np.linalg.inv(G[i]) @ G[j]
        dP = np.linalg.inv(P[i]) @ P[j]
        E = np.linalg.inv(dG) @ dP
        et = np.linalg.norm(E[:, :3, 3], axis=1)
        dist = np.linalg.norm(dG[:, :3, 3], axis=1)
        moving = dist > 0.05
        out["rpe"] = {"window_s": window_s, "trans_rmse_m": float(np.sqrt(np.mean(et ** 2))),
                      "trans_pct": float(100 * np.median(et[moving] / dist[moving])) if moving.any() else None,
                      "rot_rmse_deg": float(np.sqrt(np.mean(rot_angle_deg(E[:, :3, :3]) ** 2)))}
    out["_aligned"] = (t_est, P)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    seq = rgbdi.load(args.seq)
    scores = [score(seq, r) for r in args.results]
    os.makedirs(args.out, exist_ok=True)

    def f(v, fmt="{:.3f}"):
        return "–" if v is None else fmt.format(v)

    lines = [f"# RGB-D-inertial benchmark: {os.path.basename(os.path.normpath(args.seq))}", "",
             f"{len(seq.t_ns)} frames, {(seq.t_ns[-1] - seq.t_ns[0]) * 1e-9:.1f} s; "
             f"{seq.calib.get('source', '')}", "",
             "| run | mode | GPU | tracked | first pose (s) | IMU init (s) | ATE RMSE (m) | ATE max (m) | "
             "drift 1 s (m / %) | rot drift 1 s (deg) | latency mean / p50 / p95 / p99 / max (ms) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in scores:
        L, ate, rpe = s["latency_ms"], s.get("ate_m", {}), s.get("rpe", {})
        lines.append(
            f"| {s['name']} | {s['mode']} | {s['gpu']} | {100 * s['tracked_frac']:.1f} % | {f(s['first_pose_s'], '{:.2f}')} | "
            f"{f(s.get('imu_init_s'), '{:.1f}')} | {f(ate.get('rmse'))} | {f(ate.get('max'))} | "
            f"{f(rpe.get('trans_rmse_m'))} / {f(rpe.get('trans_pct'), '{:.1f}')} | {f(rpe.get('rot_rmse_deg'), '{:.2f}')} | "
            f"{L['mean']:.1f} / {L['p50']:.1f} / {L['p95']:.1f} / {L['p99']:.1f} / {L['max']:.1f} |")
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump([{k: v for k, v in s.items() if not k.startswith("_")} for s in scores], fh, indent=1)
    print("\n".join(lines))

    if len(seq.gt_T) == 0:
        return 0                     # no ground truth (real track): accuracy is markers.py's
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    g = seq.gt_T[:, :3, 3]
    # the plane the trajectory spans: drop the axis with the least spread (gravity for a car)
    keep = np.sort(np.argsort(g.std(0))[1:])
    ax.plot(g[:, keep[0]], g[:, keep[1]], "k-", lw=2, label="ground truth")
    for s in scores:
        if "_aligned" in s:
            p = s["_aligned"][1][:, :3, 3]
            ax.plot(p[:, keep[0]], p[:, keep[1]], ".", ms=2, label=s["name"])
    ax.set_aspect("equal")
    ax.legend()
    fig.savefig(os.path.join(args.out, "trajectories.png"), dpi=110, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
