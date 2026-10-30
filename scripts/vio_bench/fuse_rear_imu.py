#!/usr/bin/env python3
"""Vehicle state from SLAM poses + the rear IMU, scored causally against ground truth.

    python scripts/vio_bench/fuse_rear_imu.py <seq dir> <orbslam3 result dir> --out <dir>

The sequence needs rear_imu.csv / rear_truth.csv (make_rear_imu.py); the result is a
run_orbslam3.py output. Replays what the car would see: each SLAM pose arrives at its
frame time + its tracking time (frames.csv) + --pipeline-ms, the rear IMU at its host
stamps. Three estimators get the same stream and are read at every rear-IMU tick:

  imu_ekf    control.ImuEKF: rear IMU driven, SLAM poses applied at their capture time
  pose_ekf   control.PoseEKF: SLAM poses only (the car's StateWorker today)
  slam_raw   the latest SLAM pose that has arrived (no estimator)

SLAM poses are turned into the planar pose of the IMU point (vehicle frame): camera ->
vehicle with the sequence's T_vehicle_cam, levelled by a plane fit of each map's
trajectory (the car does it by fitting the floor). Each map is scored after one SE(2)
alignment of its SLAM poses to the truth. Before the run the rear IMU's time offset is
estimated by cross-correlating its yaw rate with the camera IMU's (--no-time-sync skips
it). Metrics: position / yaw error at every tick, in SLAM gaps (latest pose older than
--gap-s) separately, speed, yaw-rate and sideslip error, and availability.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402
from control.estimator import ImuEKF, ImuEstimatorConfig, PoseEKF  # noqa: E402
from control.types import wrap_angle  # noqa: E402


def planar(T_wv: np.ndarray) -> np.ndarray:
    """(N, 4, 4) levelled poses -> (N, 3) x, y, yaw of the vehicle x axis."""
    return np.column_stack([T_wv[:, 0, 3], T_wv[:, 1, 3], np.arctan2(T_wv[:, 1, 0], T_wv[:, 0, 0])])


def level(T_wv: np.ndarray) -> np.ndarray:
    """Rotation taking the plane of a flat-track trajectory to z = 0, z up along the vehicles' z."""
    p = T_wv[:, :3, 3]
    if len(p) < 10 or np.ptp(p, axis=0).max() < 0.5:
        n = T_wv[:, :3, 2].mean(axis=0)
    else:
        n = np.linalg.svd(p - p.mean(axis=0))[2][2]
    n = n / np.linalg.norm(n)
    if n @ T_wv[:, :3, 2].mean(axis=0) < 0:
        n = -n
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(n, z), float(n @ z)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def se2_align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 rigid 2-D transform minimising |A src - dst| over x, y."""
    ms, md = src[:, :2].mean(0), dst[:, :2].mean(0)
    U, _, Vt = np.linalg.svd((dst[:, :2] - md).T @ (src[:, :2] - ms))
    S = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    A = np.eye(3)
    A[:2, :2], A[:2, 2] = R, md - R @ ms
    return A


def apply_se2(A: np.ndarray, p: np.ndarray) -> np.ndarray:
    xy = p[:, :2] @ A[:2, :2].T + A[:2, 2]
    return np.column_stack([xy, wrap_angle(p[:, 2] + np.arctan2(A[1, 0], A[0, 0]))])


def time_offset(cam_imu: np.ndarray, rear: np.ndarray, max_s: float = 0.08) -> float:
    """s to subtract from the rear IMU's stamps: yaw-rate cross-correlation against the
    camera IMU (device-timestamped on the frames' clock)."""
    t0, t1 = max(cam_imu[0, 0], rear[0, 0]), min(cam_imu[-1, 0], rear[-1, 0])
    tg = np.arange(t0 + max_s, t1 - max_s, 0.001)
    a = np.interp(tg, cam_imu[:, 0], cam_imu[:, 6])
    a = a - a.mean()
    lags = np.arange(-max_s, max_s, 0.0005)
    score = []
    for lag in lags:
        b = np.interp(tg + lag, rear[:, 0], rear[:, 6])
        score.append(float(a @ (b - b.mean())))
    return float(lags[int(np.argmax(score))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq")
    ap.add_argument("result")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pipeline-ms", type=float, default=5.0, help="frame transfer + publishing, on top of tracking")
    ap.add_argument("--gap-s", type=float, default=0.1, help="latest pose older than this = in a SLAM gap")
    ap.add_argument("--no-time-sync", action="store_true", help="use the rear IMU's host stamps as they are")
    ap.add_argument("--vy-std", type=float, default=0.1, help="ImuEKF no-slip pseudo-measurement (m/s), <= 0: off")
    ap.add_argument("--gate-chi2", type=float, default=16.3, help="ImuEKF pose gate, <= 0: off")
    ap.add_argument("--gate-reset-s", type=float, default=2.0, help="ImuEKF re-anchors to SLAM after rejecting this long")
    args = ap.parse_args()

    seq = rgbdi.load(args.seq)
    T_vc = seq.T_imu_cam
    rear = np.loadtxt(os.path.join(args.seq, "rear_imu.csv"), delimiter=",", skiprows=1)
    rear[:, 0] *= 1e-9
    truth = np.loadtxt(os.path.join(args.seq, "rear_truth.csv"), delimiter=",", skiprows=1)
    truth[:, 0] *= 1e-9
    cam_imu = seq.imu.copy()
    cam_imu[:, 0] *= 1e-9
    offset = 0.0 if args.no_time_sync else time_offset(cam_imu, rear)
    rear_t = rear[:, 0] - offset

    # SLAM: vehicle-frame planar poses per map, with arrival times
    with open(os.path.join(args.result, "frames.csv")) as f:
        frames = {int(r["t_ns"]): r for r in csv.DictReader(f)}
    t_ns, T_wc = rgbdi.read_poses(os.path.join(args.result, "poses.csv"))
    T_wv = T_wc @ np.linalg.inv(T_vc)
    maps = np.array([int(frames[int(t)].get("map_id", 0)) for t in t_ns])
    arrive = np.array([int(t) * 1e-9 + float(frames[int(t)]["track_ms"]) * 1e-3 + args.pipeline_ms * 1e-3 for t in t_ns])
    meas = np.zeros((len(t_ns), 3))
    G, inside = rgbdi.interpolate_poses(seq.gt_t_ns, seq.gt_T, t_ns)
    gt_pl = planar(G @ np.linalg.inv(T_vc))
    align = {}
    for m in np.unique(maps):
        k = maps == m
        L = np.eye(4)
        L[:3, :3] = level(T_wv[k])
        meas[k] = planar(L @ T_wv[k])
        ok = k & inside
        align[m] = se2_align(meas[ok], gt_pl[ok]) if ok.sum() >= 3 else None

    # replay
    cfg = ImuEstimatorConfig(vy_std=args.vy_std if args.vy_std > 0 else None,
                             gate_chi2=args.gate_chi2 if args.gate_chi2 > 0 else None, gate_reset_s=args.gate_reset_s)
    names = ("imu_ekf", "pose_ekf", "slam_raw")
    est = {"imu_ekf": ImuEKF(cfg), "pose_ekf": PoseEKF()}
    events = sorted([(rear_t[i], 0, i) for i in range(len(rear_t))] + [(arrive[j], 1, j) for j in range(len(t_ns))])
    out = {n: np.full((len(rear_t), 6), np.nan) for n in names}      # x, y, yaw, v, yaw rate (aligned), vy
    tick_map = np.full(len(rear_t), -1)
    last_pose_t = np.full(len(rear_t), np.nan)
    cur_map, raw, raw_t = None, None, None
    for t_ev, kind, i in events:
        if kind == 1:
            m = maps[i]
            if m != cur_map:                       # new world frame: start over (the car resets too)
                cur_map = m
                for e in est.values():
                    e.reset()
            tm = int(t_ns[i]) * 1e-9
            for e in est.values():
                e.update_pose(tm, *meas[i])
            raw, raw_t = meas[i], tm
            continue
        t = rear_t[i]
        est["imu_ekf"].update_imu(t, rear[i, 1], rear[i, 2], rear[i, 6])
        if cur_map is None or align.get(cur_map) is None:
            continue
        A = align[cur_map]
        tick_map[i], last_pose_t[i] = cur_map, raw_t
        for n in ("imu_ekf", "pose_ekf"):
            st = est[n].predict_to(t)
            if st is not None:
                vy = est[n].x[4] if n == "imu_ekf" else np.nan
                out[n][i] = np.r_[apply_se2(A, np.array([[st.x, st.y, st.yaw]]))[0], st.v, st.yaw_rate, vy]
        if raw is not None:
            out["slam_raw"][i, :3] = apply_se2(A, raw[None])[0]

    # truth at the ticks (the rear IMU's own stamps are the ticks)
    tr = np.column_stack([np.interp(rear_t, truth[:, 0], truth[:, c]) for c in range(1, 7)])
    tr[:, 2] = np.interp(rear_t, truth[:, 0], np.unwrap(truth[:, 3]))
    speed = np.hypot(tr[:, 3], tr[:, 4])
    in_gap = (rear_t - last_pose_t) > args.gap_s
    scored = tick_map >= 0
    report = {"sequence": os.path.abspath(args.seq), "slam": os.path.abspath(args.result),
              "rear_imu_time_offset_ms": 1e3 * offset, "ticks": int(scored.sum()),
              "gap_ticks": int((scored & in_gap).sum()), "estimators": {}}

    def stats(e):
        e = e[np.isfinite(e)]
        return None if not len(e) else {"rmse": float(np.sqrt(np.mean(e ** 2))), "p95": float(np.percentile(np.abs(e), 95)),
                                        "max": float(np.abs(e).max())}

    for n in names:
        o = out[n]
        valid = scored & np.isfinite(o[:, 0])
        pos = np.hypot(o[:, 0] - tr[:, 0], o[:, 1] - tr[:, 1])
        yaw = np.degrees(wrap_angle(o[:, 2] - tr[:, 2]))
        r = {"available": float(valid.sum() / max(scored.sum(), 1)),
             "pos_m": stats(pos[valid]), "pos_in_gaps_m": stats(pos[valid & in_gap]),
             "yaw_deg": stats(yaw[valid])}
        if n != "slam_raw":
            r["speed_mps"] = stats((np.abs(o[:, 3]) - speed)[valid])
            r["yaw_rate_dps"] = stats(np.degrees(o[:, 4] - tr[:, 5])[valid])
        if n == "imu_ekf":
            r["sideslip_vy_mps"] = stats((o[:, 5] - tr[:, 4])[valid])
            r["truth_vy_rms_mps"] = float(np.sqrt(np.mean(tr[valid, 4] ** 2)))
        report["estimators"][n] = r
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ticks.csv"), "w") as f:      # per tick, for plots
        f.write("t,map,in_gap,truth_x,truth_y,truth_yaw,truth_speed,truth_yaw_rate,"
                + ",".join(f"{n}_x,{n}_y,{n}_yaw,{n}_v,{n}_r" for n in names) + "\n")
        for i in np.nonzero(scored)[0]:
            f.write(f"{rear_t[i]:.4f},{tick_map[i]},{int(in_gap[i])},{tr[i, 0]:.4f},{tr[i, 1]:.4f},{tr[i, 2]:.4f},{speed[i]:.3f},{tr[i, 5]:.4f},"
                    + ",".join(f"{out[n][i, 0]:.4f},{out[n][i, 1]:.4f},{out[n][i, 2]:.4f},{out[n][i, 3]:.3f},{out[n][i, 4]:.4f}"
                               for n in names) + "\n")
    with open(os.path.join(args.out, "fusion.json"), "w") as f:
        json.dump(report, f, indent=1)

    def g(d, *k):
        for x in k:
            if d is None:
                return "–"
            d = d.get(x) if isinstance(d, dict) else None
        return "–" if d is None else f"{d:.3f}"

    lines = [f"rear IMU time offset {1e3 * offset:.1f} ms; {report['ticks']} ticks, {report['gap_ticks']} in SLAM gaps",
             "| estimator | available | pos RMSE / p95 / max (m) | pos in gaps RMSE / max (m) | yaw RMSE (deg) | speed RMSE (m/s) | yaw rate RMSE (deg/s) |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        r = report["estimators"][n]
        lines.append(f"| {n} | {100 * r['available']:.1f} % | {g(r, 'pos_m', 'rmse')} / {g(r, 'pos_m', 'p95')} / {g(r, 'pos_m', 'max')} | "
                     f"{g(r, 'pos_in_gaps_m', 'rmse')} / {g(r, 'pos_in_gaps_m', 'max')} | {g(r, 'yaw_deg', 'rmse')} | "
                     f"{g(r, 'speed_mps', 'rmse')} | {g(r, 'yaw_rate_dps', 'rmse')} |")
    r = report["estimators"]["imu_ekf"]
    lines.append(f"imu_ekf sideslip vy RMSE {g(r, 'sideslip_vy_mps', 'rmse')} m/s (truth vy RMS {r['truth_vy_rms_mps']:.3f} m/s)")
    with open(os.path.join(args.out, "fusion.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
