#!/usr/bin/env python3
"""AutoDRIVE sim run -> rgbdi sequence (rgbdi.py), the sim half of the RGB-D-inertial
benchmark (README.md here).

    python scripts/export_recording.py <run dir>                 # -> <run dir>/export
    python scripts/vio_bench/sim_to_rgbdi.py <run dir> --out <seq dir>

The run must have been recorded with sim_depth (RGB-D: the camera upscaled, depth from
the LiDAR) and frame.sample_k: 1. The sim has no IMU worth the name (its telemetry
arrives at the frame rate), so the IMU is synthesised from the ground-truth vehicle pose
on the sim's own clock: resampled at --imu-rate, low-passed at --cutoff Hz with a
zero-phase filter (the sim's pose and clock jitter would otherwise differentiate into
metres per second squared of noise), differentiated, plus white noise and a random-walk
bias at the noise densities written to calib.yaml. Ground truth is the same smoothed
trajectory. The sequence starts after the last pose jump (the bench's reset to the line).
That IMU is cleaner and smoother than a real one; this sequence is for exercising the
benchmark, not for judging the IMU fusion.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np
import yaml
from scipy.signal import butter, filtfilt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402
from workers.lidar_depth import CameraModel  # noqa: E402

G = 9.81
VEHICLE_Z = 0.0592     # m, the sim's vehicle frame (rear axle) above the floor (V1 Position z)
# consumer-grade MEMS (the cuVSLAM multisensor example's values; BMI270 / BNO086 class)
IMU_NOISE = {"noise_gyro": 1.6e-4, "gyro_walk": 2.2e-5, "noise_acc": 2.8e-3, "acc_walk": 8.6e-4}


def sim_log(sim_messages: str) -> tuple:
    """The sim's new frames from sim_messages.csv: t_rx, continuous sim time, x, y, yaw.
    V1 Lap Time restarts at every lap and reset, so a backwards step continues at the
    median frame period. Positions and that clock agree (step / dt matches the reported
    speed to a few %), unlike the receive times."""
    rows = []
    with open(sim_messages) as f:
        for r in csv.DictReader(f):
            if r["new_frame"] == "1":
                rows.append([float(r[k]) for k in ("t_rx", "sim_time", "x", "y", "yaw")])
    a = np.array(rows)
    d = np.diff(a[:, 1])
    period = float(np.median(d[d > 0]))
    d = np.where((d <= 0) | (d > 1.0), period, d)
    t_sim = np.concatenate([[0.0], np.cumsum(d)])
    return a[:, 0], t_sim, a[:, 2], a[:, 3], a[:, 4]


def to_sim_time(t_mono: np.ndarray, t_rx: np.ndarray, t_sim: np.ndarray, tol: float = 0.05) -> np.ndarray:
    """Sim time of records written while handling the message received at t_rx (the latest
    one at or before t_mono); NaN when there is none within tol seconds."""
    i = np.searchsorted(t_rx, t_mono, side="right") - 1
    out = np.full(len(t_mono), np.nan)
    ok = (i >= 0) & (t_mono - t_rx[np.clip(i, 0, None)] <= tol)
    out[ok] = t_sim[i[ok]]
    return out


def T_vehicle_cam(cam: CameraModel) -> np.ndarray:
    """Camera optical frame (x right, y down, z forward) in the vehicle frame (X forward,
    Y left, Z up, at the rear axle), pitched down by cam.pitch_deg."""
    th = np.radians(cam.pitch_deg)
    fwd = np.array([np.cos(th), 0.0, -np.sin(th)])
    down = np.array([-np.sin(th), 0.0, -np.cos(th)])
    right = np.cross(down, fwd)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([right, down, fwd])
    T[:3, 3] = [cam.cam_x, 0.0, cam.cam_height - VEHICLE_Z]
    return T


def smooth_trajectory(t: np.ndarray, xyz: np.ndarray, yaw: np.ndarray, rate: float, cutoff: float):
    """Vehicle trajectory on a uniform grid at rate: (ts, xyz, yaw), zero-phase low-passed."""
    ts = np.arange(t[0], t[-1], 1.0 / rate)
    raw = np.column_stack([np.interp(ts, t, xyz[:, k]) for k in range(3)] + [np.interp(ts, t, np.unwrap(yaw))])
    b, a = butter(4, cutoff / (rate / 2.0))
    sm = filtfilt(b, a, raw, axis=0, padtype="even")
    return ts, sm[:, :3], sm[:, 3]


def synth_imu(ts: np.ndarray, xyz: np.ndarray, yaw: np.ndarray, rate: float, rng) -> np.ndarray:
    """(M, 7) [t_s, f_body, w_body]: specific force and angular rate of the vehicle frame
    (flat track: roll = pitch = 0) along the smoothed trajectory."""
    dt = 1.0 / rate
    acc_w = np.gradient(np.gradient(xyz, dt, axis=0), dt, axis=0) + np.array([0.0, 0.0, G])
    c, s = np.cos(yaw), np.sin(yaw)
    f_b = np.column_stack([c * acc_w[:, 0] + s * acc_w[:, 1], -s * acc_w[:, 0] + c * acc_w[:, 1], acc_w[:, 2]])
    w_b = np.column_stack([np.zeros_like(ts), np.zeros_like(ts), np.gradient(yaw, dt)])
    # white noise (density * sqrt(rate)) + random-walk bias
    n = len(ts)
    b_a = np.cumsum(rng.normal(0, IMU_NOISE["acc_walk"] * np.sqrt(dt), (n, 3)), axis=0)
    b_g = np.cumsum(rng.normal(0, IMU_NOISE["gyro_walk"] * np.sqrt(dt), (n, 3)), axis=0)
    f_b += b_a + rng.normal(0, IMU_NOISE["noise_acc"] * np.sqrt(rate), (n, 3))
    w_b += b_g + rng.normal(0, IMU_NOISE["noise_gyro"] * np.sqrt(rate), (n, 3))
    return np.column_stack([ts, f_b, w_b])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run dir with export/ (scripts/export_recording.py) and config.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imu-rate", type=float, default=200.0)
    ap.add_argument("--cutoff", type=float, default=6.0, help="Hz, trajectory low-pass before differentiating")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exp = os.path.join(args.run, "export")
    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    sd = cfg.get("sim_depth") or {}
    if not sd:
        raise SystemExit("the run was recorded without sim_depth: no depth to benchmark on")
    cam = CameraModel.from_yaml(sd.get("camera", "sim_camera/camera.yaml"), scale=float(sd.get("scale", 2.0)))

    t_rx, t_sim, x, y, yaw = sim_log(os.path.join(exp, "sim_messages.csv"))
    p_t, p_yaw = t_sim, yaw
    p_xyz = np.column_stack([x, y, np.full_like(x, VEHICLE_Z)])     # flat track
    # start after the last teleport (reset to the line, or a respawn after a wall hit):
    # a step faster than any car speed
    step_v = np.linalg.norm(np.diff(p_xyz, axis=0), axis=1) / np.diff(p_t)
    jumps = np.nonzero(step_v > 15.0)[0]
    if len(jumps):
        start = np.searchsorted(p_t, p_t[jumps[-1] + 1] + 0.5)
        print(f"{len(jumps)} pose jump(s); last at {p_t[jumps[-1] + 1]:.2f} s, starting at {p_t[start]:.2f} s")
        p_t, p_xyz, p_yaw = p_t[start:], p_xyz[start:], p_yaw[start:]

    with open(os.path.join(exp, "camera.csv")) as f:
        C = [r for r in csv.DictReader(f)]
    c_t = to_sim_time(np.array([float(r["t_mono"]) for r in C]), t_rx, t_sim)

    g_t, g_xyz, g_yaw = smooth_trajectory(p_t, p_xyz, p_yaw, args.imu_rate, args.cutoff)
    imu = synth_imu(g_t, g_xyz, g_yaw, args.imu_rate, np.random.default_rng(args.seed))

    # ground truth: smoothed vehicle pose at each frame -> camera pose
    Tvc = T_vehicle_cam(cam)

    # the car body (no depth there) is painted with shading that is fixed to the camera:
    # features on it would read as no motion, so it is made flat
    body = cam.load_mask()                                # at the output resolution

    os.makedirs(os.path.join(args.out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "depth"), exist_ok=True)
    frames, gt_t, gt_T = [], [], []
    last = -1
    for r, t in zip(C, c_t):
        if not np.isfinite(t) or t < g_t[0] or t > g_t[-1]:
            continue
        t_ns = int(round(t * 1e9))
        if t_ns <= last:
            continue                                      # one frame per sim time
        last = t_ns
        rgb, dep = f"rgb/{t_ns}.png", f"depth/{t_ns}.png"
        img = cv2.imread(os.path.join(exp, r["rgb_png"]), cv2.IMREAD_COLOR)
        if body is not None:
            img[body] = 128
        cv2.imwrite(os.path.join(args.out, rgb), img)
        shutil.copyfile(os.path.join(exp, r["depth_png_mm"]), os.path.join(args.out, dep))
        frames.append((t_ns, rgb, dep))
        psi = float(np.interp(t, g_t, g_yaw))
        Twv = np.eye(4)
        Twv[:3, :3] = [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
        Twv[:3, 3] = [np.interp(t, g_t, g_xyz[:, k]) for k in range(3)]
        gt_t.append(t_ns)
        gt_T.append(Twv @ Tvc)

    with open(os.path.join(args.out, "frames.csv"), "w") as f:
        f.write("t_ns,rgb,depth\n")
        for t, a, b in frames:
            f.write(f"{t},{a},{b}\n")
    imu_ns = imu.copy()
    imu_ns[:, 0] = np.round(imu[:, 0] * 1e9)
    rgbdi.write_imu(os.path.join(args.out, "imu.csv"), imu_ns)
    rgbdi.write_poses(os.path.join(args.out, "groundtruth.csv"), np.array(gt_t), np.array(gt_T))

    fx, fy, cx, cy = cam.intrinsics()
    w, h = cam.size
    calib = {
        "source": f"AutoDRIVE sim run {os.path.abspath(args.run)}",
        "notes": "depth from the 2-D LiDAR (walls extruded, floor plane); IMU synthesised from the "
                 "ground-truth pose (cubic spline) plus consumer-MEMS noise",
        "width": w, "height": h, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "fps": float(1.0 / np.median(np.diff([fr[0] for fr in frames]) * 1e-9)),
        "T_imu_cam": [[float(v) for v in row] for row in Tvc],
        "imu": {"rate_hz": args.imu_rate, **IMU_NOISE},
    }
    with open(os.path.join(args.out, "calib.yaml"), "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    print(f"{len(frames)} frames ({calib['fps']:.1f} fps), {len(imu)} IMU samples, "
          f"{(frames[-1][0] - frames[0][0]) * 1e-9:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
