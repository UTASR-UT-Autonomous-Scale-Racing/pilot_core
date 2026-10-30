#!/usr/bin/env python3
"""Add a simulated rear-of-car IMU (the Arduino's ICM-20948) to an rgbdi sequence.

    python scripts/vio_bench/make_rear_imu.py <seq dir> [--poll-hz 100] [--dlpf-hz 24]

The sequence's IMU (imu.csv) stands for the OAK-D's: device-timestamped on the frames'
clock, what ORB-SLAM3 uses. This one is the second IMU, at the rear axle in the vehicle
frame (x forward, y left, z up), as the car reads it today:

- ICM-20948 noise densities (gyro 0.015 deg/s/rtHz, accel 230 ug/rtHz) plus a constant
  bias left after the firmware's start-up calibration and a slow random walk;
- the chip's digital low-pass filter (DLPF ~24 Hz, a 2nd-order causal filter at 1.1 kHz:
  it delays the signal, ~10 ms);
- the Jetson polling the latest sample over I2C at --poll-hz with timing jitter, each
  sample stamped with the host's clock when the read returns (not when it was measured).

Truth comes from groundtruth.csv (camera poses -> vehicle poses, cubic spline; the track
is flat). Writes rear_imu.csv (t_ns, ax, ay, az [m/s^2], gx, gy, gz [rad/s]) and
rear_truth.csv (t_ns at the poll times: x, y, yaw, vx, vy, yaw rate of the IMU point, for
scoring), and a rear_imu block in calib.yaml.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml
from scipy.interpolate import CubicSpline
from scipy.signal import butter, lfilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402

G = 9.81
ICM20948 = {"noise_gyro": np.radians(0.015), "noise_acc": 230e-6 * G,     # per sqrt(Hz)
            "bias_gyro": np.radians(0.1), "bias_acc": 0.02 * G,             # 1-sigma, after calibration
            "gyro_walk": 2e-4, "acc_walk": 2e-3}                            # per sqrt(s)
INTERNAL_HZ = 1125.0


def vehicle_truth(seq: rgbdi.Sequence, rate: float):
    """IMU-point (vehicle frame origin) trajectory at rate: t, x, y, yaw, body velocity,
    body acceleration (specific force, gravity on z) and yaw rate."""
    T_wv = seq.gt_T @ np.linalg.inv(seq.T_imu_cam)          # T_imu_cam is T_vehicle_cam here
    t = seq.gt_t_ns * 1e-9
    yaw = np.unwrap(np.arctan2(T_wv[:, 1, 0], T_wv[:, 0, 0]))
    sx, sy, syaw = (CubicSpline(t, T_wv[:, 0, 3]), CubicSpline(t, T_wv[:, 1, 3]), CubicSpline(t, yaw))
    ts = np.arange(t[0], t[-1], 1.0 / rate)
    psi, r = syaw(ts), syaw(ts, 1)
    c, s = np.cos(psi), np.sin(psi)
    vxw, vyw, axw, ayw = sx(ts, 1), sy(ts, 1), sx(ts, 2), sy(ts, 2)
    vb = np.column_stack([c * vxw + s * vyw, -s * vxw + c * vyw])
    ab = np.column_stack([c * axw + s * ayw, -s * axw + c * ayw, np.full_like(ts, G)])
    return ts, sx(ts), sy(ts), psi, vb, ab, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq")
    ap.add_argument("--poll-hz", type=float, default=100.0)
    ap.add_argument("--dlpf-hz", type=float, default=24.0, help="0 = no filter")
    ap.add_argument("--jitter-ms", type=float, default=1.0, help="1-sigma poll timing jitter")
    ap.add_argument("--read-ms", type=float, default=0.6, help="I2C read time (60 bytes at 1 MHz)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    seq = rgbdi.load(args.seq)
    ts, x, y, psi, vb, ab, r = vehicle_truth(seq, INTERNAL_HZ)
    n, dt = len(ts), 1.0 / INTERNAL_HZ
    gyro = np.column_stack([np.zeros(n), np.zeros(n), r])
    acc = ab.copy()
    b_g = rng.normal(0, ICM20948["bias_gyro"], 3) + np.cumsum(rng.normal(0, ICM20948["gyro_walk"] * np.sqrt(dt), (n, 3)), 0)
    b_a = rng.normal(0, ICM20948["bias_acc"], 3) + np.cumsum(rng.normal(0, ICM20948["acc_walk"] * np.sqrt(dt), (n, 3)), 0)
    gyro += b_g + rng.normal(0, ICM20948["noise_gyro"] * np.sqrt(INTERNAL_HZ / 2), (n, 3))
    acc += b_a + rng.normal(0, ICM20948["noise_acc"] * np.sqrt(INTERNAL_HZ / 2), (n, 3))
    if args.dlpf_hz > 0:
        b, a = butter(2, args.dlpf_hz / (INTERNAL_HZ / 2))
        gyro = lfilter(b, a, gyro - gyro[:1], axis=0) + gyro[:1]
        acc = lfilter(b, a, acc - acc[:1], axis=0) + acc[:1]

    # the Jetson polls: each read returns the latest internal sample and is stamped when it returns
    polls = np.arange(ts[0] + 0.05, ts[-1], 1.0 / args.poll_hz)
    polls = polls + rng.normal(0, args.jitter_ms * 1e-3, len(polls))
    polls = np.sort(polls[(polls > ts[0]) & (polls < ts[-1])])
    idx = np.searchsorted(ts, polls, side="right") - 1
    stamp = polls + args.read_ms * 1e-3
    out = np.column_stack([np.round(stamp * 1e9), acc[idx], gyro[idx]])
    rgbdi.write_imu(os.path.join(args.seq, "rear_imu.csv"), out)

    # truth at the poll times (where the estimate is scored)
    k = np.clip(np.searchsorted(ts, stamp), 0, n - 1)
    with open(os.path.join(args.seq, "rear_truth.csv"), "w") as f:
        f.write("t_ns,x,y,yaw,vx,vy,yaw_rate\n")
        for j in range(len(stamp)):
            i = k[j]
            f.write(f"{int(round(stamp[j] * 1e9))},{x[i]:.6f},{y[i]:.6f},{psi[i]:.6f},{vb[i, 0]:.6f},{vb[i, 1]:.6f},{r[i]:.6f}\n")

    calib_path = os.path.join(args.seq, "calib.yaml")
    with open(calib_path) as f:
        calib = yaml.safe_load(f)
    calib["rear_imu"] = {
        "model": "ICM-20948 (Arduino, I2C)", "T_vehicle_imu": np.eye(4).tolist(),
        "poll_hz": args.poll_hz, "dlpf_hz": args.dlpf_hz, "jitter_ms": args.jitter_ms, "read_ms": args.read_ms,
        **{k: float(v) for k, v in ICM20948.items()},
    }
    with open(calib_path, "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    print(f"{len(out)} rear IMU samples at {args.poll_hz:g} Hz (DLPF {args.dlpf_hz:g} Hz) -> {args.seq}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
