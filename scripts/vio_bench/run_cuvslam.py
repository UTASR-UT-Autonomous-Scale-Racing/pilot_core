#!/usr/bin/env python3
"""cuVSLAM (PyCuVSLAM 17.x, release wheel) on an rgbdi sequence.

    python scripts/vio_bench/run_cuvslam.py <seq dir> --out <result dir> [--no-imu] [--odom-only]

Multisensor odometry with the one RGB-D camera plus the IMU (with --no-imu, the RGBD
mode), and SLAM on top unless --odom-only. The rig frame is the camera's. Each frame
is preceded by every IMU sample up to its time. Writes poses.csv (SLAM poses, or odometry
with --odom-only), odom_poses.csv, frames.csv (every frame: tracked, track time in ms)
and meta.json. Runs in its own venv (the wheel pins numpy).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpulimit  # noqa: E402,F401  (before cv2 and cuvslam: sizes their thread pools)

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402


def pose_to_T(pose) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(pose.rotation)).as_matrix()
    T[:3, 3] = np.asarray(pose.translation)
    return T


def T_to_pose(vslam, T: np.ndarray):
    return vslam.Pose(rotation=Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
                      translation=T[:3, 3].tolist())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imu-noise-scale", type=float, default=1.0,
                    help="multiply calib.yaml's IMU noise densities and random walks (tuning)")
    ap.add_argument("--no-imu", action="store_true", help="RGBD odometry, no IMU")
    ap.add_argument("--odom-only", action="store_true", help="no SLAM (no loop closure)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed: frames are fed on the sequence's clock (1 = real time, as on the car), "
                         "0 = as fast as tracking allows (the mapping threads then fall behind)")
    ap.add_argument("--cpus", type=int, help="pin to this many cores (cpulimit.py); default the cgroup's CPU cap")
    args = ap.parse_args()

    import cuvslam as vslam

    seq = rgbdi.load(args.seq)
    c = seq.calib
    T = vslam.Tracker

    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole)
    cam.focal = [c["fx"], c["fy"]]
    cam.principal = [c["cx"], c["cy"]]
    cam.size = [c["width"], c["height"]]
    cam.rig_from_camera = T_to_pose(vslam, np.eye(4))
    rig = vslam.Rig()
    rig.cameras = [cam]

    inertial = not args.no_imu
    if inertial:
        im = c["imu"]
        imu = vslam.ImuCalibration()
        imu.rig_from_imu = T_to_pose(vslam, np.linalg.inv(seq.T_imu_cam))    # rig = camera
        imu.gyroscope_noise_density = im["noise_gyro"] * args.imu_noise_scale
        imu.gyroscope_random_walk = im["gyro_walk"] * args.imu_noise_scale
        imu.accelerometer_noise_density = im["noise_acc"] * args.imu_noise_scale
        imu.accelerometer_random_walk = im["acc_walk"] * args.imu_noise_scale
        imu.frequency = im["rate_hz"]
        rig.imus = [imu]
        cfg = T.OdometryConfig(
            odometry_mode=T.OdometryMode.Multisensor,
            multisensor_settings=T.OdometryMultisensorSettings(depth_camera_ids=[0], depth_scale_factor=1000.0),
        )
    else:
        cfg = T.OdometryConfig(
            odometry_mode=T.OdometryMode.RGBD,
            rgbd_settings=T.OdometryRGBDSettings(depth_camera_id=0, depth_scale_factor=1000.0),
        )
    slam_cfg = None if args.odom_only else T.SlamConfig()
    vslam.warm_up_gpu()
    tracker = vslam.Tracker(rig, cfg, slam_cfg)

    k = int(np.searchsorted(seq.imu[:, 0], seq.t_ns[0], side="right")) if inertial else 0
    rows, odom_t, odom_T, slam_t, slam_T = [], [], [], [], []
    wall0 = time.perf_counter()
    for t_ns, rp, dp in zip(seq.t_ns, seq.rgb, seq.depth):
        rgb = cv2.cvtColor(cv2.imread(rp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        if inertial:
            j = int(np.searchsorted(seq.imu[:, 0], t_ns, side="right"))
            for r in seq.imu[k:j]:
                m = vslam.ImuMeasurement()
                m.timestamp_ns = int(r[0])
                m.linear_accelerations = np.asarray(r[1:4])
                m.angular_velocities = np.asarray(r[4:7])
                tracker.register_imu_measurement(0, m)
            k = j
        if args.speed > 0:                 # feed on the sequence's clock
            due = wall0 + (t_ns - seq.t_ns[0]) * 1e-9 / args.speed
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        t0 = time.perf_counter()
        odom, slam = tracker.track(int(t_ns), [rgb], depths=[depth])
        ms = 1e3 * (time.perf_counter() - t0)
        ok = odom.world_from_rig is not None
        rows.append((int(t_ns), int(ok), ms))
        if ok:
            odom_t.append(int(t_ns))
            odom_T.append(pose_to_T(odom.world_from_rig.pose))
        if slam is not None:
            slam_t.append(int(t_ns))
            slam_T.append(pose_to_T(slam))

    os.makedirs(args.out, exist_ok=True)
    rgbdi.write_poses(os.path.join(args.out, "odom_poses.csv"), np.array(odom_t, np.int64),
                      np.array(odom_T) if odom_T else np.zeros((0, 4, 4)))
    # the SLAM pose as track() returned it at the time (what a car would have had then)
    main_t, main_T = (odom_t, odom_T) if args.odom_only else (slam_t, slam_T)
    # SLAM poses only count on frames odometry tracked
    if not args.odom_only:
        tracked = set(odom_t)
        keep = [i for i, t in enumerate(main_t) if t in tracked]
        main_t, main_T = [main_t[i] for i in keep], [main_T[i] for i in keep]
    rgbdi.write_poses(os.path.join(args.out, "poses.csv"), np.array(main_t, np.int64),
                      np.array(main_T) if main_T else np.zeros((0, 4, 4)))
    with open(os.path.join(args.out, "frames.csv"), "w") as f:
        f.write("t_ns,tracked,track_ms\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.3f}\n")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        gpu = "none"
    meta = {"system": f"cuVSLAM {vslam.get_version()}",
            "mode": ("Multisensor RGB-D + IMU" if inertial else "RGBD") + ("" if args.odom_only else " + SLAM"),
            "speed": args.speed, "cpus": cpulimit.CPUS,
            "gpu": gpu, "imu_noise_scale": args.imu_noise_scale, "sequence": os.path.abspath(args.seq),
            "frames": len(rows), "tracked": len(main_t)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
