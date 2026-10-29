#!/usr/bin/env python3
"""ORB-SLAM3 (the UTASR CUDA fork, through orbslam3_python) on an rgbdi sequence.

    python scripts/vio_bench/run_orbslam3.py <seq dir> --out <result dir> [--no-imu]

IMU_RGBD by default (RGBD with --no-imu). Each frame gets the IMU samples since the
previous one. Poses are taken as pilot_core's SLAMWorker publishes them: relative to the
map's origin keyframe, so the gravity re-rotations during IMU initialisation don't move
them. Writes poses.csv (tracked frames), frames.csv (every frame: tracking state, map id,
track time in ms, IMU initialised) and meta.json. ORB_SLAM3_CUDA=0 runs the CPU path.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpulimit  # noqa: E402,F401  (before cv2 and the SLAM module: sizes their thread pools)

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402

SETTINGS = """%YAML:1.0
File.version: "1.0"
Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps:.0f}
Camera.RGB: 1
RGBD.DepthMapFactor: 1.0
Stereo.b: {baseline:.6f}
Stereo.ThDepth: {th_depth:.3f}
ORBextractor.nFeatures: {features}
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""

IMU_BLOCK = """IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [{T}]
IMU.InsertKFsWhenLost: 0
IMU.UpstreamPolicy: {upstream_policy}
IMU.InitTime: {init_time:.3f}
IMU.EarlyLostTimeout: {early_lost:.3f}
IMU.NoiseGyro: {noise_gyro:g}
IMU.NoiseAcc: {noise_acc:g}
IMU.GyroWalk: {gyro_walk:g}
IMU.AccWalk: {acc_walk:g}
IMU.Frequency: {rate_hz:.1f}
"""

_TRACKING_OK = 2


def gpu_name() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        return "none"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imu-noise-scale", type=float, default=1.0,
                    help="multiply calib.yaml's IMU noise densities and random walks (tuning)")
    ap.add_argument("--no-imu", action="store_true", help="RGBD instead of IMU_RGBD")
    ap.add_argument("--imu-init-time", type=float, default=1.0,
                    help="s of keyframes before the first IMU initialisation (IMU.InitTime; upstream 1.0)")
    ap.add_argument("--early-lost-timeout", type=float, default=0.5,
                    help="s lost before inertial BA 2 after which the map is given up (IMU.EarlyLostTimeout; "
                         "upstream 5.0)")
    ap.add_argument("--upstream-policy", action="store_true",
                    help="upstream ORB-SLAM3's inertial policy (IMU.UpstreamPolicy: 1): reset the map on a "
                         "tracking loss before the IMU is initialised (and before inertial BA 2), no "
                         "keyframes for weak visual tracking; default: the fork's depth-inertial one")
    ap.add_argument("--features", type=int, default=1000)
    ap.add_argument("--depth-baseline", type=float,
                    help="m, the virtual stereo baseline ORB-SLAM3 weighs depth with (Stereo.b); default "
                         "calib.yaml's depth_baseline_m, else the OAK-D's 0.075. Smaller = depth trusted less")
    ap.add_argument("--close-depth", type=float, default=3.0,
                    help="m, depth below which points are 'close' (Stereo.ThDepth = this / baseline)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed: frames are fed on the sequence's clock (1 = real time, as on the car), "
                         "0 = as fast as tracking allows (the mapping threads then fall behind)")
    ap.add_argument("--cpus", type=int, help="pin to this many cores (cpulimit.py); default the cgroup's CPU cap")
    ap.add_argument("--vocab", default=os.path.join(REPO_ROOT, "ORB_SLAM3/Vocabulary/ORBvoc.txt"))
    args = ap.parse_args()

    import importlib
    importlib.import_module("ssl")       # see SLAMWorker.setup: libcrypto load order
    import orbslam3_python as slam_mod

    seq = rgbdi.load(args.seq)
    c = seq.calib
    os.makedirs(args.out, exist_ok=True)
    settings = os.path.join(args.out, "orbslam3_settings.yaml")
    baseline = args.depth_baseline or float(c.get("depth_baseline_m", 0.075))
    with open(settings, "w") as f:
        f.write(SETTINGS.format(fx=c["fx"], fy=c["fy"], cx=c["cx"], cy=c["cy"], width=c["width"],
                                height=c["height"], fps=c.get("fps", 30.0), features=args.features,
                                baseline=baseline, th_depth=args.close_depth / baseline))
        if not args.no_imu:
            im = {k: (v * args.imu_noise_scale if k != "rate_hz" else v) for k, v in c["imu"].items()}
            f.write(IMU_BLOCK.format(T=", ".join(f"{v:.9f}" for v in seq.T_imu_cam.ravel()),
                                     upstream_policy=int(args.upstream_policy),
                                     init_time=args.imu_init_time, early_lost=args.early_lost_timeout, **im))

    inertial = not args.no_imu
    system = slam_mod.System(args.vocab, settings, slam_mod.IMU_RGBD if inertial else slam_mod.RGBD)

    imu = seq.imu
    imu_s = imu.copy()
    imu_s[:, 0] = imu[:, 0] * 1e-9
    k = 0
    if inertial:                       # samples before the first frame are not used
        k = int(np.searchsorted(imu[:, 0], seq.t_ns[0], side="right"))
    poses_t, poses_T, rows = [], [], []
    wall0 = time.perf_counter()
    for i, (t_ns, rp, dp) in enumerate(zip(seq.t_ns, seq.rgb, seq.depth)):
        rgb = cv2.cvtColor(cv2.imread(rp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32) * 1e-3
        batch = None
        if inertial:
            j = int(np.searchsorted(imu[:, 0], t_ns, side="right"))
            batch = np.ascontiguousarray(imu_s[k:j])
            k = j
        if args.speed > 0:                 # feed on the sequence's clock
            due = wall0 + (t_ns - seq.t_ns[0]) * 1e-9 / args.speed
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        t0 = time.perf_counter()
        Tcw = system.track_rgbd(rgb, depth, t_ns * 1e-9, batch) if inertial else \
            system.track_rgbd(rgb, depth, t_ns * 1e-9)
        ms = 1e3 * (time.perf_counter() - t0)
        state = int(system.get_tracking_state())
        map_id = int(system.map_info()[0])
        imu_init = bool(system.imu_initialized()) if inertial else False
        rows.append((int(t_ns), state, map_id, ms, int(imu_init)))
        if state == _TRACKING_OK:
            Twc = np.linalg.inv(np.asarray(Tcw, dtype=np.float64))
            if inertial:
                Twc = np.linalg.inv(np.asarray(system.origin_pose(), dtype=np.float64)) @ Twc
            poses_t.append(int(t_ns))
            poses_T.append(Twc)
    system.shutdown()

    rgbdi.write_poses(os.path.join(args.out, "poses.csv"), np.array(poses_t, np.int64),
                      np.array(poses_T) if poses_T else np.zeros((0, 4, 4)))
    with open(os.path.join(args.out, "frames.csv"), "w") as f:
        f.write("t_ns,state,map_id,track_ms,imu_initialized\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.3f},{r[4]}\n")
    meta = {"system": "ORB-SLAM3 (UTASR CUDA fork)", "mode": "IMU_RGBD" if inertial else "RGBD",
            "cuda": os.environ.get("ORB_SLAM3_CUDA", "1") != "0", "gpu": gpu_name(),
            "cpu": platform.processor() or platform.machine(), "features": args.features,
            "depth_baseline_m": baseline, "speed": args.speed, "cpus": cpulimit.CPUS,
            "imu_noise_scale": args.imu_noise_scale, "upstream_policy": args.upstream_policy,
            "imu_init_time": args.imu_init_time, "early_lost_timeout": args.early_lost_timeout,
            "sequence": os.path.abspath(args.seq), "frames": len(rows), "tracked": len(poses_t)}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
