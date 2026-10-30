#!/usr/bin/env python3
"""Car recording (start.py --recording with raw_imu) -> rgbdi sequence (rgbdi.py).

    python scripts/vio_bench/car_to_rgbdi.py <recording dir> --settings orbslam_oakd_rgbd.yaml --out <seq dir>

<recording dir> holds frame_*.bin (CameraWorker's undistorted RGB + aligned depth) and
raw_imu_*.bin (the OAK-D's accel + gyro), both stamped with the device capture time on
time.monotonic()'s clock. Record with make_record_config.py's config: every frame, no
rotated-out files, the IMU at 200 Hz. The settings file (scripts/gen_orbslam_settings.py
--imu, for the same camera, resolution and IMU rate) gives the intrinsics, IMU.T_b_c1 and
the IMU noise. There is no groundtruth.csv: accuracy comes from AprilTags (markers.py).
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402
from read_recording import iter_records, list_files  # noqa: E402


def read_settings(path: str) -> dict:
    """Intrinsics, T_b_c1 and the IMU block of an ORB-SLAM3 settings file (File.version 1.0)."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    try:
        def num(key):
            node = fs.getNode(key)
            if node.empty():
                raise SystemExit(f"{path} has no {key}" + (" (generate it with --imu)" if key.startswith("IMU") else ""))
            return float(node.real())

        T = fs.getNode("IMU.T_b_c1").mat() if not fs.getNode("IMU.T_b_c1").empty() else None
        if T is None:
            raise SystemExit(f"{path} has no IMU.T_b_c1: generate it with scripts/gen_orbslam_settings.py --imu")
        return {
            "width": int(num("Camera.width")), "height": int(num("Camera.height")),
            "fx": num("Camera1.fx"), "fy": num("Camera1.fy"), "cx": num("Camera1.cx"), "cy": num("Camera1.cy"),
            "fps": num("Camera.fps"),
            "T_imu_cam": np.asarray(T, dtype=np.float64).reshape(4, 4).tolist(),
            "imu": {"rate_hz": num("IMU.Frequency"), "noise_gyro": num("IMU.NoiseGyro"),
                    "noise_acc": num("IMU.NoiseAcc"), "gyro_walk": num("IMU.GyroWalk"),
                    "acc_walk": num("IMU.AccWalk")},
        }
    finally:
        fs.release()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", help="the session directory with frame_*.bin and raw_imu_*.bin")
    ap.add_argument("--settings", default=os.path.join(REPO_ROOT, "orbslam_oakd_rgbd.yaml"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="s to skip at the start")
    ap.add_argument("--end", type=float, help="s from the start to stop at")
    args = ap.parse_args()

    names = {rf.buffer for rf in list_files(args.recording)}
    missing = {"frame", "raw_imu"} - names
    if missing:
        raise SystemExit(f"{args.recording} has no {', '.join(sorted(missing))} recording "
                         "(record with scripts/vio_bench/make_record_config.py's config)")
    calib = read_settings(args.settings)

    imu = np.array([[r.produced_ts, *r.value.tolist()] for r in iter_records(args.recording, "raw_imu")])
    imu = imu[np.argsort(imu[:, 0], kind="stable")]
    t0 = imu[0, 0]

    os.makedirs(os.path.join(args.out, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "depth"), exist_ok=True)
    frames, last, dropped = [], None, 0
    for r in iter_records(args.recording, "frame"):
        rel = r.produced_ts - t0
        if rel < args.start or (args.end is not None and rel > args.end):
            continue
        t_ns = int(round(r.produced_ts * 1e9))
        if last is not None and t_ns <= last:
            dropped += 1
            continue
        last = t_ns
        rgb, depth = r.value
        if rgb.shape[1] != calib["width"] or rgb.shape[0] != calib["height"]:
            raise SystemExit(f"frames are {rgb.shape[1]}x{rgb.shape[0]}, the settings are for "
                             f"{calib['width']}x{calib['height']}")
        a, b = f"rgb/{t_ns}.png", f"depth/{t_ns}.png"
        cv2.imwrite(os.path.join(args.out, a), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        d = np.clip(np.nan_to_num(np.asarray(depth, np.float32)) * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(args.out, b), d)
        frames.append((t_ns, a, b))
    if not frames:
        raise SystemExit("no frames in the selected time range")

    f_t = np.array([f[0] for f in frames]) * 1e-9
    keep = (imu[:, 0] >= f_t[0] - 0.05) & (imu[:, 0] <= f_t[-1] + 0.05)
    imu = imu[keep]
    gaps = np.diff(imu[:, 0])
    rate = 1.0 / float(np.median(gaps))
    imu_ns = imu.copy()
    imu_ns[:, 0] = np.round(imu[:, 0] * 1e9)
    with open(os.path.join(args.out, "frames.csv"), "w") as f:
        f.write("t_ns,rgb,depth\n")
        for t, a, b in frames:
            f.write(f"{t},{a},{b}\n")
    rgbdi.write_imu(os.path.join(args.out, "imu.csv"), imu_ns)
    if "rear_imu" in names:          # the rear IMU, in its own axes, host-stamped (check_rear_imu.py)
        rear = np.array([[r.produced_ts, *r.value.tolist()] for r in iter_records(args.recording, "rear_imu")])
        rear = rear[np.argsort(rear[:, 0], kind="stable")]
        rear = rear[(rear[:, 0] >= f_t[0] - 0.05) & (rear[:, 0] <= f_t[-1] + 0.05)]
        rear[:, 0] = np.round(rear[:, 0] * 1e9)
        rgbdi.write_imu(os.path.join(args.out, "rear_imu.csv"), rear)
    calib.update({"source": f"car recording {os.path.abspath(args.recording)}",
                  "notes": f"settings {os.path.abspath(args.settings)}"})
    if abs(rate - calib["imu"]["rate_hz"]) > 0.1 * calib["imu"]["rate_hz"]:
        print(f"warning: the IMU ran at {rate:.0f} Hz, the settings say {calib['imu']['rate_hz']:.0f} Hz")
    with open(os.path.join(args.out, "calib.yaml"), "w") as f:
        yaml.safe_dump(calib, f, sort_keys=False)
    big = int(np.sum(gaps > 5.0 / rate))
    print(f"{len(frames)} frames ({len(frames) / (f_t[-1] - f_t[0]):.1f} fps), {len(imu)} IMU samples "
          f"({rate:.0f} Hz, {big} gaps > 5 samples), {f_t[-1] - f_t[0]:.1f} s"
          + (f", {dropped} out-of-order frames dropped" if dropped else "") + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
