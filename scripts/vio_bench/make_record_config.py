#!/usr/bin/env python3
"""Write a car config for benchmark recordings (README.md here) from an existing one.

    python scripts/vio_bench/make_record_config.py config.yaml -o config_vio_record.yaml
    python start.py --config config_vio_record.yaml     # drive the laps, [q] to stop

Sets: recording on; the OAK-D's 6-axis IMU recorded (raw_imu) at --imu-rate; every
frame and IMU sample kept (sample_k 1, no file rotation, nothing deleted). Everything
else, including SLAM, control and the operator's arming, stays as in the source config
(the benchmark replays the recording through both systems afterwards).
"""

from __future__ import annotations

import argparse
import sys

import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--imu-rate", type=int, default=200, help="Hz; regenerate the ORB-SLAM3 settings at it")
    ap.add_argument("--recording-path", default="/tmp/pilot_recordings")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("sim"):
        raise SystemExit("this is for the car (OAK-D); sim runs use sim_to_rgbdi.py")
    cfg["recording"] = True
    cfg["recording_path"] = args.recording_path
    cfg["raw_imu"] = True
    cfg.setdefault("camera", {})["imu_rate_hz"] = args.imu_rate
    rc = cfg.setdefault("recording_config", {})
    for name in ("frame", "raw_imu"):
        rc[name] = {"sample_k": 1, "top_n": 0, "keep_files": 0}
    with open(args.output, "w") as f:
        f.write(f"# benchmark recording config from {args.config} "
                "(scripts/vio_bench/make_record_config.py)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {args.output}: recording frames + raw IMU ({args.imu_rate} Hz) to {args.recording_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
