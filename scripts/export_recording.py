#!/usr/bin/env python3
"""Export a pilot_core run (start.py --recording, e.g. via autodrive_bench.py run) to plain
files, every record on one clock, for building synchronised telemetry videos.

    python scripts/export_recording.py <run dir> [--out <run dir>/export] [--no-images]

<run dir> holds rec/<session>/ (the buffer recording) and, from autodrive_bench.py,
screen.mp4 + screen.mp4.json (the sim window) and config.yaml. Writes:

  camera/rgb_NNNNNN.png, camera/depth_NNNNNN.png (uint16 mm), camera.csv
  segmentation/seg_NNNNNN.png (255 = track), segmentation.csv
  pose.csv state.csv control.csv actuation.csv imu.csv
  lidar.npz (t, seq, ranges[N, 1080]) + lidar.csv (per-scan summary)
  pointcloud.npz (t, seq, offsets, points) + pointcloud.csv;  boundary.npz + boundary.csv
  video_frames.csv (sim screen video frame -> time), timeline.csv, timeline_camera.csv
  events.csv (laps, collisions, resets), sim_messages.csv, config.yaml, README.md

Times: t_mono is time.monotonic() on the machine that ran pilot_core (every buffer's
produced_ts); t_s is seconds since the first record of the run. In timeline*.csv each
buffer column is the index of the LATEST record produced at or before that row's time
(what the car had at that moment), with its age in ms.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from read_recording import list_files, load_buffer  # noqa: E402

TIMELINE_BUFFERS = ("camera", "segmentation", "pose", "state", "control", "actuation", "imu", "lidar",
                    "pointcloud", "boundary")


def _write_csv(path: str, header: list[str], rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fmt(x, nd=6):
    return f"{x:.{nd}f}" if isinstance(x, (float, np.floating)) else x


def _video_frame_times(run: str) -> list[float]:
    """Monotonic time of each frame of screen.mp4 (recorder t0 + the frame's own pts)."""
    mp4, meta = os.path.join(run, "screen.mp4"), os.path.join(run, "screen.mp4.json")
    if not (os.path.isfile(mp4) and os.path.isfile(meta)):
        return []
    with open(meta) as f:
        t0 = float(json.load(f)["t0_monotonic"])
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "frame=pts_time", "-of", "csv=p=0", mp4], capture_output=True, text=True, check=True)
    return [t0 + float(x.strip(",")) for x in out.stdout.split() if x.strip(",")]


def _latest_index(t_ref: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    """Index of the last t_ref <= each t_query (sorted t_ref), -1 if none yet."""
    return np.searchsorted(t_ref, t_query, side="right") - 1


def export(run: str, out: str, images: bool = True) -> dict:
    import cv2

    recs = sorted(glob.glob(os.path.join(run, "rec", "*", "")))
    if not recs:
        raise SystemExit(f"no recording under {run}/rec")
    rec = recs[-1]
    os.makedirs(out, exist_ok=True)
    has = lambda name: bool(list_files(rec, name))
    load = lambda name: load_buffer(rec, name) if has(name) else []

    B = {n: load(n) for n in ("frame", "segmentation", "pose", "state", "control", "actuation", "imu",
                              "lidar", "pointcloud", "boundary")}
    t_first = min(b[0].produced_ts for b in B.values() if b)
    rel = lambda t: t - t_first
    times = {}
    summary = {"recording": rec, "t_first_mono": t_first}

    # camera: the RGB-D frame as the pipeline saw it (sim_depth: upscaled, car painted out)
    if B["frame"]:
        os.makedirs(os.path.join(out, "camera"), exist_ok=True)
        rows = []
        for i, r in enumerate(B["frame"]):
            rgb, depth = r.value
            fn_rgb, fn_d = f"camera/rgb_{i:06d}.png", f"camera/depth_{i:06d}.png"
            if images:
                cv2.imwrite(os.path.join(out, fn_rgb), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                d = np.asarray(depth, np.float32)
                cv2.imwrite(os.path.join(out, fn_d), np.clip(np.nan_to_num(d) * 1000.0, 0, 65535).astype(np.uint16))
            rows.append([i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no, rgb.shape[1], rgb.shape[0], fn_rgb, fn_d])
        _write_csv(os.path.join(out, "camera.csv"),
                   ["idx", "t_mono", "t_s", "seq", "width", "height", "rgb_png", "depth_png_mm"], rows)
        times["camera"] = np.array([r.produced_ts for r in B["frame"]])

    if B["segmentation"]:
        os.makedirs(os.path.join(out, "segmentation"), exist_ok=True)
        rows = []
        for i, r in enumerate(B["segmentation"]):
            m = np.asarray(r.value[0] if isinstance(r.value, tuple) else r.value)
            fn = f"segmentation/seg_{i:06d}.png"
            if images:
                cv2.imwrite(os.path.join(out, fn), (m > 0).astype(np.uint8) * 255)
            rows.append([i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no, f"{(m > 0).mean():.4f}", fn])
        _write_csv(os.path.join(out, "segmentation.csv"), ["idx", "t_mono", "t_s", "seq", "track_fraction", "mask_png"], rows)
        times["segmentation"] = np.array([r.produced_ts for r in B["segmentation"]])

    if B["pose"]:
        rows = []
        for i, r in enumerate(B["pose"]):
            M = np.asarray(r.value[0])
            speed = r.value[2] if len(r.value) > 2 else float("nan")
            yaw = float(np.arctan2(M[1, 0], M[0, 0]))
            rows.append([i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no,
                         *[_fmt(float(v)) for v in M[:3, 3]], _fmt(yaw), _fmt(float(speed))])
        _write_csv(os.path.join(out, "pose.csv"), ["idx", "t_mono", "t_s", "seq", "x", "y", "z", "yaw", "speed"], rows)
        times["pose"] = np.array([r.produced_ts for r in B["pose"]])

    if B["state"]:
        rows = [[i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no, _fmt(r.value.t), _fmt(r.value.x),
                 _fmt(r.value.y), _fmt(r.value.yaw), _fmt(r.value.v), _fmt(r.value.yaw_rate)]
                for i, r in enumerate(B["state"])]
        _write_csv(os.path.join(out, "state.csv"),
                   ["idx", "t_mono", "t_s", "seq", "state_t", "x", "y", "yaw", "v", "yaw_rate"], rows)
        times["state"] = np.array([r.produced_ts for r in B["state"]])

    for name in ("control", "actuation"):
        if B[name]:
            rows = [[i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no, _fmt(float(r.value[0])),
                     _fmt(float(r.value[1]))] for i, r in enumerate(B[name])]
            _write_csv(os.path.join(out, f"{name}.csv"),
                       ["idx", "t_mono", "t_s", "seq", "steering_unit", "throttle_unit"], rows)
            times[name] = np.array([r.produced_ts for r in B[name]])

    if B["imu"]:
        rows = [[i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no, *[_fmt(float(v)) for v in np.asarray(r.value)]]
                for i, r in enumerate(B["imu"])]
        _write_csv(os.path.join(out, "imu.csv"), ["idx", "t_mono", "t_s", "seq", "qw", "qx", "qy", "qz",
                                                 "ax", "ay", "az", "extra0", "extra1"], rows)
        times["imu"] = np.array([r.produced_ts for r in B["imu"]])

    if B["lidar"]:
        t = np.array([r.produced_ts for r in B["lidar"]])
        R = np.stack([np.asarray(r.value, np.float32) for r in B["lidar"]])
        np.savez_compressed(os.path.join(out, "lidar.npz"), t_mono=t, t_s=t - t_first,
                            seq=np.array([r.seq_no for r in B["lidar"]]), ranges=R,
                            angle_min=-2.35619449, angle_increment=0.00436332)
        fin = np.where(np.isfinite(R), R, np.nan)
        _write_csv(os.path.join(out, "lidar.csv"), ["idx", "t_mono", "t_s", "seq", "min_range", "front_range"],
                   [[i, _fmt(t[i]), _fmt(t[i] - t_first), B["lidar"][i].seq_no, _fmt(float(np.nanmin(fin[i]))),
                     _fmt(float(fin[i, R.shape[1] // 2]))] for i in range(len(t))])
        times["lidar"] = t

    if B["pointcloud"]:
        t = np.array([r.produced_ts for r in B["pointcloud"]])
        pts = [np.asarray(r.value, np.float32).reshape(-1, 3) for r in B["pointcloud"]]
        offsets = np.cumsum([0] + [len(p) for p in pts])
        np.savez_compressed(os.path.join(out, "pointcloud.npz"), t_mono=t, t_s=t - t_first,
                            seq=np.array([r.seq_no for r in B["pointcloud"]]), offsets=offsets,
                            points=np.concatenate(pts) if pts else np.zeros((0, 3), np.float32))
        _write_csv(os.path.join(out, "pointcloud.csv"), ["idx", "t_mono", "t_s", "seq", "n_points", "cum_points"],
                   [[i, _fmt(t[i]), _fmt(t[i] - t_first), B["pointcloud"][i].seq_no, len(pts[i]), int(offsets[i + 1])]
                    for i in range(len(t))])
        times["pointcloud"] = t

    if B["boundary"]:
        t = np.array([r.produced_ts for r in B["boundary"]])
        np.savez_compressed(os.path.join(out, "boundary.npz"), t_mono=t, t_s=t - t_first,
                            **{f"outer_{i}": np.asarray(r.value[0]) for i, r in enumerate(B["boundary"])},
                            **{f"inner_{i}": np.asarray(r.value[1]) for i, r in enumerate(B["boundary"])})
        _write_csv(os.path.join(out, "boundary.csv"), ["idx", "t_mono", "t_s", "seq", "n_outer", "n_inner"],
                   [[i, _fmt(r.produced_ts), _fmt(rel(r.produced_ts)), r.seq_no, len(r.value[0]), len(r.value[1])]
                    for i, r in enumerate(B["boundary"])])
        times["boundary"] = t

    # sim-side logs
    ev_path = os.path.join(rec, "sim_events.jsonl")
    if os.path.isfile(ev_path):
        with open(ev_path) as f:
            ev = [json.loads(line) for line in f if line.strip()]
        _write_csv(os.path.join(out, "events.csv"), ["t_mono", "t_s", "event", "detail"],
                   [[_fmt(e["t"]), _fmt(e["t"] - t_first), e["event"],
                     json.dumps({k: v for k, v in e.items() if k not in ("t", "event")})] for e in ev])
    for fn in ("sim_messages.csv",):
        if os.path.isfile(os.path.join(rec, fn)):
            shutil.copy(os.path.join(rec, fn), os.path.join(out, fn))
    if os.path.isfile(os.path.join(run, "config.yaml")):
        shutil.copy(os.path.join(run, "config.yaml"), os.path.join(out, "config.yaml"))

    # the master timelines
    cols = [b for b in TIMELINE_BUFFERS if b in times]

    def timeline(path: str, key_name: str, t_rows: np.ndarray) -> None:
        idx = {b: _latest_index(times[b], t_rows) for b in cols}
        header = [key_name, "t_mono", "t_s"] + [h for b in cols for h in (f"{b}_idx", f"{b}_age_ms")]
        rows = []
        for k, t in enumerate(t_rows):
            row = [k, _fmt(t), _fmt(t - t_first)]
            for b in cols:
                i = int(idx[b][k])
                row += [i, f"{(t - times[b][i]) * 1000:.1f}" if i >= 0 else ""]
            rows.append(row)
        _write_csv(path, header, rows)

    vt = np.array(_video_frame_times(run))
    if len(vt):
        shutil.copy(os.path.join(run, "screen.mp4"), os.path.join(out, "screen.mp4"))
        _write_csv(os.path.join(out, "video_frames.csv"), ["video_frame", "t_mono", "t_s"],
                   [[k, _fmt(t), _fmt(t - t_first)] for k, t in enumerate(vt)])
        timeline(os.path.join(out, "timeline.csv"), "video_frame", vt)
    if "camera" in times:
        timeline(os.path.join(out, "timeline_camera.csv"), "row", times["camera"])

    summary.update({b: len(times[b]) for b in times}, video_frames=int(len(vt)))
    with open(os.path.join(out, "export_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--no-images", action="store_true", help="CSV/NPZ only, no PNG per frame")
    a = ap.parse_args()
    s = export(a.run, a.out or os.path.join(a.run, "export"), images=not a.no_images)
    print(json.dumps(s, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
