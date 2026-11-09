#!/usr/bin/env python3
"""Per-stage rates and latencies of a recorded run (start.py --recording).

Every record carries the time its sample was produced and the time the recorder read it
(within ~2 ms of the write). Frames, masks and poses are stamped with the camera's capture
time, so for them read - produced is the capture-to-output latency of that stage. For each
actuator command it also reports the age of the newest pose then: how old the vehicle state
behind the command was when it went out.

    python scripts/stage_latency.py /tmp/pilot_recordings/<run>
    python scripts/stage_latency.py <run> --json > latency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from read_recording import iter_file, list_files  # noqa: E402

# buffers stamped with the camera's capture time: their age is the stage's latency
CAPTURE_STAMPED = ("frame", "segmentation", "pose")
ORDER = ("frame", "imu", "segmentation", "pose", "state", "control", "actuation")


def _times(rec: str, buffer: str) -> tuple[np.ndarray, np.ndarray]:
    produced, read = [], []
    for rf in list_files(rec, buffer):
        for r in iter_file(rf, decode=False):
            produced.append(r.produced_ts)
            read.append(r.read_ts)
    return np.asarray(produced), np.asarray(read)


def _pct(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {}
    return {"p50_ms": float(np.percentile(x, 50) * 1e3), "p95_ms": float(np.percentile(x, 95) * 1e3),
            "max_ms": float(x.max() * 1e3)}


def analyse(rec: str) -> dict:
    out = {}
    present = {rf.buffer for rf in list_files(rec)}
    for name in [b for b in ORDER if b in present]:
        produced, read = _times(rec, name)
        if len(produced) < 2:
            continue
        span = float(produced[-1] - produced[0])
        row = {"n": int(len(produced)), "rate_hz": float((len(produced) - 1) / span) if span > 0 else 0.0,
               "interval": _pct(np.diff(produced))}
        if name in CAPTURE_STAMPED:
            row["latency"] = _pct(read - produced)
        out[name] = row
    if "actuation" in present and "pose" in present:
        p_prod, p_read = _times(rec, "pose")
        a_prod, _ = _times(rec, "actuation")
        order = np.argsort(p_read)
        p_prod, p_read = p_prod[order], p_read[order]
        i = np.searchsorted(p_read, a_prod, side="right") - 1
        ok = i >= 0
        out["pose_age_at_actuation"] = _pct(a_prod[ok] - p_prod[i[ok]])
    return out


def table(res: dict) -> str:
    lines = ["| stage | samples | rate (Hz) | interval p50 / p95 (ms) | capture to output p50 / p95 / max (ms) |",
             "|---|---|---|---|---|"]
    for name in ORDER:
        r = res.get(name)
        if not r:
            continue
        iv = r["interval"]
        lat = r.get("latency")
        lat_s = f"{lat['p50_ms']:.1f} / {lat['p95_ms']:.1f} / {lat['max_ms']:.1f}" if lat else "-"
        lines.append(f"| {name} | {r['n']} | {r['rate_hz']:.1f} | {iv['p50_ms']:.1f} / {iv['p95_ms']:.1f} | {lat_s} |")
    a = res.get("pose_age_at_actuation")
    if a:
        lines.append("")
        lines.append(f"Pose age when a command went out: p50 {a['p50_ms']:.1f} ms, p95 {a['p95_ms']:.1f} ms, "
                     f"max {a['max_ms']:.1f} ms")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="recording directory")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args()
    res = analyse(a.recording)
    print(json.dumps(res, indent=1) if a.json else table(res))


if __name__ == "__main__":
    main()
