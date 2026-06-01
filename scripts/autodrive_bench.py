#!/usr/bin/env python3
"""Run pilot_core (start.py) against the live AutoDRIVE sim and score it: single runs and
the benchmark matrix (docs/autodrive.md, step 3).

    # one run: map with LiDAR, then race N laps on pure_pursuit, recording on
    python scripts/autodrive_bench.py run --out runs/map --laps 3 --set enable_slam=true

    # the matrix: stages A..D from bench/matrix.yaml, 3 repeats per cell
    python scripts/autodrive_bench.py matrix bench/matrix.yaml --stage A --out bench/2026-10-19
    python scripts/autodrive_bench.py summary bench/2026-10-19        # results.csv -> chart + results.md

A run: start.py with stdin piped (KeyboardControl reads one key per line), the sim is
connected through its HUD (scripts/autodrive_sim_ctl.py), reset (r), armed (a); laps are
read from sim_events.jsonl, then pilot_core is stopped (q, or SIGTERM / SIGKILL to
check what the sim is left with). The screen is recorded throughout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob
import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Any, Optional

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
PY = sys.executable


def set_path(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def parse_sets(sets: list[str]) -> dict:
    return {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in sets or [])}


def _tail_has(path: str, needle: str, since: int = 0) -> bool:
    try:
        with open(path, errors="ignore") as f:
            f.seek(since)
            return needle in f.read()
    except FileNotFoundError:
        return False


def _wait_log(path: str, needle: str, timeout: float, proc: subprocess.Popen) -> float:
    deadline = time.monotonic() + timeout
    while not _tail_has(path, needle):
        if proc.poll() is not None:
            raise RuntimeError(f"start.py exited ({proc.returncode}) before '{needle}'; see {path}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"no '{needle}' in {path} after {timeout:.0f} s")
        time.sleep(0.1)
    return time.monotonic()


def _events(rec_dir: Optional[str]) -> list[dict]:
    if not rec_dir:
        return []
    try:
        with open(os.path.join(rec_dir, "sim_events.jsonl")) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        return []


def _worker_pids(log_path: str) -> dict[str, int]:
    """name -> pid from start.py's 'Spawned worker: <name> (pid=N)' lines (last wins)."""
    import re

    pids = {}
    with open(log_path, errors="ignore") as f:
        for line in f:
            m = re.search(r"(?:Spawned|Restarted) worker: (\w+) \((?:new )?pid=(\d+)\)", line)
            if m:
                pids[m.group(1)] = int(m.group(2))
    return pids


def _cpu_ticks(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return int(fields[11]) + int(fields[12])      # utime + stime
    except (OSError, IndexError, ValueError):
        return None


def run_pilot(base_config: str, out: str, overrides: dict, laps: int = 3, timeout: float = 600.0,
              stop: str = "quit", capture: bool = True, settle: float = 2.0,
              drive_s: Optional[float] = None) -> dict:
    """One start.py run against the sim. Returns timings, laps and paths.
    stop: quit (key q) | term (SIGTERM to start.py) | kill (SIGKILL to start.py) |
    killall (SIGKILL to every pilot_core process). drive_s: stop that long after the
    racing starts instead of after `laps` laps."""
    import autodrive_capture as cap
    import autodrive_sim_ctl as sim_ctl

    os.makedirs(out, exist_ok=True)
    with open(os.path.join(REPO_ROOT, base_config)) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        set_path(cfg, k, v)
    cfg["recording"] = True
    cfg["recording_path"] = os.path.join(out, "rec")
    cfg_path = os.path.join(out, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log_path = os.path.join(out, "pilot_core.log")
    video = os.path.join(out, "screen.mp4")
    res: dict = {"out": out, "config": cfg_path, "overrides": overrides, "log": log_path}
    if capture:
        res["video"] = video
        res["video_t0"] = cap.start(video)["t0_monotonic"]
    logf = open(log_path, "w")
    t_launch = time.monotonic()
    proc = subprocess.Popen([PY, "-u", "start.py", "--config", cfg_path], cwd=REPO_ROOT, stdin=subprocess.PIPE,
                            stdout=logf, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    res["pid"] = proc.pid

    def key(k: str) -> float:
        proc.stdin.write(k + "\n")
        proc.stdin.flush()
        return time.monotonic()

    try:
        _wait_log(log_path, "AutoDRIVE server listening", 60, proc)
        time.sleep(1.0)
        # the sim does not reconnect on its own: off, then on
        sim_ctl.set_state("disconnected")
        time.sleep(0.5)
        sim_ctl.set_state("connected")
        res["t_connected"] = _wait_log(log_path, "Sim connected", 30, proc)
        sim_ctl.close_menu()
        rec_dirs = sorted(glob.glob(os.path.join(out, "rec", "*")))
        rec_dir = rec_dirs[-1] if rec_dirs else None
        res["recording_dir"] = rec_dir
        time.sleep(1.0)
        res["t_reset"] = key("r")
        time.sleep(settle)
        res["t_armed"] = key("a")
        pids = _worker_pids(log_path)
        pids["start"] = proc.pid
        res["worker_pids"] = pids
        ticks0, t_race_ticks_t = {n: _cpu_ticks(p) for n, p in pids.items()}, time.monotonic()

        mapping = bool(cfg.get("enable_slam")) and not cfg.get("use_saved_map")
        t_mapped = None if mapping else res["t_armed"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"start.py exited ({proc.returncode}) while driving; see {log_path}")
            if t_mapped is None and _tail_has(log_path, "Track mapping complete"):
                t_mapped = time.monotonic()
                res["t_mapped"] = t_mapped
                # CPU is measured over the racing part only
                ticks0, t_race_ticks_t = {n: _cpu_ticks(p) for n, p in pids.items()}, time.monotonic()
            if t_mapped is not None:
                if drive_s is not None:
                    if time.monotonic() - t_mapped >= drive_s:
                        break
                    time.sleep(0.05)
                    continue
                done = [e for e in _events(rec_dir) if e["event"] == "lap" and e["t"] > t_mapped + 1.0]
                if len(done) >= laps:
                    break
            time.sleep(0.2)
        else:
            res["timed_out"] = True
        res["t_stop"] = time.monotonic()
        ticks1 = {n: _cpu_ticks(p) for n, p in pids.items()}
        hz, dt = os.sysconf("SC_CLK_TCK"), res["t_stop"] - t_race_ticks_t
        res["cpu_percent"] = {n: 100.0 * (ticks1[n] - ticks0[n]) / hz / dt for n in pids
                              if ticks0.get(n) is not None and ticks1.get(n) is not None and dt > 0}
        if stop == "quit":
            key("q")
        elif stop == "killall":           # every pilot_core process at once: nobody left to send neutral
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            os.kill(proc.pid, signal.SIGTERM if stop == "term" else signal.SIGKILL)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            res["stop_forced"] = True
        if stop == "kill":                   # SIGKILL leaves the workers orphaned: they exit on parent death
            time.sleep(1.0)
        res["t_exited"] = time.monotonic()
        res["exit_code"] = proc.returncode
        time.sleep(3.0)                      # let the screen show what the car does afterwards
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        logf.close()
        if capture:
            cap.stop(video)
    res["t_launch"] = t_launch
    res["events"] = _events(res.get("recording_dir"))
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def _messages(rec_dir: str) -> dict[str, np.ndarray]:
    with open(os.path.join(rec_dir, "sim_messages.csv")) as f:
        rows = list(csv.DictReader(f))

    def num(v):
        return math.inf if v == "inf" else (math.nan if v == "" else float(v))
    return {k: np.array([num(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def score_run(run_dir: str, vehicle_params: str = "vehicle_params_roboracer.yaml") -> dict:
    """Score a run_pilot() run from its logs: the live trajectory (sim poses) against the
    map it raced on, the sim's laps / collisions, and link timing."""
    from read_recording import load_buffer

    from control.sim import score_trajectory
    from control.types import VehicleParams

    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    rec = run["recording_dir"]
    with open(run["config"]) as f:
        cfg = yaml.safe_load(f)
    t0 = run.get("t_mapped", run["t_armed"])
    t1 = run["t_stop"]
    m = _messages(rec)
    win = (m["t_rx"] >= t0) & (m["t_rx"] <= t1)
    new = win & (m["new_frame"] == 1)

    # the map the controller raced on: the extracted one in the recording, else the saved one
    bnd = load_buffer(rec, "boundary") if glob.glob(os.path.join(rec, "boundary_*")) else []
    if bnd:
        outer, inner = bnd[-1].value
    else:
        path = cfg.get("saved_map_path")
        path = path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
        outer, inner = np.load(os.path.join(path, "outer_line.npy")), np.load(os.path.join(path, "inner_line.npy"))
    params = VehicleParams.load(cfg.get("vehicle_params", vehicle_params))
    poses = np.column_stack([m["t_rx"][new], m["x"][new], m["y"][new], m["speed"][new]])
    r = score_trajectory(poses, outer[:, :2], inner[:, :2], params,
                         steering=m["cmd_steering"][new], throttle=m["cmd_throttle"][new])

    events = run.get("events", [])
    laps = [e for e in events if e["event"] == "lap" and e["t"] > t0 + 1.0]
    coll = [e for e in events if e["event"] == "collision" and t0 <= e["t"] <= t1]
    lap_times = [float(e["lap_time"]) for e in laps if e.get("lap_time") not in (None, math.inf)]

    # link timing: every message, and the new sim frames
    t_new = m["t_rx"][new]
    period = np.diff(t_new) if len(t_new) > 1 else np.array([np.nan])
    reply_ms = (m["t_reply"][win] - m["t_rx"][win]) * 1e3
    stalled = float(period[period > 0.4].sum()) if len(t_new) > 1 else math.nan

    # pose age at command: each controller output vs the newest pose before it
    pose_ts = np.array([rr.produced_ts for rr in load_buffer(rec, "pose", decode=False)])
    ctrl_ts = np.array([rr.produced_ts for rr in load_buffer(rec, "control", decode=False)])
    ctrl_ts = ctrl_ts[(ctrl_ts >= t0) & (ctrl_ts <= t1)]
    k = np.searchsorted(pose_ts, ctrl_ts) - 1
    ok = k >= 0
    pose_age_ms = (ctrl_ts[ok] - pose_ts[k[ok]]) * 1e3

    out = {
        "laps_completed": len(laps), "lap_times": lap_times,
        "best_lap": min(lap_times) if lap_times else math.nan,
        "mean_lap": float(np.mean(lap_times)) if lap_times else math.nan,
        "collisions": len(coll),
        "xtrack_rms": r.xtrack_rms, "xtrack_max": r.xtrack_max, "off_track_pct": 100 * r.off_track_fraction,
        "speed_mean": r.mean_speed, "speed_max": r.max_speed, "steer_rate_rms": r.steer_rate_rms,
        "msg_rate_hz": float(win.sum() / (t1 - t0)), "frame_rate_hz": float(new.sum() / (t1 - t0)),
        "frame_period_p50_ms": float(np.nanmedian(period) * 1e3), "frame_period_p95_ms": float(np.nanpercentile(period, 95) * 1e3),
        "stalled_pct": 100 * stalled / (t1 - t0),
        "reply_latency_p50_ms": float(np.median(reply_ms)), "reply_latency_p99_ms": float(np.percentile(reply_ms, 99)),
        "pose_age_p50_ms": float(np.median(pose_age_ms)) if len(pose_age_ms) else math.nan,
        "pose_age_p95_ms": float(np.percentile(pose_age_ms, 95)) if len(pose_age_ms) else math.nan,
        "control_rate_hz": float(len(ctrl_ts) / (t1 - t0)),
        "wall_s": t1 - t0, "timed_out": bool(run.get("timed_out", False)),
        "cpu_percent": run.get("cpu_percent", {}),
    }
    with open(os.path.join(run_dir, "score.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    return out


def run_gifs(run_dir: str, gif_dir: str, prefix: str, which=("mapping", "racing"), speed: float = 2.0,
             width: int = 640) -> list[str]:
    """Clips of a run's screen recording: the mapping lap (armed -> mapped) and the racing
    part (mapped -> stop), captioned with the sim's lap events."""
    import autodrive_capture as cap

    with open(os.path.join(run_dir, "run.json")) as f:
        run = json.load(f)
    v0 = run["video_t0"]
    t_map = run.get("t_mapped")
    laps = [e for e in run.get("events", []) if e["event"] == "lap"]
    labels = []
    if t_map:
        labels.append((run["t_armed"] - v0, t_map - v0, "mapping lap: LiDAR follow-the-gap"))
    lo = t_map or run["t_armed"]
    marks = [lo] + [e["t"] for e in laps if e["t"] > lo + 1.0] + [run["t_stop"]]
    mode = os.path.basename(run["config"])
    with open(run["config"]) as f:
        ctrl = yaml.safe_load(f).get("control_mode", "pure_pursuit")
    for i in range(len(marks) - 1):
        labels.append((marks[i] - v0, marks[i + 1] - v0, f"{ctrl}: lap {i + 1}" + (f" ({mode})" if mode != "config.yaml" else "")))
    for e in laps:
        if e.get("lap_time") not in (None, math.inf):
            labels.append((e["t"] - v0, e["t"] - v0 + 3.0, f"lap done: {e['lap_time']:.2f} s (sim time)"))
    labels.append((run["t_stop"] - v0, run["t_stop"] - v0 + 6.0, "pilot_core stopped"))
    os.makedirs(gif_dir, exist_ok=True)
    out = []
    video = os.path.join(run_dir, "screen.mp4")
    if "mapping" in which and t_map:
        out.append(cap.gif(video, os.path.join(gif_dir, f"{prefix}_mapping_lap.gif"),
                           run["t_armed"] - v0 - 1.0, t_map - v0 + 2.0, width=width, fps=8, speed=speed, labels=labels))
    if "racing" in which:
        out.append(cap.gif(video, os.path.join(gif_dir, f"{prefix}_racing.gif"),
                           lo - v0 - 1.0, run["t_stop"] - v0 + 4.0, width=width, fps=8, speed=speed, labels=labels))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--config", default="config_autodrive.yaml")
    r.add_argument("--set", action="append", default=[], help="dotted.key=value config override")
    r.add_argument("--laps", type=int, default=3)
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--stop", choices=["quit", "term", "kill", "killall"], default="quit")
    r.add_argument("--drive-s", type=float, help="stop after this long racing instead of --laps")
    sc = sub.add_parser("score")
    sc.add_argument("run_dir")
    g = sub.add_parser("gif")
    g.add_argument("run_dir")
    g.add_argument("--gif-dir", default="/workspaces/proof/gifs")
    g.add_argument("--prefix", required=True)
    g.add_argument("--which", default="mapping,racing")
    g.add_argument("--speed", type=float, default=2.0)
    a = ap.parse_args()
    os.environ.setdefault("DISPLAY", ":99")
    if a.cmd == "gif":
        print("\n".join(run_gifs(a.run_dir, a.gif_dir, a.prefix, a.which.split(","), a.speed)))
        return 0
    if a.cmd == "score":
        print(json.dumps(score_run(a.run_dir), indent=1, default=float))
        return 0
    if a.cmd == "run":
        res = run_pilot(a.config, a.out, parse_sets(a.set), a.laps, a.timeout, a.stop, drive_s=a.drive_s)
        laps = [e for e in res["events"] if e["event"] == "lap"]
        coll = [e for e in res["events"] if e["event"] == "collision"]
        print(json.dumps({"laps": [(round(e["t"] - res["t_armed"], 1), e.get("lap_time")) for e in laps],
                          "collisions": len(coll), "mapped_after_s": (res.get("t_mapped", math.nan) - res["t_armed"]),
                          "timed_out": res.get("timed_out", False), "exit_code": res.get("exit_code"),
                          "recording": res.get("recording_dir")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
