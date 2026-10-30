#!/usr/bin/env python3
"""Run pilot_core (start.py) against the live AutoDRIVE sim and score it: single runs and
the benchmark matrix (docs/autodrive.md "Benchmark").

    # one run: map with LiDAR, then race N laps on pure_pursuit, recording on
    python scripts/autodrive_bench.py run --out runs/map --laps 3 --set enable_slam=true

    # the matrix: stages A..D from bench/matrix.yaml, 3 repeats per cell
    python scripts/autodrive_bench.py matrix bench/matrix.yaml --stage A --out bench/<date>
    python scripts/autodrive_bench.py summary bench/<date>            # results.csv -> chart + results.md

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
              drive_s: Optional[float] = None, max_collisions: Optional[int] = None) -> dict:
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
    # the run resets the sim and then arms: a config that arms itself on connect (the
    # competition container) would start driving before that reset teleports it
    cfg["auto_arm"] = False
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
                evs = _events(rec_dir)
                done = [e for e in evs if e["event"] == "lap" and e["t"] > t_mapped + 1.0]
                if len(done) >= laps:
                    break
                if max_collisions is not None and sum(
                        e["event"] == "collision" and e["t"] > t_mapped for e in evs) > max_collisions:
                    res["aborted"] = f"more than {max_collisions} collisions"
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


# ---------------------------------------------------------------- matrix

METRICS = ("laps_completed", "best_lap", "mean_lap", "collisions", "xtrack_rms", "xtrack_max", "off_track_pct",
           "speed_mean", "speed_max", "steer_rate_rms", "msg_rate_hz", "frame_rate_hz", "stalled_pct",
           "reply_latency_p50_ms", "reply_latency_p99_ms", "pose_age_p50_ms", "pose_age_p95_ms",
           "control_rate_hz", "wall_s")


def _cell_id(params: dict) -> str:
    return "__".join(f"{k.split('.')[-1]}={v}" for k, v in params.items())


def _rank_key(cell: dict):
    m = cell["metrics"]
    return (m["collisions"]["mean"], -m["laps_completed"]["mean"],
            m["mean_lap"]["mean"] if math.isfinite(m["mean_lap"]["mean"]) else math.inf)


def _load_cells(out: str, stage: str) -> list[dict]:
    with open(os.path.join(out, stage, "cells.json")) as f:
        return json.load(f)


def stage_cells(mx: dict, stage: str, out: str) -> list[dict]:
    """[{id, params, overrides, track, skip}] for one stage of the matrix."""
    import itertools

    st = mx["stages"][stage]
    vary = st["vary"]
    if "from_stage" in st:
        ranked = sorted([c for c in _load_cells(out, st["from_stage"]) if not c.get("skip")], key=_rank_key)
        bases, seen = [], set()
        for c in ranked:
            key = c["params"].get(st["distinct"]) if st.get("distinct") else c["id"]
            if key in seen:
                continue
            seen.add(key)
            bases.append(c)
            if len(bases) == st.get("top", 1):
                break
    else:
        bases = [{"params": {}, "overrides": {}}]
    cells = []
    for base in bases:
        for combo in itertools.product(*vary.values()):
            params = {**{k: v for k, v in base["params"].items() if k not in vary}, **dict(zip(vary, combo))}
            ov = {**mx.get("fixed", {}), **base.get("overrides", {})}
            track, skip = "practice", None
            for k, v in zip(vary, combo):
                if k == "pose_map":
                    skip = mx.get("skip", {}).get(v)
                    ov.update(mx.get("pose_map", {}).get(v, {}))
                elif k == "track":
                    skip = mx.get("skip", {}).get(v)
                    if not skip:
                        track = mx["tracks"][v]["variant"]
                        ov["saved_map_path"] = mx["tracks"][v]["map"]
                else:
                    ov[k] = v
            if "track_control.speed.v_max" in ov and "speed_cap_margin" in st:
                ov["safety.speed_cap"] = float(ov["track_control.speed.v_max"]) + st["speed_cap_margin"]
            cells.append({"id": _cell_id(params), "params": params, "overrides": ov, "track": track, "skip": skip})
    return cells


def ensure_sim_variant(variant: str) -> None:
    """Switch the supervisor-run sim between the practice and compete builds."""
    env = "/workspaces/autodrive_sim/sim.env"
    with open(env) as f:
        lines = f.read().splitlines()
    cur = next((ln.split("=", 1)[1] for ln in lines if ln.startswith("SIM_VARIANT=")), "practice")
    if cur == variant:
        return
    lines = [ln for ln in lines if not ln.startswith("SIM_VARIANT=")] + [f"SIM_VARIANT={variant}"]
    with open(env, "w") as f:
        f.write("\n".join(lines) + "\n")
    log = "/workspaces/logs/autodrive_sim.log"
    size = os.path.getsize(log) if os.path.exists(log) else 0
    subprocess.run(["supervisorctl", "restart", "autodrive_sim"], check=True, capture_output=True)
    deadline = time.monotonic() + 120
    while not _tail_has(log, "UnloadTime", since=size):
        if time.monotonic() > deadline:
            raise RuntimeError(f"sim ({variant}) did not come up")
        time.sleep(1)
    time.sleep(3)


def _summarise(runs: list[dict]) -> dict:
    out = {}
    for k in METRICS:
        vals = np.array([r[k] for r in runs if isinstance(r.get(k), (int, float))], dtype=float)
        fin = vals[np.isfinite(vals)]
        out[k] = {"mean": float(fin.mean()) if len(fin) else math.nan,
                  "std": float(fin.std(ddof=1)) if len(fin) > 1 else (0.0 if len(fin) else math.nan),
                  "n": int(len(fin))}
    return out


def run_matrix(matrix: str, stage: str, out: str, only: Optional[list[str]] = None) -> list[dict]:
    with open(matrix) as f:
        mx = yaml.safe_load(f)
    cells = stage_cells(mx, stage, out)
    os.makedirs(os.path.join(out, stage), exist_ok=True)
    done_cells = []
    for cell in cells:
        if only and cell["id"] not in only:
            continue
        cdir = os.path.join(out, stage, cell["id"])
        os.makedirs(cdir, exist_ok=True)
        runs = []
        if not cell["skip"]:
            ensure_sim_variant(cell["track"])
            if cell["track"] == "compete" and not os.path.isfile(
                    os.path.join(REPO_ROOT, cell["overrides"]["saved_map_path"], "outer_line.npy")):
                print(f"[{stage}] mapping the compete track first", flush=True)
                run_pilot(mx["base_config"], os.path.join(out, "maps", "compete"),
                          {"enable_slam": True, "use_saved_map": False,
                           "mapping.boundary_save_path": cell["overrides"]["saved_map_path"]},
                          laps=0, timeout=600)
            for k in range(mx.get("repeats", 3)):
                rdir = os.path.join(cdir, f"rep{k + 1}")
                if os.path.isfile(os.path.join(rdir, "score.json")):          # resume
                    with open(os.path.join(rdir, "score.json")) as f:
                        runs.append(json.load(f))
                    continue
                print(f"[{stage}] {cell['id']} rep {k + 1}", flush=True)
                try:
                    run = run_pilot(mx["base_config"], rdir, cell["overrides"], laps=mx.get("laps", 2),
                                    timeout=mx.get("timeout_s", 300), max_collisions=mx.get("max_collisions"))
                    sc = score_run(rdir)
                    sc["aborted"] = run.get("aborted")
                except Exception as e:                                # a broken run is a result too
                    sc = {"error": repr(e)}
                    with open(os.path.join(rdir, "score.json"), "w") as f:
                        json.dump(sc, f, indent=1)
                print(f"    laps={sc.get('laps_completed')} best={sc.get('best_lap')} coll={sc.get('collisions')} "
                      f"xtrack={sc.get('xtrack_rms')} stalled={sc.get('stalled_pct')} {sc.get('aborted') or ''}"
                      f"{sc.get('error') or ''}", flush=True)
                runs.append(sc)
        summary = {**{k: cell[k] for k in ("id", "params", "overrides", "track", "skip")},
                   "runs": runs, "metrics": _summarise(runs) if runs else {}}
        with open(os.path.join(cdir, "cell.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        done_cells.append(summary)
    with open(os.path.join(out, stage, "cells.json"), "w") as f:
        json.dump(done_cells, f, indent=1, default=float)
    write_results_csv(out)
    return done_cells


def _skip_status(reason: str) -> tuple[str, str]:
    """A matrix skip reason "FAILED: ..." is a cell that was tried offline and failed."""
    if reason.startswith("FAILED:"):
        return "FAILED", reason[len("FAILED:"):].strip()
    return "SKIPPED", reason


def write_results_csv(out: str) -> str:
    """One row per run, every stage, plus a SKIPPED row per skipped cell."""
    rows = []
    for stage in sorted(d for d in os.listdir(out) if os.path.isfile(os.path.join(out, d, "cells.json"))):
        for c in _load_cells(out, stage):
            base = {"stage": stage, "cell": c["id"], **{k.split(".")[-1]: v for k, v in c["params"].items()},
                    "track": c["track"]}
            if c.get("skip"):
                status, note = _skip_status(c["skip"])
                rows.append({**base, "status": status, "note": note})
                continue
            for i, r in enumerate(c["runs"]):
                status = "ERROR" if r.get("error") else ("ABORTED" if r.get("aborted") else "OK")
                rows.append({**base, "rep": i + 1, "status": status, "note": r.get("error") or r.get("aborted") or "",
                             **{k: r.get(k) for k in METRICS},
                             "lap_times": " ".join(f"{x:.2f}" for x in r.get("lap_times", []))})
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    path = os.path.join(out, "results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return path


# ---------------------------------------------------------------- summary

BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SERIES = (BLUE, ORANGE, AQUA, PURPLE)


def _fmt(m: dict, scale: float = 1.0, nd: int = 2) -> str:
    if not m or m.get("n", 0) == 0 or not math.isfinite(m["mean"]):
        return "-"
    return f"{m['mean'] * scale:.{nd}f} ± {m['std'] * scale:.{nd}f}" if m["n"] > 1 else f"{m['mean'] * scale:.{nd}f}"


def summary_chart(out: str, path: str) -> None:
    """Stage A as lines over v_max per controller (mean lap, cross-track RMS, collisions per
    run, sim stall share), stages B-D as grouped bars; ±1 sd over the repeats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = [st for st in "ABCD" if os.path.isfile(os.path.join(out, st, "cells.json"))]
    panels = [("mean_lap", "mean lap time (s, sim clock)", 1.0), ("xtrack_rms", "cross-track RMS (cm)", 100.0),
              ("collisions", "collisions per run", 1.0), ("stalled_pct", "sim frozen (% of run)", 1.0)]
    fig, axes = plt.subplots(len(stages), len(panels), figsize=(4.2 * len(panels), 3.3 * len(stages)),
                             squeeze=False, facecolor=SURFACE)
    for row, st in zip(axes, stages):
        cells = [c for c in _load_cells(out, st) if not c.get("skip") and c.get("metrics")]
        for ax, (key, label, scale) in zip(row, panels):
            ax.set_facecolor(SURFACE)
            if st == "A":
                ctrls = sorted({c["params"]["control_mode"] for c in cells})
                for color, ctrl in zip(SERIES, ctrls):
                    cs = sorted((c for c in cells if c["params"]["control_mode"] == ctrl),
                                key=lambda c: c["params"]["track_control.speed.v_max"])
                    x = [c["params"]["track_control.speed.v_max"] for c in cs]
                    y = [c["metrics"][key]["mean"] * scale for c in cs]
                    e = [(c["metrics"][key]["std"] or 0) * scale for c in cs]
                    ax.errorbar(x, y, yerr=e, marker="o", ms=4, color=color, label=ctrl, capsize=2, lw=1.6)
                ax.set_xlabel("v_max (m/s)", color=INK2)
            else:
                # label with only the parameters that differ between this stage's cells
                keys = [k for k in cells[0]["params"] if len({str(c["params"][k]) for c in cells}) > 1]
                names = [", ".join(f"{k.split('.')[-1]}={c['params'][k]}" for k in keys) for c in cells]
                y = [c["metrics"][key]["mean"] * scale for c in cells]
                e = [(c["metrics"][key]["std"] or 0) * scale for c in cells]
                ax.barh(range(len(cells)), y, xerr=e, color=BLUE, height=0.6, capsize=2)
                ax.set_xlim(0, 1.1 * max([v + d for v, d in zip(y, e) if math.isfinite(v + d)] + [1.0]))
                ax.set_yticks(range(len(cells)), names if ax is row[0] else [], fontsize=7, color=INK)
                ax.invert_yaxis()
            ax.set_title(f"stage {st}: {label}", loc="left", fontsize=9, color=INK, fontweight="bold")
            ax.grid(color=GRID)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.tick_params(colors=INK2, labelsize=8)
        if st == "A":
            row[0].legend(frameon=False, fontsize=8)
    fig.suptitle("pilot_core on the AutoDRIVE RoboRacer sim (stock 2026-iros build; practice track unless noted), mean ± sd over repeats",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def write_summary(out: str) -> str:
    """results.csv + chart + results.md (tables per stage, best / worst cell)."""
    write_results_csv(out)
    chart = os.path.join(out, "summary.png")
    summary_chart(out, chart)
    lines = ["# AutoDRIVE benchmark results", "",
             "Each cell: repeated `start.py` runs on the live stock sim (practice track unless noted), "
             "racing a saved LiDAR map after a reset to the start line. Lap times are the sim's own "
             "(sim clock, which pauses while the sim is frozen); lap 1 of a run is a standing start. "
             "Cross-track error is the sim's own pose against the map raced on. mean ± sd over repeats.", "",
             f"![summary]({os.path.basename(chart)})", ""]
    cols = [("laps", "laps_completed", 1, 1), ("best lap (s)", "best_lap", 1, 2), ("mean lap (s)", "mean_lap", 1, 2),
            ("collisions", "collisions", 1, 1), ("xtrack RMS (cm)", "xtrack_rms", 100, 1),
            ("xtrack max (cm)", "xtrack_max", 100, 1), ("off-track %", "off_track_pct", 1, 1),
            ("speed mean (m/s)", "speed_mean", 1, 2), ("frames/s", "frame_rate_hz", 1, 1),
            ("sim frozen %", "stalled_pct", 1, 0), ("pose age p50/p95 (ms)", None, 1, 0)]
    for st in "ABCD":
        p = os.path.join(out, st, "cells.json")
        if not os.path.isfile(p):
            continue
        cells = _load_cells(out, st)
        lines += [f"## Stage {st}", "", "| cell | " + " | ".join(c[0] for c in cols) + " |",
                  "|" + " --- |" * (len(cols) + 1)]
        for c in sorted(cells, key=lambda c: (bool(c.get("skip")),) + (_rank_key(c) if c.get("metrics") else (0,))):
            name = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in c["params"].items())
            if c.get("skip"):
                lines.append(f"| {name} | {': '.join(_skip_status(c['skip']))} |" + " |" * (len(cols) - 1))
                continue
            m = c["metrics"]
            vals = []
            for _, key, scale, nd in cols:
                if key is None:
                    vals.append(f"{_fmt(m['pose_age_p50_ms'], 1, 0)} / {_fmt(m['pose_age_p95_ms'], 1, 0)}")
                else:
                    vals.append(_fmt(m[key], scale, nd))
            errs = sum(1 for r in c["runs"] if r.get("error"))
            lines.append(f"| {name}{f' ({errs} run errors)' if errs else ''} | " + " | ".join(vals) + " |")
        lines.append("")
    md = os.path.join(out, "results.md")
    with open(md, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md


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
    r.add_argument("--max-collisions", type=int, help="stop once the run has more collisions than this")
    mxp = sub.add_parser("matrix")
    mxp.add_argument("matrix")
    mxp.add_argument("--stage", required=True)
    mxp.add_argument("--out", required=True)
    mxp.add_argument("--only", action="append", help="cell id(s) to run")
    sc = sub.add_parser("score")
    sc.add_argument("run_dir")
    sm = sub.add_parser("summary")
    sm.add_argument("out")
    g = sub.add_parser("gif")
    g.add_argument("run_dir")
    g.add_argument("--gif-dir", default="/workspaces/proof/gifs")
    g.add_argument("--prefix", required=True)
    g.add_argument("--which", default="mapping,racing")
    g.add_argument("--speed", type=float, default=2.0)
    a = ap.parse_args()
    os.environ.setdefault("DISPLAY", ":99")
    if a.cmd == "summary":
        print(write_summary(a.out))
        return 0
    if a.cmd == "matrix":
        run_matrix(a.matrix, a.stage, a.out, a.only)
        return 0
    if a.cmd == "gif":
        print("\n".join(run_gifs(a.run_dir, a.gif_dir, a.prefix, a.which.split(","), a.speed)))
        return 0
    if a.cmd == "score":
        print(json.dumps(score_run(a.run_dir), indent=1, default=float))
        return 0
    if a.cmd == "run":
        res = run_pilot(a.config, a.out, parse_sets(a.set), a.laps, a.timeout, a.stop, drive_s=a.drive_s,
                        max_collisions=a.max_collisions)
        laps = [e for e in res["events"] if e["event"] == "lap"]
        coll = [e for e in res["events"] if e["event"] == "collision"]
        print(json.dumps({"laps": [(round(e["t"] - res["t_armed"], 1), e.get("lap_time")) for e in laps],
                          "collisions": len(coll), "mapped_after_s": (res.get("t_mapped", math.nan) - res["t_armed"]),
                          "timed_out": res.get("timed_out", False), "exit_code": res.get("exit_code"),
                          "recording": res.get("recording_dir")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
