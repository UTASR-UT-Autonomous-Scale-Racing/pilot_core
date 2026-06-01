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


def run_pilot(base_config: str, out: str, overrides: dict, laps: int = 3, timeout: float = 600.0,
              stop: str = "quit", capture: bool = True, settle: float = 2.0) -> dict:
    """One start.py run against the sim. Returns timings, laps and paths."""
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

        mapping = bool(cfg.get("enable_slam")) and not cfg.get("use_saved_map")
        t_mapped = None if mapping else res["t_armed"]
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"start.py exited ({proc.returncode}) while driving; see {log_path}")
            if t_mapped is None and _tail_has(log_path, "Track mapping complete"):
                t_mapped = time.monotonic()
                res["t_mapped"] = t_mapped
            if t_mapped is not None:
                done = [e for e in _events(rec_dir) if e["event"] == "lap" and e["t"] > t_mapped + 1.0]
                if len(done) >= laps:
                    break
            time.sleep(0.2)
        else:
            res["timed_out"] = True
        res["t_stop"] = time.monotonic()
        if stop == "quit":
            key("q")
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--config", default="config_autodrive.yaml")
    r.add_argument("--set", action="append", default=[], help="dotted.key=value config override")
    r.add_argument("--laps", type=int, default=3)
    r.add_argument("--timeout", type=float, default=600)
    r.add_argument("--stop", choices=["quit", "term", "kill"], default="quit")
    a = ap.parse_args()
    os.environ.setdefault("DISPLAY", ":99")
    if a.cmd == "run":
        res = run_pilot(a.config, a.out, parse_sets(a.set), a.laps, a.timeout, a.stop)
        laps = [e for e in res["events"] if e["event"] == "lap"]
        coll = [e for e in res["events"] if e["event"] == "collision"]
        print(json.dumps({"laps": [(round(e["t"] - res["t_armed"], 1), e.get("lap_time")) for e in laps],
                          "collisions": len(coll), "mapped_after_s": (res.get("t_mapped", math.nan) - res["t_armed"]),
                          "timed_out": res.get("timed_out", False), "exit_code": res.get("exit_code"),
                          "recording": res.get("recording_dir")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
