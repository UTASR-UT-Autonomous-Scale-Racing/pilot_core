"""Throttle -> steady-state speed and steering -> curvature calibration on the live sim
used by scripts/autodrive_validate.py calibrate (docs/autodrive.md "Calibration").

Each throttle level starts from the reset pose and holds that throttle while pure_pursuit
on the saved map steers; the steady-state speed is the median sim speed once the first
half of the level is over. The safety supervisor still runs (arming, watchdog) but its
speed bound is lifted so it doesn't shape the measurement. speed_per_throttle is the
least-squares slope through the origin. From the same frames, the effective wheelbase is
L = v tan(delta) / yaw_rate with delta = steering * max_steering_angle.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from autodrive_validate import REPO_ROOT, SimSession, load_messages, load_safety, segment

LEVELS = (0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10)
HOLD_S = 6.0


def run_calibration(out: str, config: str, map_dir: str) -> dict:
    import autodrive_capture as cap

    from control.controllers import PurePursuit
    from control.track import TrackModel
    from control.types import VehicleParams
    from workers.control_worker import signed_to_unit

    import yaml
    with open(os.path.join(REPO_ROOT, config)) as f:
        cfg = yaml.safe_load(f)
    params = VehicleParams.load(cfg.get("vehicle_params", "vehicle_params_roboracer.yaml"))
    outer = np.load(os.path.join(REPO_ROOT, map_dir, "outer_line.npy"))[:, :2]
    inner = np.load(os.path.join(REPO_ROOT, map_dir, "inner_line.npy"))[:, :2]
    track = TrackModel(outer, inner)
    lateral = PurePursuit(params)

    safety = load_safety(config, {"speed_cap": 6.0, "speed_per_throttle": None, "max_throttle": 1.0,
                                  "max_throttle_no_speed": 1.0, "throttle_rate": 2.0})
    os.makedirs(out, exist_ok=True)
    video = os.path.join(out, "screen.mp4")
    cap.start(video)
    try:
        with SimSession(out, safety) as s:
            s.reset()
            st = s.state.read_latest()[1]
            reversed_ = track.align_direction(st.x, st.y, st.yaw)

            def hold(level: float):
                def run():
                    lateral.reset()
                    s.armed.set()
                    t0 = time.monotonic()
                    seq, t_seq, max_gap, hint = s.pose.latest_seq(), t0, 0.0, None
                    while time.monotonic() - t0 < HOLD_S:
                        state = s.state.read_latest()[1]
                        proj = track.project(state.x, state.y, hint=hint)
                        hint = proj.idx
                        steer = float(np.clip(lateral.steer(state, track, proj) / params.max_steering_angle, -1, 1))
                        s.ctrl.write(*signed_to_unit(steer, level))
                        now = time.monotonic()
                        if s.pose.latest_seq() != seq:
                            seq, t_seq = s.pose.latest_seq(), now
                        max_gap = max(max_gap, now - t_seq)
                        time.sleep(0.02)
                    ph = {"name": f"throttle_{level:.2f}", "t0": t0, "t1": time.monotonic(), "throttle": level,
                          "steering": None, "armed": True, "max_pose_gap": max_gap}
                    s.phases.append(ph)
                    return [ph]
                return run

            for level in LEVELS:
                s.reset()
                s.attempt(hold(level), max_gap=0.5)
                s.neutral("stop", 1.0)
            phases = s.phases
    finally:
        cap.stop(video)

    m = load_messages(out)
    rows, l_eff = [], []
    for level in LEVELS:
        ph = [p for p in phases if p["name"] == f"throttle_{level:.2f}"][-1]
        seg = segment(m, ph["t0"], ph["t1"])
        hit = np.flatnonzero(seg["collisions"] > seg["collisions"][0])
        end = hit[0] if len(hit) else len(seg["t_rx"])
        tt = seg["t_rx"][:end] - ph["t0"]
        steady = tt >= HOLD_S / 2
        v = seg["speed"][:end][steady]
        rows.append({"throttle": level, "speed_median": float(np.median(v)) if len(v) else math.nan,
                     "speed_std": float(np.std(v)) if len(v) else math.nan, "frames": int(len(v)),
                     "collided": bool(len(hit)), "attempts": ph.get("attempt", 1)})
        # steering -> curvature: frames with real steering and speed
        yaw = np.unwrap(seg["yaw"][:end])
        t = seg["t_rx"][:end]
        if len(t) > 4:
            yr = np.gradient(yaw, t)
            delta = seg["echo_steering"][:end] * params.max_steering_angle
            vv = seg["speed"][:end]
            sel = (np.abs(delta) > 0.08) & (vv > 0.3) & (np.abs(yr) > 0.05)
            l_eff.extend((vv[sel] * np.tan(delta[sel]) / yr[sel]).tolist())

    good = [r for r in rows if r["frames"] >= 5 and np.isfinite(r["speed_median"])]
    th = np.array([r["throttle"] for r in good])
    vs = np.array([r["speed_median"] for r in good])
    spt = float(th @ vs / (th @ th)) if len(good) else math.nan
    slope, intercept = (np.polyfit(th, vs, 1) if len(good) >= 2 else (math.nan, math.nan))
    resid = vs - spt * th
    l_arr = np.array([x for x in l_eff if 0.1 < x < 1.0])
    res = {
        "table": rows, "speed_per_throttle": spt, "fit_with_intercept": [float(slope), float(intercept)],
        "max_abs_residual_mps": float(np.max(np.abs(resid))) if len(resid) else math.nan,
        "effective_wheelbase_m": float(np.median(l_arr)) if len(l_arr) else math.nan,
        "effective_wheelbase_iqr_m": [float(np.percentile(l_arr, 25)), float(np.percentile(l_arr, 75))] if len(l_arr) else None,
        "wheelbase_samples": int(len(l_arr)), "configured_wheelbase_m": params.wheelbase,
        "track_reversed": bool(reversed_), "phases": phases, "safety_during_calibration": safety,
    }
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    _plot(res, os.path.join(out, "calibration.png"))
    return res


def _plot(res: dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in res["table"] if np.isfinite(r["speed_median"])]
    th = np.array([r["throttle"] for r in rows])
    v = np.array([r["speed_median"] for r in rows])
    e = np.array([r["speed_std"] for r in rows])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(th, v, yerr=e, fmt="o", color="#2a78d6", label="sim steady-state speed (median, ±1 sd)")
    xs = np.linspace(0, max(th) * 1.1, 10)
    ax.plot(xs, res["speed_per_throttle"] * xs, color="#eb6834",
            label=f"fit through 0: {res['speed_per_throttle']:.1f} m/s per unit throttle")
    ax.set_xlabel("throttle (signed units)")
    ax.set_ylabel("speed (m/s)")
    ax.set_title("AutoDRIVE RoboRacer: throttle -> steady-state speed", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
