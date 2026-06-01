#!/usr/bin/env python3
"""Scripted checks of the pilot_core <-> AutoDRIVE sim link against the live Unity sim
(docs/autodrive.md, validation step 2).

Runs pilot_core's own AutoDRIVEWorker and ActuatorWorker (the full safety chain) as
processes and writes scripted commands into ctrl_buf, as ControlWorker would; the sim
connects to the worker, the screen is recorded, and every phase is scored from the
worker's sim_messages.csv (what the sim reported, what was sent back).

    python scripts/autodrive_validate.py commands --out /workspaces/proof/runs/2bcd
    python scripts/autodrive_validate.py calibrate --out ... --map maps/autodrive_practice

commands: disarmed hold, armed constant command + echo, steering sign (left / right),
coast vs brake, reset. calibrate: constant throttle levels with pure_pursuit steering
on the mapped track -> steady-state speed table and a speed_per_throttle fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
import uuid

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import yaml  # noqa: E402

from shared_buffer import ControlBuffer, PoseBuffer, StateBuffer  # noqa: E402
from workers import ActuatorWorker, AutoDRIVEWorker  # noqa: E402
from workers.control_worker import signed_to_unit  # noqa: E402


class SimSession:
    """AutoDRIVEWorker + ActuatorWorker in their own processes, sim connected."""

    def __init__(self, out: str, safety: dict, port: int = 4567, rate_hz: float = 50.0) -> None:
        self.out = out
        os.makedirs(out, exist_ok=True)
        tag = uuid.uuid4().hex[:6]
        self.pose = PoseBuffer(slots=16, name=f"v_pose_{tag}")
        self.state = StateBuffer(slots=16, name=f"v_state_{tag}")
        self.ctrl = ControlBuffer(slots=16, name=f"v_ctrl_{tag}")
        self.act = ControlBuffer(slots=16, name=f"v_act_{tag}")
        self.armed = mp.Event()
        self.reset_event = mp.Event()
        self.laps, self.coll = mp.Value("i", 0), mp.Value("i", 0)
        self.rate_hz = rate_hz
        self.workers = [
            AutoDRIVEWorker(name="autodrive", host="127.0.0.1", port=port, pose_buffer=self.pose,
                            state_buffer=self.state, command_buffer=self.act,
                            command_timeout=safety.get("command_timeout", 0.15),
                            reset_event=self.reset_event, lap_counter=self.laps,
                            collision_counter=self.coll, log_dir=out),
            ActuatorWorker(name="actuator", input_buffer=self.ctrl, output_buffer=self.act,
                           armed_event=self.armed, state_buffer=self.state, safety=safety,
                           rate_hz=rate_hz),
        ]
        self.procs: list[mp.Process] = []
        self.phases: list[dict] = []
        self.t_start = time.monotonic()

    def __enter__(self) -> "SimSession":
        import autodrive_sim_ctl as sim_ctl

        for w in self.workers:
            p = mp.Process(target=w.run, name=w.name)
            p.start()
            self.procs.append(p)
        time.sleep(1.5)
        state = sim_ctl.set_state("connected")
        if state != "connected":
            sim_ctl.set_state("disconnected")
            state = sim_ctl.set_state("connected")
        deadline = time.monotonic() + 15
        while self.pose.latest_seq() < 5:
            if time.monotonic() > deadline:
                raise RuntimeError("no telemetry from the sim (is it running and connected?)")
            time.sleep(0.05)
        sim_ctl.close_menu()
        return self

    def __exit__(self, *exc) -> None:
        self.armed.clear()
        for w in self.workers:
            w.stop()
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        for b in (self.pose, self.state, self.ctrl, self.act):
            b.unlink()
        with open(os.path.join(self.out, "phases.json"), "w") as f:
            json.dump(self.phases, f, indent=1)

    # helpers

    def latest_pose(self) -> tuple[float, float, float]:
        _, pose, _, _ = self.pose.read_latest()
        return float(pose[0, 3]), float(pose[1, 3]), math.atan2(pose[1, 0], pose[0, 0])

    def drive(self, name: str, steering: float, throttle: float, seconds: float, armed: bool = True,
              **info) -> dict:
        """Write (steering, throttle) at the control rate for `seconds`."""
        if armed:
            self.armed.set()
        else:
            self.armed.clear()
        t0 = time.monotonic()
        period = 1.0 / self.rate_hz
        nxt = t0
        seq, t_seq, max_gap = self.pose.latest_seq(), t0, 0.0
        while time.monotonic() - t0 < seconds:
            self.ctrl.write(*signed_to_unit(steering, throttle))
            now = time.monotonic()
            if self.pose.latest_seq() != seq:
                seq, t_seq = self.pose.latest_seq(), now
            max_gap = max(max_gap, now - t_seq)
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
        # the sim freezes for seconds on this box (docs/autodrive.md): flag the phase
        ph = {"name": name, "t0": t0, "t1": time.monotonic(), "steering": steering, "throttle": throttle,
              "armed": armed, "max_pose_gap": max_gap, **info}
        self.phases.append(ph)
        return ph

    def attempt(self, run, max_gap: float = 0.35, tries: int = 5) -> list[dict]:
        """run() -> its phases; repeated (after a reset) while the sim stalls in any of them."""
        for k in range(tries):
            phases = run()
            worst = max(p["max_pose_gap"] for p in phases)
            for p in phases:
                p["attempt"] = k + 1
            if worst <= max_gap:
                return phases
            for p in phases:
                p["name"] += "_stalled"      # keep it in the log, out of the scoring
            print(f"sim stalled {worst:.2f} s during {phases[0]['name']}, retrying", flush=True)
            self.reset()
        return phases

    def neutral(self, name: str, seconds: float) -> dict:
        return self.drive(name, 0.0, 0.0, seconds, armed=False)

    def reset(self, settle: float = 2.0) -> dict:
        self.armed.clear()
        t0 = time.monotonic()
        self.reset_event.set()
        deadline = t0 + 5
        while self.reset_event.is_set():
            if time.monotonic() > deadline:
                raise RuntimeError("reset never went out")
            time.sleep(0.01)
        time.sleep(settle)
        ph = {"name": "reset", "t0": t0, "t1": time.monotonic(), "armed": False}
        self.phases.append(ph)
        return ph


def load_messages(out: str) -> dict[str, np.ndarray]:
    with open(os.path.join(out, "sim_messages.csv")) as f:
        rows = list(csv.DictReader(f))

    def col(k, default=np.nan):
        return np.array([float(r[k]) if r[k] not in ("", "inf") else (np.inf if r[k] == "inf" else default)
                         for r in rows])
    return {k: col(k) for k in rows[0]} if rows else {}


def segment(m: dict, t0: float, t1: float, new_only: bool = True) -> dict:
    sel = (m["t_rx"] >= t0) & (m["t_rx"] <= t1)
    if new_only:
        sel &= m["new_frame"] == 1
    return {k: v[sel] for k, v in m.items()}


def yaw_rate(s: dict) -> float:
    if len(s["t_rx"]) < 3:
        return float("nan")
    yaw = np.unwrap(s["yaw"])
    return float(np.polyfit(s["t_rx"], yaw, 1)[0])


def load_safety(config: str, overrides: dict) -> dict:
    with open(os.path.join(REPO_ROOT, config)) as f:
        safety = dict((yaml.safe_load(f) or {}).get("safety", {}))
    safety.update(overrides)
    return safety


# check sets


def run_commands(out: str, config: str) -> dict:
    """2b, 2c, 2d (reset): disarmed hold, armed echo, steering sign, coast vs brake, reset."""
    import autodrive_capture as cap

    safety = load_safety(config, {})
    video = os.path.join(out, "screen.mp4")
    os.makedirs(out, exist_ok=True)
    cap.start(video)
    res: dict = {"safety": safety}
    try:
        with SimSession(out, safety) as s:
            s.reset()
            start_pose = s.latest_pose()
            res["start_pose"] = start_pose
            s.neutral("settle", 1.0)
            # sim scale: throttle ~25 m/s per unit, so 0.04 ~ 1 m/s
            # 2b: disarmed, the controller asks for throttle -> the sim must get neutral
            s.attempt(lambda: [s.drive("disarmed_hold", 0.5, 0.04, 4.0, armed=False)], max_gap=1.0)
            # 2b: armed constant command -> V1 Throttle / V1 Steering echo it
            s.attempt(lambda: [s.drive("armed_constant", 0.2, 0.03, 2.5)])
            s.neutral("stop", 1.5)
            s.reset()
            # 2c: steering sign, both ways, from the start pose (short: walls ~1 m away)
            s.attempt(lambda: [s.drive("steer_left", 0.4, 0.03, 1.2)])
            s.neutral("stop", 1.5)
            s.reset()
            s.attempt(lambda: [s.drive("steer_right", -0.4, 0.03, 1.2)])
            s.neutral("stop", 1.5)
            s.reset()
            # 2c: negative throttle brakes: neutral vs negative throttle from the same speed
            s.attempt(lambda: [s.drive("accel_for_coast", 0.0, 0.06, 1.2), s.drive("coast", 0.0, 0.0, 1.5)])
            s.neutral("stop", 1.0)
            s.reset()
            s.attempt(lambda: [s.drive("accel_for_brake", 0.0, 0.06, 1.2), s.drive("brake", 0.0, -0.04, 1.5)])
            s.neutral("stop", 1.0)
            # 2d: reset returns the car to the start and zeroes the counters
            s.drive("drive_away", 0.0, 0.05, 2.0)
            s.neutral("stop", 1.0)
            pre = s.latest_pose()
            ph = s.reset(settle=2.5)
            post = s.latest_pose()
            ph.update(pre_pose=pre, post_pose=post, laps=s.laps.value, collisions=s.coll.value)
            s.neutral("end", 1.0)
            phases = s.phases
    finally:
        cap.stop(video)
    res["phases"] = phases
    res.update(score_commands(out, phases, start_pose))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    return res


def score_commands(out: str, phases: list[dict], start_pose) -> dict:
    m = load_messages(out)
    by = {}
    for ph in phases:
        by.setdefault(ph["name"], []).append(ph)
    checks = {}

    ph = by["disarmed_hold"][-1]
    seg = segment(m, ph["t0"], ph["t1"])
    disp = float(np.hypot(seg["x"] - seg["x"][0], seg["y"] - seg["y"][0]).max())
    sent = segment(m, ph["t0"], ph["t1"], new_only=False)
    checks["2b_disarmed"] = {
        "pass": bool(disp < 0.02 and np.all(sent["cmd_throttle"] == 0) and np.all(sent["cmd_steering"] == 0)
                     and np.all(seg["echo_throttle"] == 0)),
        "commanded": [ph["steering"], ph["throttle"]], "max_displacement_m": disp,
        "max_abs_sent_throttle": float(np.abs(sent["cmd_throttle"]).max()),
        "max_abs_echo_throttle": float(np.abs(seg["echo_throttle"]).max()), "frames": int(len(seg["t_rx"])),
    }

    ph = by["armed_constant"][-1]
    seg = segment(m, ph["t0"] + 0.5, ph["t1"])      # after the throttle slew settles
    sent = segment(m, ph["t0"] + 0.5, ph["t1"], new_only=False)
    echo_t, echo_s = seg["echo_throttle"], seg["echo_steering"]
    # each sim frame echoes the last command it got: compare to what was sent just before
    allm = segment(m, ph["t0"], ph["t1"] + 1.0, new_only=False)
    prev_sent_t, prev_sent_s = [], []
    for t in seg["t_rx"]:
        k = np.flatnonzero(allm["t_reply"] < t)
        prev_sent_t.append(allm["cmd_throttle"][k[-1]] if len(k) else np.nan)
        prev_sent_s.append(allm["cmd_steering"][k[-1]] if len(k) else np.nan)
    err_t = np.abs(echo_t - np.array(prev_sent_t))
    err_s = np.abs(echo_s - np.array(prev_sent_s))
    dist = float(np.hypot(seg["x"][-1] - seg["x"][0], seg["y"][-1] - seg["y"][0]))
    checks["2b_echo"] = {
        "pass": bool(np.nanmax(err_t) < 0.002 and np.nanmax(err_s) < 0.002
                     and np.allclose(sent["cmd_throttle"], ph["throttle"], atol=1e-3)
                     and np.allclose(sent["cmd_steering"], ph["steering"], atol=1e-3) and dist > 0.3),
        "commanded": [ph["steering"], ph["throttle"]],
        "sent_throttle_range": [float(sent["cmd_throttle"].min()), float(sent["cmd_throttle"].max())],
        "sent_steering_range": [float(sent["cmd_steering"].min()), float(sent["cmd_steering"].max())],
        "echo_throttle_range": [float(echo_t.min()), float(echo_t.max())],
        "echo_steering_range_normalised": [float(echo_s.min()), float(echo_s.max())],
        "max_echo_vs_sent_error": [float(np.nanmax(err_t)), float(np.nanmax(err_s))],
        "distance_m": dist, "speed_mean": float(np.nanmean(seg["speed"])), "frames": int(len(seg["t_rx"])),
    }

    rates = {}
    for name in ("steer_left", "steer_right"):
        ph = by[name][-1]
        seg = segment(m, ph["t0"] + 0.3, ph["t1"])
        hit = np.flatnonzero(seg["collisions"] > seg["collisions"][0])
        if len(hit):                      # the wall, not the steering, turns the car after that
            seg = {k: v[:hit[0]] for k, v in seg.items()}
        rates[name] = {"yaw_rate": yaw_rate(seg), "speed_mean": float(np.nanmean(seg["speed"])),
                       "echo_steering_mean": float(seg["echo_steering"].mean()),
                       "frames": int(len(seg["t_rx"])), "collided": bool(len(hit)),
                       "yaw_start_end": [float(seg["yaw"][0]), float(seg["yaw"][-1])]}
    checks["2c_steering_sign"] = {
        "pass": bool(rates["steer_left"]["yaw_rate"] > 0.05 and rates["steer_right"]["yaw_rate"] < -0.05),
        **rates,
    }

    def decel(name):
        ph = by[name][-1]
        seg = segment(m, ph["t0"], ph["t1"])
        v = seg["speed"]
        t = seg["t_rx"] - seg["t_rx"][0]
        # fit the slope from the start of the phase down to (about) standstill
        k = max(3, int(np.argmax(v < 0.05)) + 1 if np.any(v < 0.05) else len(v))
        k = min(k, len(v))
        a = float(-np.polyfit(t[:k], v[:k], 1)[0]) if k >= 3 else float("nan")
        return {"v_start": float(v[0]), "v_end": float(v[-1]), "v_min": float(v.min()), "decel_mps2": a,
                "time_to_stop_s": float(t[int(np.argmax(v < 0.05))]) if np.any(v < 0.05) else None,
                "echo_throttle_mean": float(seg["echo_throttle"].mean())}
    coast, brake = decel("coast"), decel("brake")
    # the stock sim's throttle is a speed set-point: neutral already brakes hard (no
    # coasting), so the check is that negative throttle stops the car and then reverses
    # it at the commanded set-point, not that it out-brakes neutral
    v_target = by["brake"][-1]["throttle"] * 25.0
    checks["2c_brake"] = {"pass": bool(brake["decel_mps2"] > 1.0 and brake["time_to_stop_s"] is not None
                                       and brake["echo_throttle_mean"] < 0 and brake["v_min"] < 0.5 * v_target),
                          "reverse_setpoint_mps": v_target, "coast": coast, "brake": brake,
                          "note": "neutral (throttle 0) stops the car as fast as negative throttle: set-point, not torque"}

    ph = by["reset"][-1]
    pre, post = ph["pre_pose"], ph["post_pose"]
    d_post = float(math.hypot(post[0] - start_pose[0], post[1] - start_pose[1]))
    d_pre = float(math.hypot(pre[0] - start_pose[0], pre[1] - start_pose[1]))
    dyaw = abs((post[2] - start_pose[2] + math.pi) % (2 * math.pi) - math.pi)
    checks["2d_reset"] = {"pass": bool(d_pre > 0.3 and d_post < 0.05 and dyaw < 0.02 and ph["collisions"] == 0
                                       and ph["laps"] == 0),
                          "distance_from_start_before_m": d_pre, "distance_from_start_after_m": d_post,
                          "yaw_error_after_rad": dyaw, "laps_after": ph["laps"], "collisions_after": ph["collisions"]}
    return {"checks": checks}


def _label(ph: dict) -> str:
    if ph["name"].startswith("reset"):
        return "RESET (V1 Reset held True)"
    arm = "ARMED" if ph.get("armed") else "DISARMED"
    name = ph["name"].replace("_", " ")
    return f"{name}: {arm} cmd steer {ph['steering']:+.2f} throttle {ph['throttle']:+.3f}"


def make_command_gifs(out: str, gif_dir: str) -> list[str]:
    """2b / 2c / 2d clips from a commands run's screen recording, captioned per phase."""
    import autodrive_capture as cap

    with open(os.path.join(out, "results.json")) as f:
        res = json.load(f)
    with open(os.path.join(out, "screen.mp4.json")) as f:
        t0v = json.load(f)["t0_monotonic"]
    phases = res["phases"]
    labels = [(p["t0"] - t0v, p["t1"] - t0v, _label(p)) for p in phases]

    def span(first: str, last: str, pad: float = 0.5):
        a = next(p for p in phases if p["name"] == first)["t0"] - t0v - pad
        b = [p for p in phases if p["name"] == last][-1]["t1"] - t0v + pad
        return max(a, 0.0), b

    os.makedirs(gif_dir, exist_ok=True)
    made = []
    clips = {
        # from the very start of the recording: the sim connecting, then the disarmed hold and echo
        "2b_connect_disarmed_armed_echo.gif": (0.0, span("disarmed_hold", "armed_constant")[1] + 1.5),
        "2c_steering_sign.gif": (span("steer_left", "steer_right")[0], span("steer_left", "steer_right")[1] + 1.5),
        "2c_neutral_vs_negative_throttle.gif": span("accel_for_coast", "brake", pad=1.0),
        "2d_reset.gif": (span("drive_away", "drive_away")[0], [p for p in phases if p["name"] == "reset"][-1]["t1"] - t0v),
    }
    for name, (a, b) in clips.items():
        made.append(cap.gif(os.path.join(out, "screen.mp4"), os.path.join(gif_dir, name), a, b,
                            width=640, fps=10, labels=labels))
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["commands", "rescore", "gifs", "calibrate"])
    ap.add_argument("--gif-dir", default="/workspaces/proof/gifs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="config_autodrive.yaml")
    ap.add_argument("--map", default="maps/autodrive_practice")
    a = ap.parse_args()
    mp.set_start_method("spawn", force=True)
    os.environ.setdefault("DISPLAY", ":99")
    if a.what == "commands":
        res = run_commands(a.out, a.config)
    elif a.what == "gifs":
        print("\n".join(make_command_gifs(a.out, a.gif_dir)))
        return 0
    elif a.what == "rescore":           # re-score a commands run from its logs
        with open(os.path.join(a.out, "results.json")) as f:
            res = json.load(f)
        res.update(score_commands(a.out, res["phases"], res["start_pose"]))
        with open(os.path.join(a.out, "results.json"), "w") as f:
            json.dump(res, f, indent=1, default=float)
    else:
        from autodrive_calibrate import run_calibration
        res = run_calibration(a.out, a.config, a.map)
    print(json.dumps(res.get("checks", res.get("table", {})), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
