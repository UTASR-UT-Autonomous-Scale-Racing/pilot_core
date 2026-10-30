#!/usr/bin/env python3
"""Check the rear-of-car IMU (the Arduino's ICM-20948 over I2C) before enabling rear_imu.

    python scripts/check_rear_imu.py                      # on the car: live, follow the prompts
    python scripts/check_rear_imu.py --recording <dir>    # a recording with rear_imu (+ raw_imu)

Live, three timed phases (the car on the floor, motors off):
  1. still (3 s): poll rate, new samples per second (the Jetson polls the latest sample, so
     repeats are normal), interval jitter, the gyro's residual bias and the accel at rest;
  2. turn the car LEFT (counter-clockwise seen from above) by about 90 deg in place (5 s):
     the gyro axis that integrates the turn is the car's up axis, and the angle checks the
     scale;
  3. push the car straight FORWARD and stop it (4 s): the accel axis of the push is the
     car's forward axis.
The firmware calibrates accel to (0, 0, 1 g) at rest, whatever the mounting, so the
mounting can only be seen from motion. Prints the rear_imu block for config.yaml.

A recording (scripts/vio_bench/make_record_config.py with rear_imu enabled) gives the
rates and the rear IMU's delay against the OAK-D's IMU (device-timestamped with the
frames) by cross-correlating the angular-rate magnitudes, which don't depend on either
IMU's axes: delay_s for the config.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def stream_stats(t: np.ndarray, new: np.ndarray) -> dict:
    """t: poll times (s), new: True where the poll returned a new sample."""
    dur = max(float(t[-1] - t[0]), 1e-9)
    tn = t[new]
    gaps = np.diff(tn) if len(tn) > 1 else np.array([np.nan])
    return {"polls_hz": (len(t) - 1) / dur, "new_hz": (len(tn) - 1) / dur if len(tn) > 1 else 0.0,
            "repeat_pct": 100.0 * (1.0 - new.mean()), "interval_ms": 1e3 * float(np.median(gaps)),
            "interval_p99_ms": 1e3 * float(np.percentile(gaps, 99))}


def axis_from_turn(t: np.ndarray, gyro: np.ndarray) -> tuple:
    """(axis, sign, degrees): the sensor axis that integrates a left (CCW) turn, its sign
    (+1 if positive for CCW) and the integrated angle."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed it
    ang = np.degrees(trapezoid(gyro, t, axis=0))
    k = int(np.argmax(np.abs(ang)))
    return k, int(np.sign(ang[k]) or 1), float(abs(ang[k]))


def axis_from_push(acc: np.ndarray, up_axis: int, thresh: float = 0.4) -> tuple:
    """(axis, sign): the horizontal sensor axis of a forward push, from its first clear
    acceleration (accel minus its median at rest)."""
    a = acc - np.median(acc, axis=0)
    a[:, up_axis] = 0.0
    k = int(np.argmax(np.abs(a).max(axis=0)))
    first = np.nonzero(np.abs(a[:, k]) > thresh)[0]
    if not len(first):
        raise ValueError("no clear push in the accel (more than 0.4 m/s^2)")
    return k, int(np.sign(a[first[0], k]))


def rotation_from(up: tuple, fwd: tuple) -> np.ndarray:
    """Rotation taking sensor vectors to the car frame (x forward, y left, z up)."""
    if up[0] == fwd[0]:
        raise ValueError("the turn and the push found the same sensor axis")
    ez = np.zeros(3)
    ez[up[0]] = up[1]
    ex = np.zeros(3)
    ex[fwd[0]] = fwd[1]
    ey = np.cross(ez, ex)
    return np.vstack([ex, ey, ez])


def time_offset(t_ref: np.ndarray, w_ref: np.ndarray, t: np.ndarray, w: np.ndarray, max_s: float = 0.1) -> float:
    """s by which the second stream lags the first, from |angular rate| cross-correlation."""
    a_t0, a_t1 = max(t_ref[0], t[0]) + max_s, min(t_ref[-1], t[-1]) - max_s
    tg = np.arange(a_t0, a_t1, 0.001)
    a = np.interp(tg, t_ref, np.linalg.norm(w_ref, axis=1))
    a = a - a.mean()
    wn = np.linalg.norm(w, axis=1)
    lags = np.arange(-max_s, max_s, 0.0005)
    score = []
    for lag in lags:
        b = np.interp(tg + lag, t, wn)
        score.append(float(a @ (b - b.mean())))
    return float(lags[int(np.argmax(score))])


def config_block(R: np.ndarray, delay: float = 0.0) -> str:
    rows = ", ".join("[" + ", ".join(f"{int(v)}" for v in r) + "]" for r in np.round(R).astype(int))
    return ("rear_imu:\n  enabled: true\n"
            f"  rotation: [{rows}]\n"
            "  lever_m: [0.0, 0.0]           # m, measure: the IMU from the camera (x forward, y left)\n"
            f"  delay_s: {delay:.4f}\n  estimator: {{}}\n")


def _poll(bus, addr, seconds: float, hz: float):
    import smbus2

    from workers.imu_worker import I2C_FRAME_SIZE, parse_i2c_raw

    t, s, new, last = [], [], [], None
    end, nxt = time.monotonic() + seconds, time.monotonic()
    while time.monotonic() < end:
        nxt += 1.0 / hz
        msg = smbus2.i2c_msg.read(addr, I2C_FRAME_SIZE)
        try:
            bus.i2c_rdwr(msg)
        except OSError:
            continue
        data = bytes(msg)
        x = parse_i2c_raw(data)
        if x is not None:
            t.append(time.monotonic())
            s.append(x)
            new.append(data[16:40] != last)
            last = data[16:40]
        time.sleep(max(0.0, nxt - time.monotonic()))
    return np.array(t), np.array(s), np.array(new, bool)


def live(args) -> int:
    try:
        import smbus2
    except ImportError:
        print("smbus2 is not installed: run this on the car", file=sys.stderr)
        return 1
    import yaml

    with open(os.path.join(REPO_ROOT, args.config)) as f:
        cfg = yaml.safe_load(f) or {}
    addr, busn = cfg.get("i2c_address"), cfg.get("i2c_bus", 1)
    if addr is None:
        print(f"{args.config} has no i2c_address", file=sys.stderr)
        return 1
    bus = smbus2.SMBus(int(busn))
    print("1/3  keep the car still ...")
    t, s, new = _poll(bus, addr, 3.0, args.poll_hz)
    if len(t) < 10:
        print("no IMU frames: is the Arduino running (and start.py stopped)?", file=sys.stderr)
        return 1
    st = stream_stats(t, new)
    g = s[new]
    print(f"     polls {st['polls_hz']:.0f} Hz, new samples {st['new_hz']:.0f} Hz ({st['repeat_pct']:.0f} % repeats), "
          f"interval {st['interval_ms']:.1f} ms (p99 {st['interval_p99_ms']:.1f})")
    print(f"     gyro at rest {np.degrees(g[:, 3:].mean(0)).round(3)} deg/s (noise {np.degrees(g[:, 3:].std(0)).round(3)}), "
          f"accel {g[:, :3].mean(0).round(3)} m/s^2")
    print("2/3  turn the car LEFT (counter-clockwise from above) ~90 deg in place, now (5 s) ...")
    t, s, new = _poll(bus, addr, 5.0, args.poll_hz)
    up = axis_from_turn(t[new], s[new][:, 3:])
    print(f"     sensor axis {'xyz'[up[0]]} ({'+' if up[1] > 0 else '-'}) turned {up[2]:.0f} deg")
    print("3/3  push the car straight FORWARD and stop it, now (4 s) ...")
    t, s, new = _poll(bus, addr, 4.0, args.poll_hz)
    fwd = axis_from_push(s[new][:, :3], up[0])
    print(f"     forward is sensor axis {'xyz'[fwd[0]]} ({'+' if fwd[1] > 0 else '-'})")
    R = rotation_from(up[:2], fwd)
    print("\nconfig.yaml (measure lever_m; delay_s from --recording):\n")
    print(config_block(R))
    return 0


def recording(args) -> int:
    from read_recording import iter_records, list_files

    names = {rf.buffer for rf in list_files(args.recording)}
    if "rear_imu" not in names:
        print(f"{args.recording} has no rear_imu recording (rear_imu.enabled + recording)", file=sys.stderr)
        return 1
    rear = np.array([[r.produced_ts, *r.value.tolist()] for r in iter_records(args.recording, "rear_imu")])
    rear = rear[np.argsort(rear[:, 0])]
    gaps = np.diff(rear[:, 0])
    print(f"rear IMU: {len(rear)} samples, {1 / np.median(gaps):.0f} Hz, p99 interval {1e3 * np.percentile(gaps, 99):.1f} ms, "
          f"{int(np.sum(gaps > 0.05))} gaps > 50 ms")
    if "raw_imu" not in names:
        print("no raw_imu (OAK-D) recording: no delay estimate")
        return 0
    cam = np.array([[r.produced_ts, *r.value.tolist()] for r in iter_records(args.recording, "raw_imu")])
    cam = cam[np.argsort(cam[:, 0])]
    d = time_offset(cam[:, 0], cam[:, 4:7], rear[:, 0], rear[:, 4:7])
    print(f"the rear IMU lags the OAK-D IMU by {1e3 * d:.1f} ms: delay_s: {d:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--recording", help="check a recording instead of the live IMU")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--poll-hz", type=float, default=100.0)
    args = ap.parse_args()
    return recording(args) if args.recording else live(args)


if __name__ == "__main__":
    sys.exit(main())
