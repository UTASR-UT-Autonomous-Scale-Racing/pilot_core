"""
A stand-in for the AutoDRIVE RoboRacer Unity sim, for tests: a websocket client that
speaks the stock 2026-iros client's dialect (scripts/autodrive_spike.py) to
AutoDRIVEWorker, drives the kinematic bicycle model (control/sim.py) with the commands
it gets back, ray-casts the 1081-beam LiDAR against the track walls and counts laps and
collisions on the track (oval_track() by default).

Like the real sim it emits telemetry on its own clock (rate_hz) and, with echo=True,
re-sends that snapshot once when the next command arrives; it only starts streaming after
the server's first 'Bridge'. Steering +1 turns left (CCW), throttle drives the
first-order speed model, 'V1 Reset': 'True' on two consecutive messages puts the car
back at the start (the real sim ignores a one-message pulse). As in the real sim, the
velocity is body-frame (x = signed speed) and the steering echo is the wheel angle in rad.
"""

from __future__ import annotations

import base64
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from control.sim import BicyclePlant, PlantConfig, oval_track
from control.track import TrackModel
from control.types import Command, VehicleParams
from workers.autodrive_worker import encode_lidar

N_BEAMS = 1081
ANGLES = -2.35619449 + np.arange(N_BEAMS) * 0.00436332
LIDAR_OFFSET = 0.2733          # m ahead of the rear axle
LIDAR_MAX = 10.0


def raycast(x: float, y: float, yaw: float, walls: List[np.ndarray], max_range: float = LIDAR_MAX) -> np.ndarray:
    """Ranges (inf = no return) of the 1081 beams from (x, y, yaw) to closed polylines."""
    p0 = np.vstack([w for w in walls])
    p1 = np.vstack([np.roll(w, -1, axis=0) for w in walls])
    e = p1 - p0
    ang = yaw + ANGLES
    d = np.column_stack([np.cos(ang), np.sin(ang)])
    o = np.array([x, y])
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    w_ = p0[None] - o
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w_[..., 0] * e[None, :, 1] - w_[..., 1] * e[None, :, 0]) / denom
        s = (w_[..., 0] * d[:, None, 1] - w_[..., 1] * d[:, None, 0]) / denom
    t = np.where((t > 0) & (s >= 0) & (s <= 1), t, np.inf)
    r = np.min(t, axis=1)
    return np.where(r <= max_range, r, np.inf)


def _jpeg(w: int = 192, h: int = 108) -> str:
    import cv2

    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 120, 40)       # BGR: blue sky over a grey floor
    img[h // 2:] = (90, 90, 90)
    ok, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode()


@dataclass
class FakeSimStats:
    commands: List[dict] = field(default_factory=list)   # every 'Bridge' received, as sent
    resets: int = 0
    sent: int = 0
    connected: bool = False


class FakeAutoDRIVE:
    def __init__(self, port: int, host: str = "127.0.0.1", rate_hz: float = 20.0, echo: bool = True,
                 track=None, params: Optional[VehicleParams] = None, plant: Optional[PlantConfig] = None,
                 start_s: float = 0.0, lidar: bool = True, camera: bool = True) -> None:
        self.url = f"ws://{host}:{port}/socket.io/?EIO=4&transport=websocket"
        self.rate_hz, self.echo = rate_hz, echo
        self.outer, self.inner, _ = track if track is not None else oval_track()
        self.track = TrackModel(self.outer, self.inner)
        self.params = params or VehicleParams.load("vehicle_params_roboracer.yaml")
        # measured on the stock sim: ~25 m/s per unit throttle, settles in ~0.5 s, and
        # neutral stops the car in ~0.4 s from 1.5 m/s
        self.plant_cfg = plant or PlantConfig(speed_per_throttle=25.0, tau=0.15, coast_decel=3.5,
                                              pose_latency=0.0, pos_noise=0.0, yaw_noise=0.0)
        self.lidar, self.camera = lidar, camera
        # walls for the ray-cast, decimated to ~0.25 m segments: 5x cheaper, same scan
        step = max(1, int(round(0.25 / max(self.track.length / len(self.outer), 1e-3))))
        self._walls = [self.outer[::step], self.inner[::step]]
        self._start_idx = int(start_s / self.track.length * len(self.track.points)) % len(self.track.points)
        self.stats = FakeSimStats()
        self.speed_override: Optional[float] = None   # report this speed instead of the plant's
        self.paused = False                            # stop emitting telemetry (stalled sim)
        self.cmd = Command(0.0, 0.0)
        self.lap_count = self.collisions = 0
        self.lap_times: List[float] = []
        self._reset_run = 0      # consecutive 'True' resets received
        self._jpeg = _jpeg() if camera else ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._place_at_start()
        self._thread = threading.Thread(target=self._run, daemon=True, name="fake-autodrive")

    # sim state

    def _place_at_start(self) -> None:
        p = self.track.points[self._start_idx]
        self.car = BicyclePlant(self.params, self.plant_cfg, float(p[0]), float(p[1]),
                                float(self.track.heading[self._start_idx]))
        self.t0 = self.lap_start = time.monotonic()
        self._s_prev, self._s_total, self._hint = None, 0.0, None
        self._off_track = False
        self.yaw_rate = 0.0

    @property
    def sim_time(self) -> float:
        return time.monotonic() - self.t0

    def _physics(self, t: float, dt: float) -> None:
        yaw0 = self.car.yaw
        self.car.step(self.cmd, t, dt)
        self.yaw_rate = (self.car.yaw - yaw0) / dt
        pr = self.track.project(self.car.x, self.car.y, hint=self._hint)
        self._hint = pr.idx
        if self._s_prev is not None:
            self._s_total += (pr.s - self._s_prev + self.track.length / 2) % self.track.length - self.track.length / 2
            if self._s_total >= self.track.length * (self.lap_count + 1):
                self.lap_count += 1
                self.lap_times.append(time.monotonic() - self.lap_start)
                self.lap_start = time.monotonic()
        self._s_prev = pr.s
        off = abs(pr.lateral_error) > self.track.half_width[pr.idx] - self.params.width / 2
        if off and not self._off_track:
            self.collisions += 1
        self._off_track = off

    def telemetry(self) -> dict:
        c = self.car
        v = c.v if self.speed_override is None else self.speed_override
        q = (0.0, 0.0, math.sin(c.yaw / 2), math.cos(c.yaw / 2))
        msg = {
            "V1 Throttle": f"{self.cmd.throttle:.4f}", "V1 Steering": f"{self.cmd.steering * 0.5236:.4f}",
            "V1 Encoder Ticks": "0 0", "V1 Encoder Angles": "0.0000 0.0000",
            "V1 Position": f"{c.x:.4f} {c.y:.4f} 0.0592",
            "V1 Orientation Quaternion": " ".join(f"{v:.4f}" for v in q),
            "V1 Orientation Euler Angles": f"0.0000 0.0000 {c.yaw % (2 * math.pi):.4f}",
            "V1 Linear Velocity": f"{v:.4f} 0.0000 0.0000",
            "V1 Angular Velocity": f"0.0000 0.0000 {self.yaw_rate:.4f}",
            "V1 Linear Acceleration": "0.0000 0.0000 0.0000",
            "V1 LIDAR Scan Rate": "40.0000",
            "V1 Lap Count": str(self.lap_count),
            "V1 Lap Time": f"{self.sim_time:.4f}",
            "V1 Last Lap Time": f"{self.lap_times[-1]:.4f}" if self.lap_times else "Infinity",
            "V1 Best Lap Time": f"{min(self.lap_times):.4f}" if self.lap_times else "Infinity",
            "V1 Collisions": str(self.collisions),
        }
        if self.lidar:
            lx = c.x + LIDAR_OFFSET * math.cos(c.yaw)
            ly = c.y + LIDAR_OFFSET * math.sin(c.yaw)
            msg["V1 LIDAR Range Array"] = encode_lidar(raycast(lx, ly, c.yaw, self._walls))
        if self.camera:
            msg["V1 Front Camera Image"] = self._jpeg
        return msg

    # link

    def start(self) -> "FakeAutoDRIVE":
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

    def _send(self, ws, msg: dict) -> None:
        ws.send("42" + json.dumps(["Bridge", msg]))
        self.stats.sent += 1

    def _on_command(self, ws, data: dict) -> None:
        with self._lock:
            self.stats.commands.append(dict(data))
            if data.get("V1 Reset") == "True":
                self._reset_run += 1
                if self._reset_run == 2:
                    self.stats.resets += 1
                    self.lap_count = self.collisions = 0
                    self.lap_times = []
                    self._place_at_start()
                self.cmd = Command(0.0, 0.0)
            else:
                self._reset_run = 0
                self.cmd = Command(float(data["V1 Steering"]), float(data["V1 Throttle"]))

    def _run(self) -> None:
        from websockets.sync.client import connect

        while not self._stop.is_set():          # retry until the server is up
            try:
                with connect(self.url, open_timeout=1.0, max_size=None, compression=None,
                             ping_interval=None) as ws:
                    try:
                        self._session(ws)
                    except Exception:     # server went away (pilot_core stopped): apply what arrived
                        self._drain(ws)
                    finally:
                        self.stats.connected = False
                return
            except (OSError, TimeoutError):
                time.sleep(0.1)
            except Exception:
                time.sleep(0.1)

    def _drain(self, ws) -> None:
        while True:
            try:
                raw = ws.recv(timeout=0)
            except Exception:
                return
            if isinstance(raw, str) and raw.startswith('42["Bridge"'):
                self._on_command(ws, json.loads(raw[2:])[1])

    def _session(self, ws) -> None:
        streaming = False
        last, echoed = None, True
        dt = 0.005
        t_next_tx = t_next_ping = time.monotonic()
        t_phys = time.monotonic()
        self.stats.connected = True
        while not self._stop.is_set():
            # incoming: open, namespace connect, pong, commands
            try:
                raw = ws.recv(timeout=0.0005)
            except TimeoutError:
                raw = None
            except Exception:
                return
            if isinstance(raw, str) and raw.startswith('42["Bridge"'):
                data = json.loads(raw[2:])[1]
                self._on_command(ws, data)
                if streaming and self.echo and last is not None and not echoed:
                    # the real sim re-sends its last snapshot (same sim clock) once per tick
                    self._send(ws, last)
                    echoed = True
                streaming = True
            now = time.monotonic()
            with self._lock:
                while t_phys < now:
                    self._physics(t_phys, dt)
                    t_phys += dt
            if streaming and now >= t_next_tx and not self.paused:
                t_next_tx = now + 1.0 / self.rate_hz
                with self._lock:
                    last = self.telemetry()
                self._send(ws, last)
                echoed = False
            if now >= t_next_ping:
                t_next_ping = now + 1.0
                ws.send("2")
