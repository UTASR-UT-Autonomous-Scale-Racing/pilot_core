"""
Direct link to the AutoDRIVE RoboRacer Unity sim: pilot_core is the server the sim
connects to (ws://<host>:4567/socket.io/), telemetry in on event 'Bridge', the latest
post-safety command (ActuatorWorker's output buffer) back on every message.

The stock 2026-iros client is an old Socket.IO component on an Engine.IO v4 URL: it
pings from the client side ("2" -> we answer "3"), never sends a namespace CONNECT and
only starts streaming after the server has sent a first 'Bridge'. It emits on its own
clock (~10-15 Hz on the vast box) and also echoes its current state whenever it gets a
command, so every message is answered but only new sim frames (sim clock advanced) go
into the buffers. Speaks that subset directly over websockets (scripts/autodrive_spike.py).
"""

from __future__ import annotations

import base64
import csv
import json
import math
import os
import socket
import threading
import time
from typing import Any, Optional

import numpy as np

from .control_worker import unit_to_signed
from .sim_decoder import SimDecoder

NEUTRAL = (0.0, 0.0)
SIM_MAX_STEER = 0.5236   # rad: the sim maps steering +-1 to +-30 deg and echoes the angle
_OPEN = '0{"sid":"pilot_core","upgrades":[],"pingInterval":25000,"pingTimeout":20000}'
_CSV_FIELDS = ("t_rx", "t_reply", "sim_time", "new_frame", "x", "y", "yaw", "speed",
               "echo_throttle", "echo_steering", "cmd_throttle", "cmd_steering", "cmd_age",
               "reset", "lap_count", "collisions", "sim_yaw_euler")


def _vec(s: str) -> list[float]:
    return [float(v) for v in s.split()]


def _num(s: Any, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def decode_lidar(b64: str) -> np.ndarray:
    """'V1 LIDAR Range Array': base64 -> gzip -> newline-separated ranges (inf = no return)."""
    import gzip

    text = gzip.decompress(base64.b64decode(b64)).decode()
    return np.array(text.split(), dtype=np.float32)


def encode_lidar(ranges) -> str:
    """Inverse of decode_lidar (the fake sim uses it)."""
    import gzip

    text = "\n".join("Infinity" if not math.isfinite(r) else f"{r:.4f}" for r in ranges)
    return base64.b64encode(gzip.compress(text.encode())).decode()


def autodrive_to_internal(msg: dict) -> dict:
    """AutoDRIVE 'Bridge' telemetry (string values, 'V1 ...' keys) -> the dict the
    SimDecoder decoders read. Missing keys are left out; image data stays base64."""
    out: dict[str, Any] = {}
    if "V1 Position" in msg:
        out["position"] = _vec(msg["V1 Position"])
    if "V1 Orientation Quaternion" in msg:
        out["orientation_quat"] = _vec(msg["V1 Orientation Quaternion"])       # x y z w
    if "V1 Orientation Euler Angles" in msg:
        out["euler"] = _vec(msg["V1 Orientation Euler Angles"])               # sim's own, rad
    if "V1 Linear Acceleration" in msg:
        out["linear_acceleration"] = _vec(msg["V1 Linear Acceleration"])
    if "V1 Angular Velocity" in msg:
        out["angular_velocity"] = _vec(msg["V1 Angular Velocity"])
    if "V1 Linear Velocity" in msg:
        # body frame (x forward), so x is the signed speed: negative when reversing
        vel = _vec(msg["V1 Linear Velocity"])
        out["linear_velocity"] = vel
        out["speed"] = float(vel[0])
    if msg.get("V1 LIDAR Range Array"):
        out["lidar"] = decode_lidar(msg["V1 LIDAR Range Array"]).tolist()   # the decoder truth-tests it
    if msg.get("V1 Front Camera Image"):
        out["front_camera"] = {"data": msg["V1 Front Camera Image"], "encoding": "jpeg"}
    # Joshua's build: depth camera + checkpoint counter (key names unconfirmed, no build yet)
    for key in ("V1 Depth Camera Image", "V1 Front Depth Image"):
        if msg.get(key):
            out["depth_camera"] = {"data": msg[key], "encoding": "png"}
    if "V1 Steering" in msg:
        # the echo is the applied wheel angle in rad; "steering" is normalised like the command
        out["steering_angle"] = _num(msg["V1 Steering"])
        out["steering"] = out["steering_angle"] / SIM_MAX_STEER
    for src, dst in (("V1 Throttle", "throttle"),
                     ("V1 Lap Time", "lap_time"), ("V1 Last Lap Time", "last_lap_time"),
                     ("V1 Best Lap Time", "best_lap_time")):
        if src in msg:
            out[dst] = _num(msg[src], float("inf"))
    for src, dst in (("V1 Lap Count", "lap_count"), ("V1 Collisions", "collision_count"),
                     ("V1 Checkpoint Count", "checkpoint_count"), ("V1 Checkpoints", "checkpoint_count")):
        if src in msg:
            out[dst] = int(_num(msg[src]))
    return out


def command_message(throttle: float, steering: float, reset: bool = False) -> str:
    """Server -> sim 'Bridge' frame. Signed units, steering +1 = left."""
    payload = {"V1 Throttle": f"{throttle:.4f}", "V1 Steering": f"{steering:.4f}",
               "V1 Reset": "True" if reset else "False"}
    return "42" + json.dumps(["Bridge", payload])


def yaw_from_quat(q) -> float:
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class AutoDRIVEWorker(SimDecoder):
    """AutoDRIVE sim endpoint: sensors into the buffers, supervised commands out.

    Config kwargs (plus the SimDecoder buffers: frame_buffer, imu_buffer,
    pose_buffer, state_buffer, lidar_buffer, depth_*, counters):
      host, port          listen address, default 0.0.0.0:4567
      command_buffer      ControlBuffer written by ActuatorWorker ([0, 1] units, post-safety)
      command_timeout     s; an older (or missing) command goes out as neutral, default 0.15
      reset_event         set -> replies carry 'V1 Reset': 'True' (and neutral) for reset_hold_s,
                          then the event is cleared; the sim ignores a one-message pulse
      reset_hold_s        default 0.4
      log_dir             sim_messages.csv + sim_events.jsonl go here (None = no logs)
      sim_time_dedupe     only write buffers when the sim clock moved, default True
      sim_depth           {camera: sim_camera/camera.yaml, scale: 2}: RGB-D for ORB-SLAM3 from a
                          camera without depth (workers/lidar_depth.py): the frame is upscaled,
                          the car body hidden, depth comes from the same message's LiDAR scan
    """

    def __init__(self, name: str = "autodrive", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._server = None
        self._server_thread: Optional[threading.Thread] = None
        self._ws = None
        self._ws_lock: Optional[threading.Lock] = None   # created in setup(): spawn pickles the worker
        self._cmd_buf = None
        self._cmd_timeout = 0.15
        self._reset_event = None
        self._closing = False
        self._last_sim_time: Optional[float] = None
        self._prev_lap_count: Optional[int] = None
        self._prev_collisions: Optional[int] = None
        self._csv = None
        self._csv_fp = None
        self._events_fp = None
        self._msgs = 0
        self._frames = 0
        self._connections = 0
        self._last_rx = 0.0

    # setup / teardown

    def setup(self) -> None:
        from websockets.sync.server import serve

        cfg = self.config
        self._ws_lock = threading.Lock()
        self._frame_buf = cfg.get("frame_buffer")
        self._imu_buf = cfg.get("imu_buffer")
        self._pose_buf = cfg.get("pose_buffer")
        self._state_buf = cfg.get("state_buffer")
        self._lidar_buf = cfg.get("lidar_buffer")
        self._depth_max_meters = float(cfg.get("depth_max_meters", 15.0))
        scale = cfg.get("depth_raw_scale")
        self._depth_raw_scale = float(scale) if scale is not None else None
        self._no_msg_warn_s = float(cfg.get("no_message_warn_s", 2.0))
        self._collision_counter = cfg.get("collision_counter")
        self._lap_counter = cfg.get("lap_counter")
        self._checkpoint_counter = cfg.get("checkpoint_counter")
        self._cmd_buf = cfg.get("command_buffer")
        self._cmd_timeout = float(cfg.get("command_timeout", 0.15))
        self._reset_event = cfg.get("reset_event")
        self._reset_hold = float(cfg.get("reset_hold_s", 0.4))
        self._reset_started: Optional[float] = None
        self._dedupe = bool(cfg.get("sim_time_dedupe", True))
        self._depth_fn = None
        self._frame_errors = 0
        sd = cfg.get("sim_depth")
        if sd and self._frame_buf is not None:
            from .lidar_depth import CameraModel, LidarDepth

            cam = CameraModel.from_yaml(sd.get("camera", "sim_camera/camera.yaml"), scale=float(sd.get("scale", 2.0)))
            self._depth_fn = LidarDepth(cam)
            self._depth_clahe = bool(sd.get("clahe", True))
            self.logger.info(f"RGB-D from LiDAR: {cam.size[0]}x{cam.size[1]}, vfov {cam.vfov_deg:.1f} deg, "
                             f"car mask {'on' if self._depth_fn.mask is not None else 'off'}")

        log_dir = cfg.get("log_dir")
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._csv_fp = open(os.path.join(log_dir, "sim_messages.csv"), "a", newline="")
            self._csv = csv.writer(self._csv_fp)
            if self._csv_fp.tell() == 0:
                self._csv.writerow(_CSV_FIELDS)
            self._events_fp = open(os.path.join(log_dir, "sim_events.jsonl"), "a")

        host, port = cfg.get("host", "0.0.0.0"), int(cfg.get("port", 4567))
        self._server = serve(self._handle, host, port, max_size=None, compression=None,
                             ping_interval=None)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                               name="autodrive-ws")
        self._server_thread.start()
        self._last_msg_time = self._last_report_time = time.monotonic()
        self.logger.info(
            f"AutoDRIVE server listening on {host}:{port} (command timeout "
            f"{self._cmd_timeout * 1000:.0f} ms, logs={'on' if log_dir else 'off'}) — "
            f"connect the sim (HUD 'Disconnected' toggle, or scripts/autodrive_sim_ctl.py connect)"
        )

    def step(self) -> None:
        now = time.monotonic()
        self._check_silence(now)
        if now - self._last_report_time >= 10.0:
            el = now - self._last_report_time
            self.logger.info(f"sim link: {self._msgs / el:.1f} msg/s, {self._frames / el:.1f} new frames/s")
            self._msgs = self._frames = 0
            self._last_report_time = now
        time.sleep(0.05)

    def _check_silence(self, now: float) -> None:
        if self._connections == 0:
            return    # nothing to warn about until the sim has connected once
        super()._check_silence(now)

    def teardown(self) -> None:
        # last word to the sim is neutral, then stop answering
        if self._ws_lock is None:      # setup() never ran
            return
        with self._ws_lock:
            # flag and grab the socket together: the handler drops _ws as soon as it sees _closing
            self._closing = True
            ws = self._ws
            if ws is not None:
                try:
                    ws.send(command_message(*NEUTRAL))
                    self.logger.info("Sent neutral to the sim")
                except Exception:
                    pass
                try:
                    ws.close()
                except Exception:
                    pass
        if self._server is not None:
            self._server.shutdown()
        if self._server_thread is not None:
            self._server_thread.join(timeout=2)
        for fp in (self._csv_fp, self._events_fp):
            if fp is not None:
                fp.close()
        self.logger.info("AutoDRIVEWorker shut down")

    # websocket side (server thread per connection)

    def _handle(self, ws) -> None:
        try:
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        if self._closing:
            ws.close()
            return
        with self._ws_lock:
            if self._ws is not None:
                self.logger.warning("Second sim connection — the newest one wins")
            self._ws = ws
        self._connections += 1
        self.logger.info(f"Sim connected from {ws.remote_address} ({ws.request.path})")
        try:
            ws.send(_OPEN)
            ws.send("40")
            self._reply(ws, time.monotonic(), None)   # the sim waits for a first 'Bridge'
            for raw in ws:
                if self._closing:
                    break
                if raw == "2":            # client ping (Engine.IO v3 style)
                    ws.send("3")
                    continue
                if not isinstance(raw, str) or not raw.startswith('42["Bridge"'):
                    continue
                self._on_bridge(ws, raw)
        except Exception as e:
            if not self._closing:
                self.logger.warning(f"Sim connection ended: {e!r}")
        finally:
            with self._ws_lock:
                if self._ws is ws:
                    self._ws = None
            self.logger.info("Sim disconnected")

    def _on_bridge(self, ws, raw: str) -> None:
        t_rx = time.monotonic()
        if self._silent:
            self.logger.info(f"Sim stream back after {t_rx - self._last_msg_time:.1f} s")
            self._silent = False
        self._last_msg_time = self._last_rx = t_rx
        self._msgs += 1
        try:
            msg = json.loads(raw[2:])[1] or {}
        except (ValueError, IndexError) as e:
            self.logger.warning(f"Bad 'Bridge' frame: {e}")
            self._reply(ws, t_rx, None)
            return
        sim_time = _num(msg.get("V1 Lap Time"), float("nan"))
        new_frame = not self._dedupe or self._last_sim_time is None or sim_time != self._last_sim_time
        # reply first: the command must not wait on image decoding
        sent = self._reply(ws, t_rx, msg)
        data = autodrive_to_internal(msg)
        if new_frame:
            self._last_sim_time = sim_time
            self._frames += 1
            # the camera on its own: a frame that fails must not hold back the pose
            # (and so the controller) or the other sensors
            try:
                if self._frame_buf is not None:
                    if self._depth_fn is not None and data.get("front_camera") and data.get("lidar"):
                        self._write_rgbd(data)
                    else:
                        self._process_frame_data(data)
            except Exception as e:
                self._frame_errors += 1
                if self._frame_errors in (1, 10, 100) or self._frame_errors % 1000 == 0:
                    self.logger.error(f"Error writing the sim frame ({self._frame_errors} so far): {e}")
            try:
                self._process_imu_data(data)
                self._process_pose_data(data)
                self._process_lidar_data(data)
                self._process_telemetry_data(data)
            except Exception as e:
                self.logger.error(f"Error writing sim data to buffers: {e}")
            self._log_events(data, t_rx)
        self._log_message(data, sim_time, new_frame, *sent)

    def _write_rgbd(self, data: dict) -> None:
        """Sim camera + LiDAR depth -> frame_buf (see workers/lidar_depth.py)."""
        import cv2

        from .lidar_depth import mask_rgb

        raw = np.frombuffer(base64.b64decode(data["front_camera"]["data"]), np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            return
        w, h = self._depth_fn.cam.size
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = mask_rgb(rgb, self._depth_fn.mask, clahe=self._depth_clahe)
        depth = self._depth_fn(np.asarray(data["lidar"], dtype=np.float32))
        self._frame_buf.write(np.ascontiguousarray(rgb), depth)

    def _command(self, now: float) -> tuple[float, float, float]:
        """(throttle, steering, age) of the latest supervised command; neutral if stale."""
        if self._cmd_buf is None:
            return 0.0, 0.0, float("inf")
        res = self._cmd_buf.read_latest()
        if res is None:
            return 0.0, 0.0, float("inf")
        header, steer_unit, throttle_unit = res
        age = now - header.produced_ts
        if not (-0.01 <= age <= self._cmd_timeout):
            return 0.0, 0.0, age
        steering, throttle = unit_to_signed(steer_unit, throttle_unit)
        if not (math.isfinite(steering) and math.isfinite(throttle)):
            return 0.0, 0.0, age
        return float(np.clip(throttle, -1, 1)), float(np.clip(steering, -1, 1)), age

    def _reply(self, ws, t_rx: float, msg: Optional[dict]) -> tuple:
        now = time.monotonic()
        throttle, steering, age = (0.0, 0.0, float("inf")) if self._closing else self._command(now)
        reset = False
        if self._reset_event is not None and self._reset_event.is_set():
            reset = True
            throttle, steering = NEUTRAL
            if self._reset_started is None:
                self._reset_started = now
                self.logger.info("Resetting the sim")
                self._event({"t": now, "event": "reset_sent"})
        with self._ws_lock:
            ws.send(command_message(throttle, steering, reset))
        if reset and now - self._reset_started >= self._reset_hold:
            self._reset_started = None
            self._reset_event.clear()
        return time.monotonic(), throttle, steering, age, reset

    # logs

    def _event(self, ev: dict) -> None:
        if self._events_fp is not None:
            self._events_fp.write(json.dumps(ev) + "\n")
            self._events_fp.flush()

    def _log_events(self, d: dict, t: float) -> None:
        laps, coll = d.get("lap_count"), d.get("collision_count")
        if laps is not None:
            if self._prev_lap_count is not None and laps > self._prev_lap_count:
                self._event({"t": t, "event": "lap", "lap_count": laps,
                             "lap_time": d.get("last_lap_time"), "best_lap_time": d.get("best_lap_time")})
                self.logger.info(f"Lap {laps}: {d.get('last_lap_time', float('nan')):.3f} s")
            elif self._prev_lap_count is not None and laps < self._prev_lap_count:
                self._event({"t": t, "event": "lap_count_reset", "lap_count": laps})
            self._prev_lap_count = laps
        if coll is not None:
            if self._prev_collisions is not None and coll > self._prev_collisions:
                self._event({"t": t, "event": "collision", "collisions": coll,
                             "position": d.get("position")})
                self.logger.warning(f"Collision ({coll} total)")
            elif self._prev_collisions is not None and coll < self._prev_collisions:
                self._event({"t": t, "event": "collision_count_reset", "collisions": coll})
            self._prev_collisions = coll

    def _log_message(self, d: dict, sim_time: float, new_frame: bool, t_reply: float,
                     throttle: float, steering: float, age: float, reset: bool) -> None:
        if self._csv is None:
            return
        pos = d.get("position") or [float("nan")] * 3
        yaw = yaw_from_quat(d["orientation_quat"]) if "orientation_quat" in d else float("nan")
        self._csv.writerow([
            f"{self._last_rx:.6f}", f"{t_reply:.6f}", sim_time, int(new_frame),
            f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{yaw:.5f}", f"{d.get('speed', float('nan')):.4f}",
            d.get("throttle", ""), d.get("steering", ""), f"{throttle:.4f}", f"{steering:.4f}",
            f"{age:.4f}" if math.isfinite(age) else "inf", int(reset),
            d.get("lap_count", ""), d.get("collision_count", ""),
            f"{d['euler'][2]:.5f}" if len(d.get("euler", ())) == 3 else "",
        ])
        if new_frame:
            self._csv_fp.flush()
