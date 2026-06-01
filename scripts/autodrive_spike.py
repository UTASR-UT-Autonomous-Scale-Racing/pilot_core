#!/usr/bin/env python3
"""Protocol spike against the stock AutoDRIVE RoboRacer sim (2026-iros).

Findings (see docs/autodrive.md): the sim's Unity client (websocket-sharp + an old
SocketIOComponent) opens ws://host:port/socket.io/?EIO=4&transport=websocket, sends
client-side pings ("2"), never sends a namespace CONNECT ("40") and only starts
emitting 'Bridge' telemetry after the server has sent one. Neither python-socketio 4
(rejects EIO=4) nor 5 (waits for "40") completes that handshake, so this speaks the
small subset directly over websockets.

This server logs every key, the message rate and payload sizes, and replies neutral.

    python scripts/autodrive_spike.py [--port 4567] [--seconds 30] [--out spike.json]
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import socket
import sys
import time
from importlib.metadata import version

import numpy as np
from websockets.sync.server import serve

NEUTRAL = '42["Bridge",{"V1 Throttle":"0.0","V1 Steering":"0.0","V1 Reset":"False"}]'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=4567)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--phases", default=None,
                    help="comma list of mode:seconds on one connection; mode = reply | none | timer<Hz>")
    ap.add_argument("--replies", type=int, default=1, help="Bridge replies per telemetry message")
    ap.add_argument("--quickack", action="store_true", help="re-arm TCP_QUICKACK on every message")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    args = ap.parse_args()
    print(f"python {sys.version.split()[0]}  websockets {version('websockets')}", flush=True)

    stats = {"connects": 0, "pings": 0, "t": [], "sizes": [], "keys": None, "first": None,
             "lidar_beams": None, "image": None, "reply_ms": []}

    phase_log = []

    def run_phases(ws, plan):
        import threading
        state = {"mode": "reply", "hz": 0.0}
        lock = threading.Lock()

        def timer():
            nxt = time.monotonic()
            while state["mode"] != "done":
                if state["mode"].startswith("timer"):
                    with lock:
                        ws.send(NEUTRAL)
                    nxt += 1.0 / state["hz"]
                    time.sleep(max(nxt - time.monotonic(), 0))
                else:
                    nxt = time.monotonic()
                    time.sleep(0.005)
        threading.Thread(target=timer, daemon=True).start()
        for mode, dur in plan:
            state["mode"], state["hz"] = mode, float(mode[5:] or 0) if mode.startswith("timer") else 0.0
            t0, n, laps = time.monotonic(), 0, set()
            while time.monotonic() - t0 < dur:
                try:
                    msg = ws.recv(timeout=0.5)
                except TimeoutError:
                    continue
                if msg == "2":
                    with lock:
                        ws.send("3")
                    continue
                if msg.startswith('42["Bridge"'):
                    n += 1
                    laps.add(json.loads(msg[2:])[1].get("V1 Lap Time"))
                    if mode == "reply":
                        with lock:
                            ws.send(NEUTRAL)
            el = time.monotonic() - t0
            phase_log.append({"mode": mode, "msgs_hz": n / el, "distinct_hz": len(laps) / el})
            print("PHASE", phase_log[-1], flush=True)
        state["mode"] = "done"

    def handler(ws):
        if args.phases:
            ws.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            ws.send('0{"sid":"spike","upgrades":[],"pingInterval":25000,"pingTimeout":20000}')
            ws.send("40")
            ws.send(NEUTRAL)
            run_phases(ws, [(m, float(d)) for m, d in (x.split(":") for x in args.phases.split(","))])
            return
        sock = ws.socket
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        stats["connects"] += 1
        print(f"connect {ws.request.path}", flush=True)
        ws.send('0{"sid":"spike","upgrades":[],"pingInterval":25000,"pingTimeout":20000}')
        ws.send("40")
        ws.send(NEUTRAL)
        for msg in ws:
            if msg == "2":
                stats["pings"] += 1
                ws.send("3")
                continue
            if not msg.startswith('42["Bridge"'):
                print(f"other frame: {msg[:80]!r}", flush=True)
                continue
            t_rx = time.monotonic()
            if args.quickack:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_QUICKACK, 1)
            event, data = json.loads(msg[2:])
            stats["t"].append(t_rx)
            stats.setdefault("lap_time", []).append(data.get("V1 Lap Time"))
            stats["sizes"].append(len(msg))
            if stats["keys"] is None:
                stats["keys"] = {k: len(v) for k, v in data.items()}
                stats["first"] = {k: v if len(v) < 100 else v[:40] + f"...<{len(v)} chars>"
                                  for k, v in data.items()}
                ranges = np.array(gzip.decompress(base64.b64decode(data["V1 LIDAR Range Array"]))
                                  .decode().split(), dtype=float)
                stats["lidar_beams"] = len(ranges)
                stats["lidar_centre_m"] = float(ranges[len(ranges) // 2])
                import cv2
                img = cv2.imdecode(np.frombuffer(base64.b64decode(data["V1 Front Camera Image"]), np.uint8),
                                   cv2.IMREAD_COLOR)
                stats["image"] = list(img.shape)
                print("KEYS (name: chars):", flush=True)
                for k, n in stats["keys"].items():
                    print(f"  {k!r}: {n}", flush=True)
                print("FIRST:", json.dumps(stats["first"], indent=1), flush=True)
                print(f"lidar beams={stats['lidar_beams']} image={stats['image']}", flush=True)
            for _ in range(args.replies):
                ws.send(NEUTRAL)
            stats["reply_ms"].append((time.monotonic() - t_rx) * 1e3)
            if len(stats["t"]) % 200 == 0:
                ts = stats["t"][-200:]
                print(f"msg {len(stats['t'])}: {199 / (ts[-1] - ts[0]):.1f} Hz", flush=True)

    with serve(handler, "127.0.0.1", args.port, max_size=None, compression=None) as server:
        import threading
        threading.Timer(args.seconds, server.shutdown).start()
        server.serve_forever()

    t = np.array(stats["t"])
    dt = np.diff(t) if len(t) > 1 else np.array([0.0])
    summary = {
        "python": sys.version.split()[0], "websockets": version("websockets"),
        "connects": stats["connects"], "messages": len(t), "client_pings": stats["pings"],
        "distinct_frames": len(set(stats.get("lap_time", []))),
        "distinct_rate_hz": float(len(set(stats.get("lap_time", []))) / (t[-1] - t[0])) if len(t) > 1 else 0.0,
        "rate_hz": float((len(t) - 1) / (t[-1] - t[0])) if len(t) > 1 else 0.0,
        "period_ms_p50_p95_max": [float(np.percentile(dt, 50) * 1e3), float(np.percentile(dt, 95) * 1e3),
                                  float(dt.max() * 1e3)],
        "reply_ms_p50": float(np.median(stats["reply_ms"])) if stats["reply_ms"] else None,
        "mean_payload_bytes": float(np.mean(stats["sizes"])) if stats["sizes"] else 0.0,
        "lidar_beams": stats["lidar_beams"], "lidar_centre_m": stats.get("lidar_centre_m"),
        "front_camera_shape": stats["image"],
        "keys": stats["keys"], "first": stats["first"],
    }
    summary["phases"] = phase_log
    print("SUMMARY", json.dumps(summary, indent=1), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
