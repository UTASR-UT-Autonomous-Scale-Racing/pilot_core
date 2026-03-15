#!/usr/bin/env python3
"""Simple local HTTP server for WebPublishWorker telemetry.

Usage:
    python scripts/web_telemetry_receiver.py
    python scripts/web_telemetry_receiver.py --port 8080 --jsonl /tmp/telemetry.jsonl
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _TelemetryState:
    def __init__(
        self,
        jsonl_path: Optional[Path] = None,
        selected_sections: Optional[set[str]] = None,
    ) -> None:
        self.count = 0
        self.last_ts = 0.0
        self.last_payload: Dict[str, Any] = {}
        self.lock = threading.Lock()
        self.jsonl_path = jsonl_path
        self.selected_sections = selected_sections or set()

    def record(self, payload: Dict[str, Any]) -> int:
        with self.lock:
            self.count += 1
            self.last_ts = time.time()
            self.last_payload = payload
            count = self.count

            if self.jsonl_path is not None:
                row = {
                    "recv_timestamp": self.last_ts,
                    "payload": payload,
                }
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
        return count


class TelemetryHandler(BaseHTTPRequestHandler):
    server_version = "WebTelemetryReceiver/1.0"

    def _send_json(self, status: int, obj: Dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        expected_path = self.server.expected_path  # type: ignore[attr-defined]
        state = self.server.state  # type: ignore[attr-defined]

        if self.path != expected_path:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": f"POST {expected_path} expected"},
            )
            return

        content_length = self.headers.get("Content-Length", "0")
        try:
            nbytes = int(content_length)
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid Content-Length"},
            )
            return

        raw = self.rfile.read(nbytes)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": f"Invalid JSON: {exc}"},
            )
            return

        msg_no = state.record(payload)
        self._print_summary(msg_no, payload)

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "message_no": msg_no,
                "received_at": _utc_now(),
            },
        )

    def _print_summary(self, msg_no: int, payload: Dict[str, Any]) -> None:
        selected = self.server.state.selected_sections  # type: ignore[attr-defined]
        vehicle_id = payload.get("vehicle_id", "?")
        trajectory = payload.get("trajectory", []) if "trajectory" in selected else []
        pose = payload.get("pose") if "pose" in selected else None
        control = payload.get("control") if "control" in selected else None
        track = payload.get("track", {}) if "track" in selected else {}
        imu = payload.get("imu") if "imu" in selected else None
        pointcloud = payload.get("pointcloud") if "pointcloud" in selected else None

        if isinstance(trajectory, list):
            traj_len = len(trajectory)
        else:
            traj_len = 0

        if isinstance(track, dict):
            outer = track.get("outer", [])
            inner = track.get("inner", [])
            outer_len = len(outer) if isinstance(outer, list) else 0
            inner_len = len(inner) if isinstance(inner, list) else 0
        else:
            outer_len = 0
            inner_len = 0

        if isinstance(control, dict):
            steering = control.get("steering")
            throttle = control.get("throttle")
        else:
            steering = None
            throttle = None

        imu_ok = isinstance(imu, dict)
        pc_count = None
        if isinstance(pointcloud, dict):
            raw_count = pointcloud.get("count")
            if isinstance(raw_count, int):
                pc_count = raw_count

        parts = [
            f"[{_utc_now()}] #{msg_no} vehicle={vehicle_id}",
            f"pose={'yes' if isinstance(pose, dict) else 'no'}" if "pose" in selected else "",
            f"traj={traj_len}" if "trajectory" in selected else "",
            f"outer={outer_len} inner={inner_len}" if "track" in selected else "",
            f"steer={steering} throttle={throttle}" if "control" in selected else "",
            f"imu={'yes' if imu_ok else 'no'}" if "imu" in selected else "",
            f"pc_count={pc_count}" if "pointcloud" in selected else "",
        ]
        parts = [p for p in parts if p]

        print(" ".join(parts), flush=True)

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep output focused on telemetry summaries.
        return


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive telemetry from WebPublishWorker")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--path",
        default="/api/telemetry",
        help="HTTP path to accept POST telemetry (default: /api/telemetry)",
    )
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Optional JSONL output file to append all payloads",
    )
    parser.add_argument(
        "--no-kill-port-owner",
        action="store_true",
        help="Do not kill an existing process when bind fails with address in use",
    )
    parser.add_argument("--all", action="store_true", help="Show all payload sections")
    parser.add_argument("--pose", action="store_true", help="Show pose section")
    parser.add_argument("--trajectory", action="store_true", help="Show trajectory section")
    parser.add_argument("--control", action="store_true", help="Show control section")
    parser.add_argument("--track", action="store_true", help="Show track section")
    parser.add_argument("--imu", action="store_true", help="Show IMU section")
    parser.add_argument("--pointcloud", action="store_true", help="Show pointcloud section")
    return parser.parse_args()


def _selected_sections_from_args(args: argparse.Namespace) -> set[str]:
    section_flags = {
        "pose": args.pose,
        "trajectory": args.trajectory,
        "control": args.control,
        "track": args.track,
        "imu": args.imu,
        "pointcloud": args.pointcloud,
    }
    selected = {name for name, enabled in section_flags.items() if enabled}
    if args.all:
        return set(section_flags.keys())
    if not selected:
        return {"pose", "trajectory", "control", "track"}
    return selected


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _find_listening_pids(port: int) -> list[int]:
    pids: list[int] = []

    # Primary path: lsof gives direct PIDs for listening sockets.
    try:
        out = subprocess.check_output(
            ["lsof", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback: fuser output can include non-digit suffixes, so parse cautiously.
    if not pids:
        try:
            out = subprocess.check_output(
                ["fuser", "-n", "tcp", str(port)],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for token in out.replace("\n", " ").split():
                pid_txt = "".join(ch for ch in token if ch.isdigit())
                if pid_txt:
                    pids.append(int(pid_txt))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # De-dup and avoid trying to kill ourselves.
    uniq = sorted(set(pids))
    return [pid for pid in uniq if pid != os.getpid()]


def _kill_port_owner(port: int) -> bool:
    pids = _find_listening_pids(port)
    if not pids:
        print(
            f"[{_utc_now()}] Port {port} is busy but no owning PID was found.",
            flush=True,
        )
        return False

    print(
        f"[{_utc_now()}] Port {port} in use. Terminating listener PID(s): {pids}",
        flush=True,
    )

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError:
            print(
                f"[{_utc_now()}] Permission denied while terminating PID {pid}",
                flush=True,
            )

    deadline = time.time() + 1.5
    while time.time() < deadline:
        alive = [pid for pid in pids if _pid_exists(pid)]
        if not alive:
            return True
        time.sleep(0.1)

    alive = [pid for pid in pids if _pid_exists(pid)]
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    time.sleep(0.1)
    return not any(_pid_exists(pid) for pid in pids)


def main() -> None:
    args = _parse_args()
    selected_sections = _selected_sections_from_args(args)

    jsonl_path = Path(args.jsonl).expanduser().resolve() if args.jsonl else None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    state = _TelemetryState(
        jsonl_path=jsonl_path,
        selected_sections=selected_sections,
    )

    kill_port_owner = not args.no_kill_port_owner

    try:
        server = ThreadingHTTPServer((args.host, args.port), TelemetryHandler)
    except OSError as exc:
        if (
            exc.errno == errno.EADDRINUSE
            and kill_port_owner
            and _kill_port_owner(args.port)
        ):
            print(
                f"[{_utc_now()}] Retrying bind on {args.host}:{args.port}",
                flush=True,
            )
            server = ThreadingHTTPServer((args.host, args.port), TelemetryHandler)
        else:
            raise

    server.expected_path = args.path  # type: ignore[attr-defined]
    server.state = state  # type: ignore[attr-defined]

    print(
        f"[{_utc_now()}] Listening on http://{args.host}:{args.port}{args.path}",
        flush=True,
    )
    print(
        f"[{_utc_now()}] Selected sections: {sorted(selected_sections)}",
        flush=True,
    )
    if jsonl_path is not None:
        print(f"[{_utc_now()}] Writing payloads to {jsonl_path}", flush=True)

    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        print(
            f"[{_utc_now()}] Stopped. Received {state.count} payload(s).",
            flush=True,
        )


if __name__ == "__main__":
    main()
