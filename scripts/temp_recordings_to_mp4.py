#!/usr/bin/env python3
"""Render an MP4 from recorded pilot_core buffer files.

This script is intended to be run after pilot_core has stopped and recorder
workers have flushed their ``*.bin`` outputs (for example in
``/tmp/pilot_recordings``).

Output video layout (2x2):
  - Top-left: RGB frame
  - Top-right: Depth color map
  - Bottom-left: Segmentation mask (if available)
    - Bottom-right: Map plot (or telemetry panel when map plot is disabled)
"""

from __future__ import annotations

import argparse
import bisect
import glob
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import cv2
import numpy as np
import yaml

FILE_HEADER_SIZE = 64
RECORD_HEADER_FMT = "<QddIHH"
RECORD_HEADER_SIZE = struct.calcsize(RECORD_HEADER_FMT)


def _find_latest_file(recording_dir: str, prefix: str) -> Optional[str]:
    pattern = os.path.join(recording_dir, f"{prefix}_*.bin")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def _iter_records(path: str) -> Iterator[tuple[dict[str, Any], bytes]]:
    with open(path, "rb") as fp:
        raw = fp.read(FILE_HEADER_SIZE)
        # Some files may be empty/partial if a recorder started but never wrote.
        if len(raw) < FILE_HEADER_SIZE:
            return
        while True:
            raw_hdr = fp.read(RECORD_HEADER_SIZE)
            if len(raw_hdr) < RECORD_HEADER_SIZE:
                break
            seq_no, produced_ts, read_ts, payload_size, schema_ver, dtype_tag = struct.unpack(
                RECORD_HEADER_FMT, raw_hdr
            )
            payload = fp.read(payload_size)
            if len(payload) < payload_size:
                break
            yield {
                "seq_no": seq_no,
                "produced_ts": produced_ts,
                "read_ts": read_ts,
                "payload_size": payload_size,
                "schema_version": schema_ver,
                "dtype_tag": dtype_tag,
            }, payload


@dataclass
class TimedValue:
    ts: float
    value: Any


class TimedSeries:
    def __init__(self, items: Iterable[TimedValue]) -> None:
        self.items = sorted(items, key=lambda x: x.ts)
        self.ts = [x.ts for x in self.items]

    def nearest(self, ts: float) -> Any:
        if not self.items:
            return None
        i = bisect.bisect_left(self.ts, ts)
        if i == 0:
            return self.items[0].value
        if i >= len(self.items):
            return self.items[-1].value
        left = self.items[i - 1]
        right = self.items[i]
        if abs(left.ts - ts) <= abs(right.ts - ts):
            return left.value
        return right.value


def _decode_frame_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    rh, rw, dh, dw = struct.unpack_from("<IIII", payload, 0)
    off = 16
    rgb_size = rh * rw * 3
    rgb = np.frombuffer(payload, dtype=np.uint8, count=rgb_size, offset=off).reshape((rh, rw, 3)).copy()
    off += rgb_size

    depth_pixels = dh * dw
    remaining = len(payload) - off
    if remaining >= depth_pixels * 4:
        depth = np.frombuffer(payload, dtype=np.float32, count=depth_pixels, offset=off).reshape((dh, dw)).copy()
    else:
        depth_u16 = np.frombuffer(payload, dtype=np.uint16, count=depth_pixels, offset=off).reshape((dh, dw))
        depth = depth_u16.astype(np.float32) / 1000.0
    return rgb, depth


def _decode_seg_payload(payload: bytes) -> np.ndarray:
    h, w = struct.unpack_from("<II", payload, 0)
    return np.frombuffer(payload, dtype=np.float32, count=h * w, offset=8).reshape((h, w)).copy()


def _decode_imu_payload(payload: bytes) -> np.ndarray:
    n = len(payload) // 4
    return np.frombuffer(payload, dtype=np.float32, count=n).copy()


def _decode_control_payload(payload: bytes) -> tuple[float, float]:
    return struct.unpack_from("<ff", payload, 0)


def _decode_pose_payload(payload: bytes) -> tuple[np.ndarray, int]:
    pose = np.frombuffer(payload, dtype=np.float32, count=16, offset=0).reshape((4, 4)).copy()
    (n_pts,) = struct.unpack_from("<I", payload, 64)
    return pose, int(n_pts)


def _decode_boundary_payload(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    (n_outer,) = struct.unpack_from("<I", payload, 0)
    offset = 4
    outer = np.frombuffer(payload, dtype=np.float32, count=n_outer * 2, offset=offset).reshape((n_outer, 2)).copy()
    offset += n_outer * 2 * 4
    (n_inner,) = struct.unpack_from("<I", payload, offset)
    offset += 4
    inner = np.frombuffer(payload, dtype=np.float32, count=n_inner * 2, offset=offset).reshape((n_inner, 2)).copy()
    return outer, inner


def _load_series(path: Optional[str], decoder) -> TimedSeries:
    if not path:
        return TimedSeries([])
    items: list[TimedValue] = []
    for hdr, payload in _iter_records(path):
        items.append(TimedValue(ts=float(hdr["produced_ts"]), value=decoder(payload)))
    return TimedSeries(items)


def _depth_to_panel(depth: np.ndarray, out_wh: tuple[int, int]) -> np.ndarray:
    w, h = out_wh
    d = depth
    if d.shape[:2] != (h, w):
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    valid = d > 0.001
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if not np.any(valid):
        return panel
    lo = float(np.percentile(d[valid], 3))
    hi = float(np.percentile(d[valid], 97))
    hi = max(hi, lo + 1e-3)
    norm = np.zeros_like(d, dtype=np.uint8)
    norm[valid] = np.clip((d[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    panel = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    panel[~valid] = (0, 0, 0)
    cv2.putText(panel, f"depth [{lo:.2f},{hi:.2f}] m", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return panel


def _seg_to_panel(seg: Optional[np.ndarray], out_wh: tuple[int, int]) -> np.ndarray:
    w, h = out_wh
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    if seg is None or seg.size == 0:
        cv2.putText(panel, "no segmentation", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        return panel

    s = seg
    if s.shape[:2] != (h, w):
        s = cv2.resize(s, (w, h), interpolation=cv2.INTER_NEAREST)
    # Normalize class/confidence masks for visualization
    lo = float(np.min(s))
    hi = float(np.max(s))
    if hi <= lo + 1e-6:
        norm = np.zeros_like(s, dtype=np.uint8)
    else:
        norm = np.clip((s - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    panel = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    cv2.putText(panel, f"seg [{lo:.2f},{hi:.2f}]", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
    return panel


def _telemetry_panel(
    wh: tuple[int, int],
    ts: float,
    seq_no: int,
    imu: Optional[np.ndarray],
    ctrl: Optional[tuple[float, float]],
    pose_data: Optional[tuple[np.ndarray, int]],
) -> np.ndarray:
    w, h = wh
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    y = 32
    lh = 32

    def line(text: str, col=(220, 220, 220)) -> None:
        nonlocal y
        cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        y += lh

    line(f"seq={seq_no} ts={ts:.3f}")

    if imu is not None and len(imu) >= 16:
        line(f"imu q=({imu[0]:+.3f},{imu[1]:+.3f},{imu[2]:+.3f},{imu[3]:+.3f})")
        line(f"imu a=({imu[4]:+.3f},{imu[5]:+.3f},{imu[6]:+.3f})")
        line(f"imu g=({imu[7]:+.3f},{imu[8]:+.3f},{imu[9]:+.3f})")
        line(f"imu m=({imu[10]:+.3f},{imu[11]:+.3f},{imu[12]:+.3f})")
        line(f"imu raw_ctrl=({imu[13]:+.2f},{imu[14]:+.2f}) imu_ts={imu[15]:.3f}")
    elif imu is not None:
        line(f"imu n={len(imu)}", (180, 180, 180))
    else:
        line("imu: none", (180, 180, 180))

    if ctrl is not None:
        line(f"control steer={ctrl[0]:+.3f} throttle={ctrl[1]:+.3f}")
    else:
        line("control: none", (180, 180, 180))

    if pose_data is not None:
        pose, n_pts = pose_data
        line(f"pose xyz=({pose[0,3]:+.3f},{pose[1,3]:+.3f},{pose[2,3]:+.3f})")
        line(f"map points={n_pts}")
    else:
        line("pose: none", (180, 180, 180))

    return panel


def _to_px(
    points: np.ndarray,
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    width: int,
    height: int,
    margin: int,
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    x_span = max(x_max - x_min, 1e-6)
    z_span = max(z_max - z_min, 1e-6)
    x = margin + ((points[:, 0] - x_min) / x_span) * (width - 2 * margin)
    y = height - margin - ((points[:, 1] - z_min) / z_span) * (height - 2 * margin)
    return np.stack([x, y], axis=1).astype(np.int32)


def _map_panel(
    wh: tuple[int, int],
    boundary: Optional[tuple[np.ndarray, np.ndarray]],
    trail: np.ndarray,
    current_pos: Optional[np.ndarray],
    seq_no: int,
) -> np.ndarray:
    w, h = wh
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(panel, "map plot", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    outer = np.empty((0, 2), dtype=np.float32)
    inner = np.empty((0, 2), dtype=np.float32)
    if boundary is not None:
        outer, inner = boundary

    pts_for_bounds: list[np.ndarray] = []
    if len(outer) > 0:
        pts_for_bounds.append(outer)
    if len(inner) > 0:
        pts_for_bounds.append(inner)
    if len(trail) > 0:
        pts_for_bounds.append(trail)

    if not pts_for_bounds:
        cv2.putText(panel, "no map data", (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        return panel

    pts = np.vstack(pts_for_bounds)
    x_min, z_min = np.min(pts[:, 0]), np.min(pts[:, 1])
    x_max, z_max = np.max(pts[:, 0]), np.max(pts[:, 1])
    pad_x = max(1.0, 0.1 * (x_max - x_min if x_max > x_min else 1.0))
    pad_z = max(1.0, 0.1 * (z_max - z_min if z_max > z_min else 1.0))
    x_min -= pad_x
    x_max += pad_x
    z_min -= pad_z
    z_max += pad_z
    margin = 16

    if len(outer) > 1:
        closed_outer = np.vstack([outer, outer[:1]])
        px = _to_px(closed_outer, x_min, x_max, z_min, z_max, w, h, margin)
        cv2.polylines(panel, [px], False, (255, 255, 0), 2, cv2.LINE_AA)
    if len(inner) > 1:
        closed_inner = np.vstack([inner, inner[:1]])
        px = _to_px(closed_inner, x_min, x_max, z_min, z_max, w, h, margin)
        cv2.polylines(panel, [px], False, (255, 255, 255), 2, cv2.LINE_AA)

    if len(trail) > 1:
        px = _to_px(trail, x_min, x_max, z_min, z_max, w, h, margin)
        cv2.polylines(panel, [px], False, (0, 165, 255), 2, cv2.LINE_AA)

    if current_pos is not None and current_pos.size == 2:
        px = _to_px(current_pos.reshape(1, 2), x_min, x_max, z_min, z_max, w, h, margin)
        cv2.circle(panel, tuple(px[0]), 5, (0, 255, 0), -1, cv2.LINE_AA)

    cv2.putText(panel, f"seq={seq_no}", (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return panel


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create MP4 from recorded pilot_core buffer files")
    p.add_argument("--recording-dir", default="/tmp/pilot_recordings", help="Directory containing *.bin recordings")
    p.add_argument("--output", default="/tmp/pilot_recordings/replay.mp4", help="Output MP4 path")
    p.add_argument("--config", default="config.yaml", help="Config file used for replay_video defaults")
    p.add_argument("--fps", type=float, default=20.0, help="Output video FPS")
    p.add_argument("--panel-width", type=int, default=640, help="Width of each panel")
    p.add_argument("--panel-height", type=int, default=400, help="Height of each panel")
    p.add_argument("--include-map-plot", dest="include_map_plot", action="store_true", help="Render map plot panel")
    p.add_argument("--no-map-plot", dest="include_map_plot", action="store_false", help="Disable map plot panel")
    p.set_defaults(include_map_plot=None)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    rec_dir = args.recording_dir

    include_map_plot = True
    if os.path.exists(args.config):
        with open(args.config, "r") as fp:
            cfg = yaml.safe_load(fp) or {}
        include_map_plot = bool(cfg.get("replay_video", {}).get("include_map_plot", True))
    if args.include_map_plot is not None:
        include_map_plot = bool(args.include_map_plot)

    frame_path = _find_latest_file(rec_dir, "frame")
    if frame_path is None:
        print(f"No frame recording found in {rec_dir}")
        return 1

    seg_path = _find_latest_file(rec_dir, "segmentation")
    imu_path = _find_latest_file(rec_dir, "imu")
    ctrl_path = _find_latest_file(rec_dir, "control")
    pose_path = _find_latest_file(rec_dir, "pose")
    boundary_path = _find_latest_file(rec_dir, "boundary")

    print("Using files:")
    print(f"  frame: {frame_path}")
    print(f"  segmentation: {seg_path}")
    print(f"  imu: {imu_path}")
    print(f"  control: {ctrl_path}")
    print(f"  pose: {pose_path}")
    print(f"  boundary: {boundary_path}")
    print(f"  include_map_plot: {include_map_plot}")

    seg_series = _load_series(seg_path, _decode_seg_payload)
    imu_series = _load_series(imu_path, _decode_imu_payload)
    ctrl_series = _load_series(ctrl_path, _decode_control_payload)
    pose_series = _load_series(pose_path, _decode_pose_payload)
    boundary_series = _load_series(boundary_path, _decode_boundary_payload)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pw, ph = int(args.panel_width), int(args.panel_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (pw * 2, ph * 2))

    n_written = 0
    trail: list[np.ndarray] = []
    for hdr, payload in _iter_records(frame_path):
        seq_no = int(hdr["seq_no"])
        ts = float(hdr["produced_ts"])

        rgb, depth = _decode_frame_payload(payload)
        if rgb.shape[:2] != (ph, pw):
            rgb = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_LINEAR)

        depth_panel = _depth_to_panel(depth, (pw, ph))
        seg_panel = _seg_to_panel(seg_series.nearest(ts), (pw, ph))
        pose_data = pose_series.nearest(ts)
        current_pos = None
        if pose_data is not None:
            pose, _ = pose_data
            current_pos = np.array([pose[0, 3], pose[2, 3]], dtype=np.float32)
            if not trail or float(np.linalg.norm(trail[-1] - current_pos)) > 1e-4:
                trail.append(current_pos)

        if include_map_plot:
            map_panel = _map_panel(
                (pw, ph),
                boundary_series.nearest(ts),
                np.array(trail, dtype=np.float32) if trail else np.empty((0, 2), dtype=np.float32),
                current_pos,
                seq_no,
            )
            bottom_right = map_panel
        else:
            bottom_right = _telemetry_panel(
                (pw, ph),
                ts,
                seq_no,
                imu_series.nearest(ts),
                ctrl_series.nearest(ts),
                pose_data,
            )

        cv2.putText(rgb, f"rgb seq={seq_no}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

        top = np.hstack([rgb, depth_panel])
        bot = np.hstack([seg_panel, bottom_right])
        frame = np.vstack([top, bot])
        writer.write(frame)
        n_written += 1

    writer.release()
    print(f"Wrote {n_written} frames to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
