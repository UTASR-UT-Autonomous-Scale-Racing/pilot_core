#!/usr/bin/env python3
"""Render a synchronised telemetry HUD video from an exported run (scripts/export_recording.py).

    python scripts/render_hud.py <export dir> --out hud.mp4                     # full run, real time
    python scripts/render_hud.py <export dir> --out hud.gif --speed 4 --fps 8   # README time-lapse

One 1280x720 frame per output tick, every panel drawn from what the car had at that moment
(the latest record at or before the tick, on the export's t_mono clock):

  sim view        screen.mp4 (sim chase camera); the depth image when the run has no video
  car camera      camera/rgb_*.png with the VisionWorker mask (segmentation/seg_*.png)
  map             LiDAR wall points accumulated so far, the pose trail coloured by speed,
                  the current scan, the car, and the extracted boundary once the loop closes
  telemetry       speed / steering / throttle gauges (command vs. after the safety supervisor),
                  10 s strip charts, lap times, collisions and per-buffer data age

--start / --end clip the run (seconds on t_s). --speed > 1 is a time-lapse. .gif output is
encoded with a two-pass ffmpeg palette and scaled to --gif-width.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

W, H = 1280, 720
HEADER = 40
TOP_H = 360
MAP_W = 480
BOT_H = H - HEADER - TOP_H
SIM_UI_BAR = 30          # the sim window's own title bar (screen.mp4 rows), cropped off

# RGB
BG = (13, 17, 23)
PANEL = (22, 27, 34)
GRID = (48, 54, 61)
TEXT = (230, 237, 243)
MUTED = (139, 148, 158)
CMD = (255, 166, 87)     # ControlWorker command
ACT = (88, 200, 255)     # after the safety supervisor (sent to the vehicle)
SPEED = (126, 231, 135)
WALL = (190, 196, 204)
SCAN = (255, 92, 92)
BOUNDARY = (210, 168, 255)
TRACK = (46, 204, 113)

FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF", "/Library/Fonts")


def _font(size: int, bold: bool = False, mono: bool = False):
    from PIL import ImageFont

    name = "DejaVuSans" + ("Mono" if mono else "") + ("-Bold" if bold else "") + ".ttf"
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def _load_csv(path: str) -> dict[str, np.ndarray]:
    """Numeric columns of a CSV as float arrays (empty cells -> nan); text columns as lists."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: dict = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            out[k] = vals
    return out


def _latest(t_ref: np.ndarray, t: float) -> int:
    """Index of the last record at or before t, -1 if none yet."""
    return int(np.searchsorted(t_ref, t, side="right")) - 1


def _signed(u: np.ndarray) -> np.ndarray:
    """Unit-scale command (0.5 neutral) -> signed [-1, 1]."""
    return (np.asarray(u) - 0.5) * 2.0


class Run:
    """An export directory, loaded once."""

    def __init__(self, export: str):
        self.dir = export

        def opt(name):
            p = os.path.join(export, name)
            return _load_csv(p) if os.path.isfile(p) else {}

        self.camera, self.seg = opt("camera.csv"), opt("segmentation.csv")
        self.pose, self.state = opt("pose.csv"), opt("state.csv")
        self.control, self.act = opt("control.csv"), opt("actuation.csv")
        self.video = opt("video_frames.csv")
        self.sim = opt("sim_messages.csv")
        self.events = opt("events.csv")
        if not self.pose:
            raise SystemExit(f"{export}: no pose.csv; the HUD needs a pose source")

        lid = os.path.join(export, "lidar.npz")
        self.lidar = dict(np.load(lid)) if os.path.isfile(lid) else None
        pc = os.path.join(export, "pointcloud.npz")
        self.pc = dict(np.load(pc)) if os.path.isfile(pc) else None
        bd = os.path.join(export, "boundary.npz")
        self.boundary = dict(np.load(bd)) if os.path.isfile(bd) else None

        self.config = {}
        cfg = os.path.join(export, "config.yaml")
        if os.path.isfile(cfg):
            import yaml

            with open(cfg) as f:
                self.config = yaml.safe_load(f) or {}

        with open(os.path.join(export, "export_summary.json")) as f:
            self.t_first = float(json.load(f)["t_first_mono"])

        self.laps = []           # (t_mono, lap_time)
        self.t_reset = None
        for t, ev, detail in zip(self.events.get("t_mono", []), self.events.get("event", []),
                                 self.events.get("detail", [])):
            if ev == "lap":
                self.laps.append((t, float(json.loads(detail)["lap_time"])))
            elif ev == "reset_sent" and self.t_reset is None:
                self.t_reset = t

        self.v_scale = max(1.0, float(np.nanmax(np.abs(self.state["v"]))) * 1.15) if self.state else 3.0

    def t_range(self) -> tuple[float, float]:
        ts = [self.pose["t_mono"][0], self.pose["t_mono"][-1]]
        if self.video:
            ts += [self.video["t_mono"][0], self.video["t_mono"][-1]]
        return min(ts), max(ts)


class MapLayer:
    """World -> pixel transform and the layers that only grow (wall points, trail)."""

    def __init__(self, run: Run, w: int, h: int, margin: int = 16):
        self.w, self.h = w, h
        pts = [np.c_[run.pose["x"], run.pose["y"]]]
        if run.pc is not None and len(run.pc["points"]):
            pts.append(run.pc["points"][:, :2])
        if run.boundary is not None:
            pts += [run.boundary[k] for k in run.boundary if k.startswith(("outer", "inner"))]
        allp = np.concatenate(pts)
        lo, hi = np.percentile(allp, 0.5, axis=0), np.percentile(allp, 99.5, axis=0)
        span = np.maximum(hi - lo, 1e-3)
        # turn a track that is taller than wide by 90 deg so it fills the wide panel
        self.rot = bool(span[1] > span[0] and w > h)
        if self.rot:
            lo, hi, span = self._r(lo)[0], self._r(hi)[0], span[::-1]
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        self.s = min((w - 2 * margin) / span[0], (h - 2 * margin - 22) / span[1])
        self.c = (lo + hi) / 2
        self.walls = np.zeros((h, w), np.float32)
        self.trail = np.zeros((h, w, 3), np.uint8)
        self.trail_mask = np.zeros((h, w), np.uint8)
        self.pc_done = -1
        self.trail_done = 0

    def _r(self, xy: np.ndarray) -> np.ndarray:
        """World -> display frame: rotated +90 deg (x' = -y, y' = x) when self.rot."""
        xy = np.atleast_2d(xy)
        return np.c_[-xy[:, 1], xy[:, 0]] if self.rot else xy

    def px(self, xy: np.ndarray) -> np.ndarray:
        xy = self._r(np.atleast_2d(xy)[:, :2])
        u = (xy[:, 0] - self.c[0]) * self.s + self.w / 2
        v = self.h / 2 + 11 - (xy[:, 1] - self.c[1]) * self.s
        return np.c_[u, v]

    def add_walls(self, run: Run, upto: int) -> None:
        if run.pc is None or upto <= self.pc_done:
            return
        a, b = run.pc["offsets"][self.pc_done + 1], run.pc["offsets"][upto + 1]
        p = self.px(run.pc["points"][a:b, :2]).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < self.w) & (p[:, 1] >= 0) & (p[:, 1] < self.h)
        np.add.at(self.walls, (p[ok, 1], p[ok, 0]), 1.0)
        self.pc_done = upto

    def add_trail(self, run: Run, upto: int, v_scale: float) -> None:
        import cv2

        if upto <= self.trail_done:
            return
        xy = self.px(np.c_[run.pose["x"][self.trail_done:upto + 1], run.pose["y"][self.trail_done:upto + 1]])
        sp = np.abs(run.pose["speed"][self.trail_done:upto + 1])
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0, ::-1]
        for i in range(len(xy) - 1):
            p0, p1 = xy[i], xy[i + 1]
            if np.hypot(*(p1 - p0)) > 25:     # a reset teleports the car; don't draw a jump
                continue
            c = lut[int(np.clip(sp[i + 1] / v_scale, 0, 1) * 255)]
            cv2.line(self.trail, tuple(np.round(p0 * 4).astype(int)), tuple(np.round(p1 * 4).astype(int)),
                     tuple(int(x) for x in c), 2, cv2.LINE_AA, shift=2)
            cv2.line(self.trail_mask, tuple(np.round(p0 * 4).astype(int)), tuple(np.round(p1 * 4).astype(int)),
                     255, 2, cv2.LINE_AA, shift=2)
        self.trail_done = upto


class HUD:
    def __init__(self, run: Run, title: str):
        import cv2

        self.cv2 = cv2
        self.run = run
        self.title = title
        self.map = MapLayer(run, MAP_W, BOT_H)
        self.f_title = _font(17, bold=True)
        self.f_label = _font(13, bold=True)
        self.f_small = _font(12)
        self.f_mono = _font(13, mono=True)
        self.f_big = _font(34, bold=True)
        self.f_badge = _font(14, bold=True)
        self.cap = None
        self.cap_idx = -1
        self.cap_frame = None
        mp4 = os.path.join(run.dir, "screen.mp4")
        if run.video and os.path.isfile(mp4):
            self.cap = cv2.VideoCapture(mp4)
        self.ctrl_mode = str(run.config.get("control_mode", "controller"))

    # ---- panels ----------------------------------------------------------------------

    def _video_frame(self, t: float):
        if self.cap is None:
            return None
        k = _latest(self.run.video["t_mono"], t)
        if k < 0:
            return None
        while self.cap_idx < k:
            ok = self.cap.grab() if self.cap_idx < k - 1 else False
            self.cap_idx += 1
            if self.cap_idx == k:
                ok, f = self.cap.read() if not ok else self.cap.retrieve()
                if ok:
                    self.cap_frame = f[:, :, ::-1]
            elif not ok:
                break
        return self.cap_frame

    def _fit(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        cv2 = self.cv2
        ih, iw = img.shape[:2]
        s = min(w / iw, h / ih)
        r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out = np.full((h, w, 3), PANEL, np.uint8)
        y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
        out[y:y + r.shape[0], x:x + r.shape[1]] = r
        return out

    def _cover(self, img: np.ndarray, w: int, h: int) -> np.ndarray:
        """Scale to fill w x h, centre-cropping the overflow."""
        ih, iw = img.shape[:2]
        s = max(w / iw, h / ih)
        r = self.cv2.resize(img, (int(np.ceil(iw * s)), int(np.ceil(ih * s))), interpolation=self.cv2.INTER_LINEAR)
        y, x = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
        return r[y:y + h, x:x + w]

    def _camera(self, t: float, texts: list) -> np.ndarray:
        cv2, run = self.cv2, self.run
        i = _latest(run.camera["t_mono"], t) if run.camera else -1
        if i < 0:
            return np.full((TOP_H, 640, 3), PANEL, np.uint8)
        rgb = cv2.imread(os.path.join(run.dir, run.camera["rgb_png"][i]))[:, :, ::-1]
        j = _latest(run.seg["t_mono"], t) if run.seg else -1
        if j >= 0:
            m = cv2.imread(os.path.join(run.dir, run.seg["mask_png"][j]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
            tint = rgb.astype(np.float32)
            tint[m] = tint[m] * 0.55 + np.array(TRACK, np.float32) * 0.45
            rgb = tint.astype(np.uint8)
        out = self._fit(rgb, 640, TOP_H)
        if j >= 0:
            big = cv2.resize(m.astype(np.uint8) * 255, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, TRACK, 2, cv2.LINE_AA)
            age = (t - run.seg["t_mono"][j]) * 1e3
            frac = run.seg["track_fraction"][j] * 100
            texts.append((640 + 12, HEADER + TOP_H - 26, f"track {frac:4.1f} %   mask age {age:4.0f} ms",
                          self.f_mono, TEXT))
        texts.append((640 + 12, HEADER + 10, "CAR CAMERA  ·  live track segmentation (Fast-SCNN, ONNX)",
                      self.f_label, TEXT))
        return out

    def _sim(self, t: float, texts: list) -> np.ndarray:
        cv2, run = self.cv2, self.run
        f = self._video_frame(t)
        if f is not None:
            f = self._cover(f[SIM_UI_BAR:], 640, TOP_H)
            texts.append((12, HEADER + 10, "SIMULATOR  ·  AutoDRIVE RoboRacer chase view", self.f_label, TEXT))
            return f
        i = _latest(run.camera["t_mono"], t) if run.camera else -1
        if i >= 0 and "depth_png_mm" in run.camera:
            d = cv2.imread(os.path.join(run.dir, run.camera["depth_png_mm"][i]), cv2.IMREAD_UNCHANGED)
            d8 = np.clip(d.astype(np.float32) / 8000.0 * 255, 0, 255).astype(np.uint8)
            c = cv2.applyColorMap(255 - d8, cv2.COLORMAP_TURBO)[:, :, ::-1].copy()
            c[d == 0] = 0
            texts.append((12, HEADER + 10, "DEPTH  ·  0 – 8 m", self.f_label, TEXT))
            return self._fit(c, 640, TOP_H)
        return np.full((TOP_H, 640, 3), PANEL, np.uint8)

    def _map(self, t: float, texts: list) -> np.ndarray:
        cv2, run, m = self.cv2, self.run, self.map
        img = np.full((BOT_H, MAP_W, 3), PANEL, np.uint8)
        if run.pc is not None:
            k = _latest(run.pc["t_mono"], t)
            if k >= 0:
                m.add_walls(run, k)
            a = cv2.dilate(np.clip(m.walls / 2.0, 0, 1), np.ones((2, 2), np.uint8))[..., None] * 0.9
            img = (img * (1 - a) + np.array(WALL) * a).astype(np.uint8)
        b = _latest(run.boundary["t_mono"], t) if run.boundary is not None else -1
        if b >= 0:
            for key in sorted(run.boundary):
                if key.startswith(("outer", "inner")):
                    p = np.round(m.px(run.boundary[key]) * 4).astype(np.int32)
                    cv2.polylines(img, [p], True, BOUNDARY, 2, cv2.LINE_AA, shift=2)
        i = _latest(run.pose["t_mono"], t)
        if i >= 0:
            m.add_trail(run, i, run.v_scale)
            sel = m.trail_mask > 0
            img[sel] = m.trail[sel]
            x, y, yaw = run.pose["x"][i], run.pose["y"][i], run.pose["yaw"][i]
            if run.lidar is not None:
                li = _latest(run.lidar["t_mono"], t)
                if li >= 0:
                    r = run.lidar["ranges"][li]
                    ang = float(run.lidar["angle_min"]) + np.arange(len(r)) * float(run.lidar["angle_increment"])
                    ok = np.isfinite(r) & (r > 0.05)
                    wp = np.c_[x + r[ok] * np.cos(yaw + ang[ok]), y + r[ok] * np.sin(yaw + ang[ok])]
                    for u, v in m.px(wp[::2]).astype(int):
                        if 0 <= u < MAP_W and 0 <= v < BOT_H:
                            cv2.circle(img, (u, v), 1, SCAN, -1, cv2.LINE_AA)
            c = m.px(np.array([x, y]))[0]
            yaw = yaw + (np.pi / 2 if m.rot else 0.0)
            L = 9.0
            tri = np.array([[L, 0], [-L * 0.6, L * 0.55], [-L * 0.6, -L * 0.55]])
            R = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])  # y down in pixels
            pts = np.round((c + tri @ R) * 4).astype(np.int32)
            cv2.fillPoly(img, [pts], (255, 255, 255), cv2.LINE_AA, shift=2)
            cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA, shift=2)
        texts.append((12, HEADER + TOP_H + 10, "MAP  ·  LiDAR walls, trail coloured by speed", self.f_label, TEXT))
        # legend
        y0 = HEADER + TOP_H + BOT_H - 22
        x0 = 12
        for label, col in (("walls", WALL), ("scan", SCAN), ("boundary", BOUNDARY)):
            cv2.circle(img, (x0 + 5, BOT_H - 15), 4, col, -1, cv2.LINE_AA)
            texts.append((x0 + 14, y0, label, self.f_small, MUTED))
            x0 += 22 + len(label) * 7
        bx, bw = MAP_W - 132, 80
        lut = cv2.applyColorMap(np.linspace(0, 255, bw).astype(np.uint8)[None, :], cv2.COLORMAP_TURBO)[0, :, ::-1]
        img[BOT_H - 18:BOT_H - 11, bx:bx + bw] = lut
        texts.append((bx - 14, y0, "0", self.f_small, MUTED))
        texts.append((bx + bw + 5, y0, f"{run.v_scale:.1f} m/s", self.f_small, MUTED))
        cv2.line(img, (MAP_W - 1, 0), (MAP_W - 1, BOT_H), GRID, 1)
        return img

    def _bar(self, img, x, y, w, h, val_cmd, val_act, signed: bool):
        cv2 = self.cv2
        cv2.rectangle(img, (x, y), (x + w, y + h), GRID, -1)
        if signed:
            mid = x + w // 2
            if np.isfinite(val_act):
                e = int(mid + np.clip(val_act, -1, 1) * w / 2)
                cv2.rectangle(img, (min(mid, e), y), (max(mid, e), y + h), ACT, -1)
            cv2.line(img, (mid, y - 3), (mid, y + h + 3), MUTED, 1)
            if np.isfinite(val_cmd):
                c = int(mid + np.clip(val_cmd, -1, 1) * w / 2)
                cv2.line(img, (c, y - 4), (c, y + h + 4), CMD, 2)
        else:
            if np.isfinite(val_act):
                cv2.rectangle(img, (x, y), (x + int(np.clip(val_act, 0, 1) * w), y + h), SPEED, -1)

    def _chart(self, img, x, y, w, h, t, series, lo, hi, label, texts, window=10.0):
        cv2 = self.cv2
        cv2.rectangle(img, (x, y), (x + w, y + h), BG, -1)
        for f in (0.25, 0.5, 0.75):
            yy = int(y + h * f)
            cv2.line(img, (x, yy), (x + w, yy), GRID, 1)
        if lo < 0 < hi:
            yz = int(y + h * hi / (hi - lo))
            cv2.line(img, (x, yz), (x + w, yz), MUTED, 1)
        for ts, vs, col in series:
            if ts is None or not len(ts):
                continue
            a, b = np.searchsorted(ts, t - window), np.searchsorted(ts, t, side="right")
            if b - a < 2:
                continue
            px = x + (ts[a:b] - (t - window)) / window * w
            py = y + h - (np.clip(vs[a:b], lo, hi) - lo) / (hi - lo) * h
            p = np.round(np.c_[px, py] * 4).astype(np.int32)
            cv2.polylines(img, [p], False, col, 2, cv2.LINE_AA, shift=2)
        texts.append((MAP_W + x + 6, HEADER + TOP_H + y + 3, label, self.f_small, MUTED))

    def _telemetry(self, t: float, texts: list) -> np.ndarray:
        run = self.run
        cv2 = self.cv2
        w = W - MAP_W
        img = np.full((BOT_H, w, 3), PANEL, np.uint8)
        ox, oy = MAP_W, HEADER + TOP_H

        si = _latest(run.state["t_mono"], t) if run.state else -1
        ci = _latest(run.control["t_mono"], t) if run.control else -1
        ai = _latest(run.act["t_mono"], t) if run.act else -1
        v = run.state["v"][si] if si >= 0 else np.nan
        st_c = -_signed(run.control["steering_unit"][ci]) if ci >= 0 else np.nan   # + = left
        st_a = -_signed(run.act["steering_unit"][ai]) if ai >= 0 else np.nan
        th_c = _signed(run.control["throttle_unit"][ci]) if ci >= 0 else np.nan
        th_a = _signed(run.act["throttle_unit"][ai]) if ai >= 0 else np.nan

        # gauges column
        gx, gw = 16, 200
        texts.append((ox + gx, oy + 10, "SPEED", self.f_label, MUTED))
        texts.append((ox + gx, oy + 28, f"{v:4.2f}" if np.isfinite(v) else "–", self.f_big, TEXT))
        texts.append((ox + gx + 96, oy + 46, "m/s", self.f_small, MUTED))
        self._bar(img, gx, 74, gw, 8, np.nan, (v / run.v_scale) if np.isfinite(v) else np.nan, signed=False)

        texts.append((ox + gx, oy + 96, "STEERING", self.f_label, MUTED))
        texts.append((ox + gx + gw - 20, oy + 96, "L | R", self.f_small, MUTED))
        self._bar(img, gx, 118, gw, 10, -st_c, -st_a, signed=True)  # drawn left = left
        texts.append((ox + gx, oy + 136, "THROTTLE", self.f_label, MUTED))
        self._bar(img, gx, 158, gw, 10, th_c, th_a, signed=True)

        # legend
        cv2.line(img, (gx, 188), (gx + 14, 188), CMD, 2)
        texts.append((ox + gx + 18, oy + 181, "command", self.f_small, MUTED))
        cv2.line(img, (gx + 92, 188), (gx + 106, 188), ACT, 2)
        texts.append((ox + gx + 110, oy + 181, "after safety", self.f_small, MUTED))

        # laps
        done = [(tm, lt) for tm, lt in run.laps if tm <= t]
        texts.append((ox + gx, oy + 204, "LAPS", self.f_label, MUTED))
        lap_lines = [f"L{n + 1}  {lt:6.2f} s" + ("  map" if n == 0 and self._has_mapping() else "")
                     for n, (_, lt) in enumerate(done)][-2:]
        start = done[-1][0] if done else run.t_reset
        if start is not None and t >= start and len(done) < len(run.laps):
            lap_lines.append(f"L{len(done) + 1}  {t - start:6.2f} s  ▸")
        for n, s in enumerate(lap_lines or ["–"]):
            texts.append((ox + gx, oy + 222 + n * 17, s, self.f_mono, TEXT))
        coll = 0
        if run.sim:
            k = _latest(run.sim["t_rx"], t)
            coll = int(run.sim["collisions"][k]) if k >= 0 else 0
        texts.append((ox + gx, oy + BOT_H - 24, f"collisions {coll}", self.f_mono, SPEED if coll == 0 else SCAN))

        # strip charts
        cx, cw = gx + gw + 24, w - (gx + gw + 24) - 14
        ch, gap = 76, 10
        sv = (run.state["t_mono"], run.state["v"], SPEED) if run.state else (None, None, None)
        pv = (run.pose["t_mono"], np.abs(run.pose["speed"]), MUTED)
        self._chart(img, cx, 10, cw, ch, t, [pv, sv], 0, run.v_scale, "speed  m/s  (EKF; sim truth grey)", texts)
        ser = []
        if run.control:
            ser.append((run.control["t_mono"], -_signed(run.control["steering_unit"]), CMD))
        if run.act:
            ser.append((run.act["t_mono"], -_signed(run.act["steering_unit"]), ACT))
        self._chart(img, cx, 10 + ch + gap, cw, ch, t, ser, -1, 1, "steering  (+ left)", texts)
        ser = []
        if run.control:
            ser.append((run.control["t_mono"], _signed(run.control["throttle_unit"]), CMD))
        if run.act:
            ser.append((run.act["t_mono"], _signed(run.act["throttle_unit"]), ACT))
        th_hi = 0.15
        self._chart(img, cx, 10 + 2 * (ch + gap), cw, ch, t, ser, -th_hi, th_hi, "throttle  (unit, ±0.15 shown)", texts)
        texts.append((ox + cx, oy + BOT_H - 52, "-10 s", self.f_small, MUTED))
        texts.append((ox + cx + cw - 22, oy + BOT_H - 52, "now", self.f_small, MUTED))

        # data age
        ages = []
        for name, d in (("cam", run.camera), ("seg", run.seg), ("pose", run.pose), ("ekf", run.state),
                        ("ctrl", run.control), ("act", run.act)):
            if d:
                k = _latest(d["t_mono"], t)
                ages.append(f"{name} {(t - d['t_mono'][k]) * 1e3:3.0f}" if k >= 0 else f"{name}   –")
        texts.append((ox + cx, oy + BOT_H - 24, "age ms  " + "  ".join(ages), self.f_mono, MUTED))
        return img

    def _has_mapping(self) -> bool:
        return self.run.boundary is not None and not self.run.config.get("use_saved_map", True)

    def _phase(self, t: float) -> tuple[str, tuple]:
        run = self.run
        if run.t_reset is not None and t < run.t_reset:
            return "STANDBY", MUTED
        if self._has_mapping() and _latest(run.boundary["t_mono"], t) < 0:
            return "MAPPING LAP  ·  LiDAR follow-the-gap", BOUNDARY
        if run.laps and t > run.laps[-1][0]:
            return "FINISHED", MUTED
        return f"RACING  ·  {self.ctrl_mode}", SPEED

    # ---- frame -----------------------------------------------------------------------

    def frame(self, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw

        cv2 = self.cv2
        texts: list = []
        img = np.full((H, W, 3), BG, np.uint8)
        img[HEADER:HEADER + TOP_H, :640] = self._sim(t, texts)
        img[HEADER:HEADER + TOP_H, 640:] = self._camera(t, texts)
        img[HEADER + TOP_H:, :MAP_W] = self._map(t, texts)
        img[HEADER + TOP_H:, MAP_W:] = self._telemetry(t, texts)
        cv2.line(img, (640, HEADER), (640, HEADER + TOP_H), BG, 2)
        cv2.line(img, (0, HEADER + TOP_H), (W, HEADER + TOP_H), BG, 2)

        phase, pcol = self._phase(t)
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        d.text((14, 10), self.title, font=self.f_title, fill=TEXT)
        tw = d.textlength(phase, font=self.f_badge)
        bx = W // 2 + 60 - tw / 2
        d.rounded_rectangle((bx - 12, 8, bx + tw + 12, 32), radius=11, outline=pcol, width=2)
        d.text((bx, 11), phase, font=self.f_badge, fill=pcol)
        clock = f"t {t - self.run.t_first:6.2f} s"
        d.text((W - 14 - d.textlength(clock, font=self.f_mono), 12), clock, font=self.f_mono, fill=TEXT)
        for x, y, s, font, col in texts:
            # label backdrop over imagery so text stays readable
            if y < HEADER + TOP_H and font is not self.f_small:
                tw = d.textlength(s, font=font)
                d.rounded_rectangle((x - 6, y - 3, x + tw + 6, y + font.size + 5), radius=4, fill=(0, 0, 0, 160))
            d.text((x, y), s, font=font, fill=col)
        return np.asarray(pil)


def render(export: str, out: str, start: float | None, end: float | None, speed: float, fps: float,
           title: str, gif_width: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")
    run = Run(export)
    hud = HUD(run, title)
    t0, t1 = run.t_range()
    if start is not None:
        t0 = max(t0, run.t_first + start)
    if end is not None:
        t1 = min(t1, run.t_first + end)
    ticks = np.arange(t0, t1, speed / fps)

    gif = out.lower().endswith(".gif")
    tmpdir = tempfile.mkdtemp(prefix="hud_") if gif else None
    mp4 = os.path.join(tmpdir, "hud.mp4") if gif else out
    enc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "12" if gif else "18",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    try:
        for n, t in enumerate(ticks):
            enc.stdin.write(hud.frame(float(t)).tobytes())
            if n % 100 == 0:
                print(f"  frame {n}/{len(ticks)}  t_s {t - run.t_first:6.1f}", file=sys.stderr)
    finally:
        enc.stdin.close()
        enc.wait()
    if enc.returncode:
        raise SystemExit(f"ffmpeg failed ({enc.returncode})")
    if gif:
        pal = os.path.join(tmpdir, "palette.png")
        scale = f"scale={gif_width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4, "-vf",
                        f"{scale},palettegen=max_colors=128:stats_mode=diff", pal], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", mp4, "-i", pal, "-lavfi",
                        f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", out],
                       check=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
    print(f"wrote {out}  ({len(ticks)} frames, {os.path.getsize(out) / 1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="export directory from scripts/export_recording.py")
    ap.add_argument("--out", required=True, help="output .mp4 or .gif")
    ap.add_argument("--start", type=float, help="start time, seconds on t_s (default: run start)")
    ap.add_argument("--end", type=float, help="end time, seconds on t_s (default: run end)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1 = real time)")
    ap.add_argument("--fps", type=float, default=15.0, help="output frame rate (default 15)")
    ap.add_argument("--title", default="pilot_core  ·  AutoDRIVE RoboRacer", help="header title")
    ap.add_argument("--gif-width", type=int, default=960, help="width of .gif output (default 960)")
    a = ap.parse_args()
    render(a.export, a.out, a.start, a.end, a.speed, a.fps, a.title, a.gif_width)
    return 0


if __name__ == "__main__":
    sys.exit(main())
