#!/usr/bin/env python3
"""Screen capture of the sim window on the Xvfb display (ffmpeg x11grab) and GIF cutting.

    python scripts/autodrive_capture.py start out.mp4           # records until stop
    python scripts/autodrive_capture.py stop out.mp4
    python scripts/autodrive_capture.py gif out.mp4 clip.gif --start 12.5 --end 40 [--width 640 --fps 10]

start writes out.mp4.json with the recording's start time on time.monotonic() and the
wall clock, so clips can be cut from pilot_core timestamps (all on time.monotonic()):
    clip_start = t_event - meta["t0_monotonic"]
GIFs are capped at --max-mb (15) by lowering fps, then width.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

DISPLAY = os.environ.get("DISPLAY", ":99")


def _window_geometry() -> tuple[int, int, int, int]:
    env = {**os.environ, "DISPLAY": DISPLAY}
    ids = subprocess.run(["xdotool", "search", "--class", "AutoDRIVE"], env=env,
                         capture_output=True, text=True).stdout.split()
    if not ids:
        raise SystemExit("AutoDRIVE window not found")
    info = subprocess.run(["xwininfo", "-id", ids[-1]], env=env, capture_output=True, text=True,
                          check=True).stdout
    g = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in info.splitlines() if ":" in line)}
    w, h = int(g["Width"]) // 2 * 2, int(g["Height"]) // 2 * 2
    return int(g["Absolute upper-left X"]), int(g["Absolute upper-left Y"]), w, h


def start(out: str, fps: int = 15) -> dict:
    x, y, w, h = _window_geometry()
    log = open(out + ".ffmpeg.log", "w")
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "warning", "-y", "-f", "x11grab", "-framerate", str(fps),
         "-video_size", f"{w}x{h}", "-i", f"{DISPLAY}.0+{x},{y}",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", out],
        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    meta = {"pid": proc.pid, "t0_monotonic": time.monotonic(), "t0_wall": time.time(),
            "geometry": [x, y, w, h], "fps": fps}
    with open(out + ".json", "w") as f:
        json.dump(meta, f)
    return meta


def stop(out: str) -> None:
    with open(out + ".json") as f:
        meta = json.load(f)
    try:
        os.kill(meta["pid"], signal.SIGINT)      # ffmpeg finalises the file on SIGINT
    except ProcessLookupError:
        return
    for _ in range(100):
        try:
            os.kill(meta["pid"], 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _captions(labels, t_start: float) -> str:
    """[(t0, t1, text), ...] in source-video seconds -> drawtext filters (clip time)."""
    out = []
    for a, b, text in labels or []:
        text = text.replace("\\", "").replace("'", "").replace(":", "\\:").replace(",", "\\,")
        out.append(f"drawtext=fontfile={FONT}:text='{text}':x=8:y=h-th-8:fontsize=16:fontcolor=white:"
                   f"box=1:boxcolor=black@0.6:boxborderw=5:enable='between(t,{a - t_start:.2f},{b - t_start:.2f})'")
    return ",".join(out) + "," if out else ""


def gif(src: str, dst: str, t_start: float = 0.0, t_end: float | None = None, width: int = 640,
        fps: int = 10, max_mb: float = 15.0, speed: float = 1.0, labels=None) -> str:
    """Cut [t_start, t_end] s of src into a palette-optimised GIF, at most max_mb.
    labels: [(t0, t1, text)] captions in source seconds."""
    caps = _captions(labels, t_start)
    while True:
        dur = [] if t_end is None else ["-t", f"{(t_end - t_start):.2f}"]
        pts = f"setpts=PTS/{speed}," if speed != 1.0 else ""
        vf = (f"{caps}{pts}fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
              f"[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t_start:.2f}", *dur, "-i", src,
                        "-vf", vf, "-loop", "0", dst], check=True)
        mb = os.path.getsize(dst) / 1e6
        if mb <= max_mb:
            print(f"wrote {dst} ({mb:.1f} MB, {width} px, {fps} fps)")
            return dst
        if fps > 6:
            fps -= 2
        elif width > 320:
            width = int(width * 0.8) // 2 * 2
        else:
            raise SystemExit(f"{dst} still {mb:.1f} MB at {width} px / {fps} fps; cut a shorter clip")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("out")
    s.add_argument("--fps", type=int, default=15)
    sub.add_parser("stop").add_argument("out")
    g = sub.add_parser("gif")
    g.add_argument("src")
    g.add_argument("dst")
    g.add_argument("--start", type=float, default=0.0)
    g.add_argument("--end", type=float)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--fps", type=int, default=10)
    g.add_argument("--speed", type=float, default=1.0, help="playback speed-up")
    g.add_argument("--max-mb", type=float, default=15.0)
    a = ap.parse_args()
    if a.cmd == "start":
        print(json.dumps(start(a.out, a.fps)))
    elif a.cmd == "stop":
        stop(a.out)
    else:
        gif(a.src, a.dst, a.start, a.end, a.width, a.fps, a.max_mb, a.speed)


if __name__ == "__main__":
    sys.exit(main())
