#!/usr/bin/env python3
"""Drive the stock AutoDRIVE sim's HUD on an X display (Xvfb) with xdotool.

The 2026-iros build has no command-line connect option: the left menu holds the
IP/port fields and a Connected/Disconnected toggle, so this clicks it.

    python scripts/autodrive_sim_ctl.py state            # connected | disconnected
    python scripts/autodrive_sim_ctl.py connect          # toggle until connected
    python scripts/autodrive_sim_ctl.py disconnect
    python scripts/autodrive_sim_ctl.py reconnect        # off, then on (after a server restart)
    python scripts/autodrive_sim_ctl.py menu             # open the left menu if it is closed
    python scripts/autodrive_sim_ctl.py reset|quality    # HUD Reset / quality toggle

"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

DISPLAY = os.environ.get("DISPLAY", ":99")
# HUD layout in window pixels at the 1280x720 reference size; the Unity canvas scales
# with the window, so everything is rescaled to the real window geometry.
REF_W, REF_H = 1280, 720
MENU_BTN = (49, 30)            # hamburger, top left
CONNECT_BTN = (149, 275)       # Connected / Disconnected label
RESET_BTN = (119, 605)         # Reset (respawn at the start line)
QUALITY_BTN = (149, 473)       # Low / High Quality toggle
# "Disconnected" is ~40 px wider than "Connected": white text in this box = disconnected
LABEL_TAIL = (231, 265, 261, 285)      # x0, y0, x1, y1
MENU_PROBE = (27, 80, 159, 98)         # "IP Address" heading


def _env() -> dict:
    return {**os.environ, "DISPLAY": DISPLAY}


def _window() -> tuple[str, int, int, int, int]:
    """(id, x, y, w, h) of the sim window in root coordinates."""
    ids = subprocess.run(["xdotool", "search", "--class", "AutoDRIVE"], env=_env(),
                         capture_output=True, text=True).stdout.split()
    if not ids:
        raise RuntimeError("AutoDRIVE window not found on " + DISPLAY)
    wid = ids[-1]
    # xwininfo, not xdotool getwindowgeometry: the latter double-counts the WM frame
    info = subprocess.run(["xwininfo", "-id", wid], env=_env(),
                          capture_output=True, text=True, check=True).stdout
    g = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in info.splitlines() if ":" in line)}
    return (wid, int(g["Absolute upper-left X"]), int(g["Absolute upper-left Y"]),
            int(g["Width"]), int(g["Height"]))


def _to_root(x: float, y: float) -> tuple[int, int]:
    _, wx, wy, w, h = _window()
    return int(round(wx + x * w / REF_W)), int(round(wy + y * h / REF_H))


def _grab() -> np.ndarray:
    raw = subprocess.run(
        ["import", "-display", DISPLAY, "-window", "root", "-depth", "8", "rgb:-"],
        check=True, capture_output=True).stdout
    size = subprocess.run(["xdotool", "getdisplaygeometry"], env=_env(), capture_output=True,
                          text=True, check=True).stdout.split()
    return np.frombuffer(raw, np.uint8).reshape(int(size[1]), int(size[0]), 3)


def _white(img: np.ndarray, box) -> int:
    x0, y0 = _to_root(box[0], box[1])
    x1, y1 = _to_root(box[2], box[3])
    return int((img[y0:y1, x0:x1].min(axis=2) > 200).sum())


def _click(xy) -> None:
    wid = _window()[0]
    subprocess.run(["xdotool", "windowactivate", "--sync", wid], env=_env(), capture_output=True)
    rx, ry = _to_root(*xy)
    # Unity samples the pointer once per frame: move, press and release in separate frames
    subprocess.run(["xdotool", "mousemove", str(rx), str(ry)], env=_env(), check=True)
    time.sleep(0.2)
    subprocess.run(["xdotool", "mousedown", "1"], env=_env(), check=True)
    time.sleep(0.15)
    subprocess.run(["xdotool", "mouseup", "1"], env=_env(), check=True)
    time.sleep(0.2)
    px, py = _to_root(REF_W - 10, REF_H - 10)
    subprocess.run(["xdotool", "mousemove", str(px), str(py)], env=_env())   # park off the HUD


def menu_open(img=None) -> bool:
    img = _grab() if img is None else img
    # threshold scales with the box area
    _, _, _, w, h = _window()
    return _white(img, MENU_PROBE) > 50 * (w * h) / (REF_W * REF_H)


def ensure_menu() -> None:
    if not menu_open():
        _click(MENU_BTN)
        time.sleep(1.0)
    if not menu_open():
        raise RuntimeError("could not open the sim menu")


def state() -> str:
    ensure_menu()
    _, _, _, w, h = _window()
    white = _white(_grab(), LABEL_TAIL)
    return "disconnected" if white > 20 * (w * h) / (REF_W * REF_H) else "connected"


def set_state(target: str, tries: int = 4) -> str:
    for _ in range(tries):
        s = state()
        if s == target:
            return s
        _click(CONNECT_BTN)
        time.sleep(1.0)
    return state()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["state", "connect", "disconnect", "reconnect", "menu", "reset", "quality"])
    a = ap.parse_args()
    if a.cmd in ("reset", "quality"):
        ensure_menu()
        _click(RESET_BTN if a.cmd == "reset" else QUALITY_BTN)
        print(a.cmd + " clicked")
    elif a.cmd == "menu":
        ensure_menu()
        print("menu open")
    elif a.cmd == "state":
        print(state())
    elif a.cmd == "reconnect":
        set_state("disconnected")
        time.sleep(0.5)
        print(set_state("connected"))
    else:
        target = "connected" if a.cmd == "connect" else "disconnected"
        s = set_state(target)
        print(s)
        sys.exit(0 if s == target else 1)


if __name__ == "__main__":
    main()
