#!/usr/bin/env python3
"""Minimum-curvature racing line for a saved map (control/raceline.py), for
track_control.raceline_path. Writes <map>/raceline.npy and a preview PNG, and prints the
speed planner's lap-time estimate on the centre-line and on the line.

    python scripts/make_raceline.py maps/autodrive_compete --margin 0.35
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from control.raceline import min_curvature_line  # noqa: E402
from control.speed import speed_profile  # noqa: E402
from control.track import TrackModel  # noqa: E402


def lap_estimate(track: TrackModel, a_lat: float, accel: float, decel: float, v_max: float) -> tuple:
    v = speed_profile(track, v_max, a_lat, accel, decel)
    ds = np.diff(np.concatenate([track.s, [track.length]]))
    return float((ds / v).sum()), v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir", help="directory with outer_line.npy / inner_line.npy")
    ap.add_argument("--margin", type=float, default=0.35, help="m kept from either wall")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--spacing", type=float, default=0.2)
    ap.add_argument("--out", help="default <map_dir>/raceline.npy")
    ap.add_argument("--a-lat", type=float, default=10.0, help="m/s^2, for the lap estimate only")
    ap.add_argument("--accel", type=float, default=5.0)
    ap.add_argument("--decel", type=float, default=5.0)
    ap.add_argument("--v-max", type=float, default=12.0)
    args = ap.parse_args()

    outer = np.load(os.path.join(args.map_dir, "outer_line.npy"))
    inner = np.load(os.path.join(args.map_dir, "inner_line.npy"))
    centre = TrackModel(outer, inner)
    line = min_curvature_line(centre, outer, inner, margin=args.margin, spacing=args.spacing,
                              passes=args.passes)
    raced = TrackModel(outer, inner)
    raced.set_path(line)
    out = args.out or os.path.join(args.map_dir, "raceline.npy")
    np.save(out, line)

    est = (args.a_lat, args.accel, args.decel, args.v_max)
    t_c, _ = lap_estimate(centre, *est)
    t_r, v_r = lap_estimate(raced, *est)
    print(f"centre-line {centre.length:.1f} m, max |kappa| {np.abs(centre.curvature).max():.2f} 1/m, est {t_c:.2f} s")
    print(f"racing line {raced.length:.1f} m, max |kappa| {np.abs(raced.curvature).max():.2f} 1/m, est {t_r:.2f} s, "
          f"closest wall {raced.half_width.min():.2f} m")
    print(f"wrote {out}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 9))
    for b in (outer, inner):
        ax.plot(b[:, 0], b[:, 1], "k-", lw=1)
    ax.plot(centre.points[:, 0], centre.points[:, 1], "b--", lw=0.8, label="centre-line")
    sc = ax.scatter(raced.points[:, 0], raced.points[:, 1], c=v_r, s=3, cmap="viridis")
    fig.colorbar(sc, ax=ax, label="planned speed (m/s)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title(f"{os.path.basename(os.path.normpath(args.map_dir))}: margin {args.margin} m")
    png = os.path.splitext(out)[0] + ".png"
    fig.savefig(png, dpi=90, bbox_inches="tight")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
