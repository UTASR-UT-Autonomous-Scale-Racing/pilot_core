#!/usr/bin/env python3
"""
Closed-loop controller bench, no hardware needed. Runs the track controllers
on a kinematic bicycle through the same estimator -> controller -> safety chain
as the car and prints lap / tracking numbers.

Usage:
    python scripts/sim_bench.py                          # all controllers, both tracks
    python scripts/sim_bench.py --summary docs/images/controllers.png   # comparison chart
    python scripts/sim_bench.py --mode stanley --track twisty --duration 90
    python scripts/sim_bench.py --v-max 1.2 --plot bench.png
    python scripts/sim_bench.py --mode depth_ftg            # map-free, depth only
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from control import VehicleParams  # noqa: E402
from control.controllers import LATERAL_CONTROLLERS  # noqa: E402
from control.sim import (  # noqa: E402
    make_follower_factory, oval_track, run_closed_loop, run_depth_closed_loop, twisty_track,
)

MODES = sorted(LATERAL_CONTROLLERS) + ["depth_ftg"]

TRACKS = {"oval": oval_track, "twisty": twisty_track}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=MODES, action="append",
                    help="controller(s) to run (default: all)")
    ap.add_argument("--track", choices=sorted(TRACKS), action="append", help="track(s) (default: all)")
    ap.add_argument("--duration", type=float, default=60.0, help="simulated seconds per run")
    ap.add_argument("--v-max", type=float, default=None, help="planner speed limit (m/s)")
    ap.add_argument("--params", default="vehicle_params.yaml", help="vehicle parameter file")
    ap.add_argument("--plot", metavar="PNG", help="save trajectories to this image")
    ap.add_argument("--summary", metavar="PNG", help="save a cross-track error comparison chart")
    args = ap.parse_args()

    params = VehicleParams.load(args.params)
    speed_cfg = {"v_max": args.v_max} if args.v_max else None
    runs = []
    for track in args.track or sorted(TRACKS):
        outer, inner, _ = TRACKS[track]()
        for mode in args.mode or MODES:
            if mode == "depth_ftg":
                from control.depth_ftg import DepthGapFollower, Intrinsics

                # the rendered depth image is 160x100, so quarter the OAK-D intrinsics
                K = Intrinsics.from_orbslam_settings(os.path.join(REPO_ROOT, "orbslam_oakd_rgbd.yaml"))
                K = Intrinsics(K.fx / 4, K.fy / 4, K.cx / 4, K.cy / 4)
                r = run_depth_closed_loop(DepthGapFollower(params, K), outer, inner, params,
                                          duration=args.duration)
            else:
                r = run_closed_loop(make_follower_factory(mode, params, speed_cfg=speed_cfg),
                                    outer, inner, params, duration=args.duration)
            print(f"{track:7s} {mode:13s} {r.summary()}")
            runs.append((track, mode, outer, inner, r))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(runs), figsize=(5 * len(runs), 5), squeeze=False)
        for ax, (track, mode, outer, inner, r) in zip(axes[0], runs):
            ax.plot(outer[:, 0], outer[:, 1], "k-", lw=1)
            ax.plot(inner[:, 0], inner[:, 1], "k-", lw=1)
            sc = ax.scatter(r.log[:, 1], r.log[:, 2], c=r.log[:, 3], s=1, cmap="viridis")
            ax.set_title(f"{track} / {mode}\nxtrack max {r.xtrack_max:.2f} m")
            ax.set_aspect("equal")
            fig.colorbar(sc, ax=ax, label="speed (m/s)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"Saved {args.plot}")
    if args.summary:
        summary_chart(runs, args.summary)
    return 0


def summary_chart(runs, path: str) -> None:
    """Cross-track error per controller, one panel per track (RMS bar, max dot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, ink, ink2, grid, surface = "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
    tracks = sorted({t for t, *_ in runs})
    fig, axes = plt.subplots(1, len(tracks), figsize=(5.5 * len(tracks), 3.6), squeeze=False,
                             facecolor=surface)
    for ax, track in zip(axes[0], tracks):
        rows = sorted(((m, r) for t, m, _, _, r in runs if t == track), key=lambda x: x[1].xtrack_rms)
        names = [m for m, _ in rows]
        rms = [100 * r.xtrack_rms for _, r in rows]
        mx = [100 * r.xtrack_max for _, r in rows]
        y = range(len(rows))
        ax.set_facecolor(surface)
        ax.barh(y, rms, height=0.55, color=blue, label="RMS")
        ax.scatter(mx, y, s=40, color=ink, zorder=3, label="max", edgecolor=surface, linewidth=1.5)
        for yi, (r_, m_) in enumerate(zip(rms, mx)):
            ax.text(m_ + 0.4, yi, f"{r_:.1f} / {m_:.1f}", va="center", fontsize=8, color=ink2)
        ax.set_yticks(list(y), names, color=ink)
        ax.invert_yaxis()
        ax.set_xlim(0, max(mx) * 1.35)
        ax.set_xlabel("cross-track error (cm)", color=ink2)
        ax.set_title(f"{track} track", loc="left", color=ink, fontweight="bold")
        ax.grid(axis="x", color=grid)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, [f"{l} (cm)" for l in labels], frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.suptitle("Closed-loop controller bench (bicycle model, estimator + safety supervisor in the loop)",
                 color=ink, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=surface)
    print(f"Saved {path}")


if __name__ == "__main__":
    sys.exit(main())
