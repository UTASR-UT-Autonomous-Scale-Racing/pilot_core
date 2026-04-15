#!/usr/bin/env python3
"""
Closed-loop controller bench (no hardware): drive the track controllers on a
kinematic-bicycle model through the same estimator → controller → safety
chain used on the car, and print lap / tracking metrics.

    python scripts/sim_bench.py                          # all controllers, both tracks
    python scripts/sim_bench.py --mode stanley --track twisty --duration 90
    python scripts/sim_bench.py --v-max 1.2 --plot bench.png
    python scripts/sim_bench.py --mode depth_ftg            # map-free, depth images only
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    args = ap.parse_args()

    params = VehicleParams.load(args.params)
    speed_cfg = {"v_max": args.v_max} if args.v_max else None
    runs = []
    for track in args.track or sorted(TRACKS):
        outer, inner, _ = TRACKS[track]()
        for mode in args.mode or MODES:
            if mode == "depth_ftg":
                from control.depth_ftg import DepthGapFollower, Intrinsics

                # OAK-D RGB intrinsics scaled to the 160×100 rendered depth image
                K = Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
