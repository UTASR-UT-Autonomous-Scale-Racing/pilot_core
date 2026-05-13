#!/usr/bin/env python3
"""
Render the figures that show what a pipeline run did, straight from a
recording (start.py --recording). Used for the README images; works the same
on a recording from the car.

    python scripts/render_recording.py /tmp/pilot_recordings/<run> --out docs/images
    python scripts/render_recording.py <run> --out docs/images --truth config_dev_local.yaml

Figures (PNG):
  car_view.png     one frame: RGB, depth, drivable-track mask, depth Follow-the-Gap scan
  frames.png       frames sampled across the run (every --stride-th frame)
  map.png          bird's-eye view: SLAM map points, trajectory, extracted boundaries
  track.png        the mapped track as the controllers see it (centre line, speed
                   profile) and a simulated pure-pursuit drive on it
  timeline.png     commands and speed estimate over the run
  demo.gif         (--gif) the camera next to the map filling in, over the run

--truth <config> (synthetic runs only) reads the synthetic_track section of a
config, aligns the ground truth to the SLAM frame (each pose is matched to its
frame by capture timestamp) and adds it to map.png / track.png with error
numbers. It also prints a Markdown summary for the README.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from read_recording import load_buffer, summarize  # noqa: E402

# reference light palette (dataviz skill): first three categorical slots + neutrals
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#b9b8b3", "#e6e5e1", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#dcebfb", "#2a78d6", "#0b3b75"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8, "font.size": 9, "lines.linewidth": 2,
})


def _bev(points3d: np.ndarray) -> np.ndarray:
    """Camera/world (x right, y down, z forward) -> BEV (x, z)."""
    return np.asarray(points3d)[:, [0, 2]]


def _save(fig, out_dir: str, name: str) -> None:
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def _intrinsics(settings: str, w: int, h: int):
    from control.depth_ftg import Intrinsics

    K = Intrinsics.from_orbslam_settings(settings)
    sw, sh = w / (2 * K.cx), h / (2 * K.cy)       # settings are for the recorded size in practice
    return Intrinsics(K.fx * sw, K.fy * sh, K.cx * sw, K.cy * sh)


# ---------------------------------------------------------------- ground truth

class Truth:
    """Synthetic ground truth aligned to the SLAM BEV frame."""

    def __init__(self, config_path: str, frames, poses):
        import yaml
        from synthetic.closed_loop_generator import ClosedLoopFrameGenerator

        from control.types import resolve_path

        with open(resolve_path(config_path)) as f:
            cfg = yaml.safe_load(f)
        w, h = cfg.get("resolution", [640, 400])
        self.gen = ClosedLoopFrameGenerator(width=w, height=h, **cfg.get("synthetic_track", {}))

        # pose ts == its frame's capture ts; frame seq n shows generator frame n - 1
        frame_seq = {round(r.produced_ts, 6): r.seq_no for r in frames}
        slam, true = [], []
        for r in poses:
            seq = frame_seq.get(round(r.produced_ts, 6))
            if seq is not None:
                slam.append(r.value[0][[0, 2], 3])
                true.append(self.gen.pose_at(seq - 1)[[0, 2], 3])
        self.slam_xy, true_xy = np.array(slam), np.array(true)
        self.R, self.t = _rigid_2d(true_xy, self.slam_xy)       # truth -> SLAM frame
        self.ate = np.linalg.norm(self.to_slam(true_xy) - self.slam_xy, axis=1)
        outer, inner, path = self.gen.walls_xz(800)
        self.outer, self.inner, self.path = (self.to_slam(p) for p in (outer, inner, path))

    def to_slam(self, xy: np.ndarray) -> np.ndarray:
        return xy @ self.R.T + self.t

    def wall_error(self, outer: np.ndarray, inner: np.ndarray) -> tuple:
        """Distance (m) of each extracted boundary point to its true wall."""
        d = lambda pts, ref: np.min(np.linalg.norm(pts[:, None] - ref[None], axis=2), axis=1)
        return d(outer, self.outer), d(inner, self.inner)


def _rigid_2d(src: np.ndarray, dst: np.ndarray):
    """Least-squares rotation + translation mapping src onto dst (Umeyama, no scale)."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    D = np.diag([1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    return R, md - R @ ms


# ---------------------------------------------------------------- figures

def car_view(frames, segs, settings, out_dir, index=None):
    from control import VehicleParams
    from control.depth_ftg import DepthFTGConfig, DepthGapFollower

    rec = frames[len(frames) // 2 if index is None else index]
    rgb, depth = rec.value
    h, w = depth.shape
    mask = None
    if segs:
        seg = min(segs, key=lambda r: abs(r.produced_ts - rec.produced_ts))
        mask = seg.value

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax in axes.flat:
        ax.grid(False)
    ax = axes[0, 0]
    ax.imshow(rgb)
    ax.set_title("Camera (RGB)")
    ax.axis("off")

    ax = axes[0, 1]
    shown = np.where(depth > 0, depth, np.nan)
    im = ax.imshow(shown, cmap=SEQ, vmin=0, vmax=np.nanpercentile(shown, 98))
    ax.set_title("Depth (m), aligned to RGB")
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK2)

    ax = axes[1, 0]
    ax.imshow(rgb)
    if mask is not None:
        overlay = np.zeros((h, w, 4))
        overlay[mask > 0] = matplotlib.colors.to_rgba(AQUA, 0.55)
        ax.imshow(overlay)
        ax.set_title(f"Drivable track mask (vision, {100 * mask.mean():.0f}% of frame)")
    else:
        ax.set_title("Drivable track mask (vision not recorded)")
    ax.axis("off")

    ax = axes[1, 1]
    ax.grid(True)
    K = _intrinsics(settings, w, h)
    ftg_cfg = DepthFTGConfig(camera_height=0.15, camera_pitch=np.radians(12))
    follower = DepthGapFollower(VehicleParams.load(os.path.join(REPO_ROOT, "vehicle_params.yaml")), K, ftg_cfg)
    cmd = follower.step(depth)
    if follower.last_scan is not None:
        bearings, ranges = follower.last_scan
        x, y = ranges * np.sin(bearings), ranges * np.cos(bearings)   # x = left, y = forward
        ax.fill(np.r_[0, -x, 0], np.r_[0, y, 0], color=BLUE, alpha=0.12, lw=0)
        ax.plot(-x, y, color=BLUE, lw=2, label="virtual scan (nearest obstacle per bearing)")
        if follower.last_target is not None:
            phi, _ = follower.last_target
            r = ftg_cfg.aim_distance * 2
            ax.annotate("", xy=(-r * np.sin(phi), r * np.cos(phi)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.5))
            ax.text(-r * np.sin(phi), r * np.cos(phi) + 0.12, "gap heading", color=INK, ha="center", fontsize=8)
    ax.plot(0, 0, "o", color=INK, ms=8)
    ax.set_aspect("equal")
    ax.set_xlabel("right (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(f"depth_ftg: steering {cmd.steering:+.2f}, throttle {cmd.throttle:+.2f}")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32))
    fig.suptitle(f"What the car sees - frame {rec.seq_no}", color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "car_view.png")


def frames_sheet(frames, controls, out_dir, stride, t0, n=8):
    picks = frames[::stride][:n]
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, rec in zip(np.atleast_1d(axes).flat, picks):
        ax.imshow(rec.value[0])
        label = f"t={rec.produced_ts - t0:.1f} s  frame {rec.seq_no}"
        if controls:
            c = min(controls, key=lambda r: abs(r.produced_ts - rec.produced_ts))
            if abs(c.produced_ts - rec.produced_ts) < 0.5:
                from workers.control_worker import unit_to_signed
                s, th = unit_to_signed(*c.value)
                label += f"\nsteer {s:+.2f}, throttle {th:+.2f}"
            else:
                label += "\n(no command yet: mapping)"
        ax.set_title(label, fontsize=8, color=INK2, loc="left")
    fig.suptitle(f"Recorded frames, every {stride}th ({len(frames)} recorded)", color=INK, fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "frames.png")


def map_figure(poses, clouds, boundaries, truth, out_dir):
    fig, ax = plt.subplots(figsize=(9, 6.4))
    if clouds:
        pts = _bev(clouds[-1].value)
        keep = np.random.default_rng(0).choice(len(pts), min(len(pts), 6000), replace=False)
        ax.scatter(pts[keep, 0], pts[keep, 1], s=2, color=MUTED, lw=0, label=f"SLAM map points ({len(pts)})")
    if truth is not None:
        for wall in (truth.outer, truth.inner):
            c = np.vstack([wall, wall[:1]])
            ax.plot(c[:, 0], c[:, 1], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.plot([], [], color=INK2, lw=1, ls=(0, (4, 3)), label="true walls")
    traj = np.array([r.value[0][[0, 2], 3] for r in poses]) if poses else np.zeros((0, 2))
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 1], color=AQUA, lw=2, label="SLAM trajectory")
        ax.text(traj[0, 0] + 0.1, traj[0, 1] - 0.25, "start", color=INK, fontsize=8)
        ax.plot(*traj[0], "o", color=AQUA, ms=8, mec=SURFACE, mew=2)
    if boundaries:
        outer, inner = boundaries[-1].value
        for line, color, name in ((outer, BLUE, "outer boundary"), (inner, ORANGE, "inner boundary")):
            c = np.vstack([line, line[:1]])
            ax.plot(c[:, 0], c[:, 1], color=color, lw=2.2, label=f"extracted {name}")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m), forward at start")
    title = "Track mapping: ORB-SLAM3 map, driven lap and extracted boundaries"
    if truth is not None and boundaries:
        eo, ei = truth.wall_error(*boundaries[-1].value)
        title += (f"\ntrajectory error vs truth: RMS {100 * np.sqrt(np.mean(truth.ate ** 2)):.1f} cm; "
                  f"boundary error: median {100 * np.median(np.r_[eo, ei]):.1f} cm, "
                  f"95th pct {100 * np.percentile(np.r_[eo, ei], 95):.1f} cm")
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, out_dir, "map.png")


def track_figure(boundaries, truth, out_dir):
    from control import VehicleParams
    from control.controllers import build_track_follower
    from control.sim import make_follower_factory, run_closed_loop
    from control.track import TrackModel

    outer, inner = (np.asarray(b, dtype=np.float64) for b in boundaries[-1].value)
    params = VehicleParams.load(os.path.join(REPO_ROOT, "vehicle_params.yaml"))
    track = TrackModel(outer, inner)
    follower = build_track_follower("pure_pursuit", track, params)
    v_profile = follower.v_profile
    result = run_closed_loop(make_follower_factory("pure_pursuit", params), outer, inner, params, duration=45.0)
    log = result.log

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    ring = np.vstack([outer, outer[:1]])
    hole = np.vstack([inner, inner[:1]])[::-1]
    ax.fill(np.r_[ring[:, 0], hole[:, 0]], np.r_[ring[:, 1], hole[:, 1]], color=GRID, lw=0)
    ax.plot(ring[:, 0], ring[:, 1], color=INK2, lw=1)
    ax.plot(hole[:, 0], hole[:, 1], color=INK2, lw=1)
    sc = ax.scatter(track.points[:, 0], track.points[:, 1], c=v_profile, cmap=SEQ, s=6, lw=0)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("speed target (m/s)")
    cb.outline.set_visible(False)
    ax.set_aspect("equal")
    ax.set_title(f"Mapped track model\n{track.length:.1f} m centre line, coloured by speed target", loc="left")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")

    ax = axes[1]
    ax.fill(np.r_[ring[:, 0], hole[:, 0]], np.r_[ring[:, 1], hole[:, 1]], color=GRID, lw=0)
    if truth is not None:
        for wall in (truth.outer, truth.inner):
            c = np.vstack([wall, wall[:1]])
            ax.plot(c[:, 0], c[:, 1], color=INK2, lw=1, ls=(0, (4, 3)))
        ax.plot([], [], color=INK2, lw=1, ls=(0, (4, 3)), label="true walls")
    ax.plot(log[:, 1], log[:, 2], color=BLUE, lw=1.6, label="simulated car (pure_pursuit)")
    ax.set_aspect("equal")
    title = (f"Simulated drive on the mapped track ({result.laps:.1f} laps)\n"
             f"cross-track RMS {100 * result.xtrack_rms:.1f} cm, mean {result.mean_speed:.2f} m/s")
    if truth is not None:
        from matplotlib.path import Path

        inside = Path(truth.outer).contains_points(log[:, 1:3]) & ~Path(truth.inner).contains_points(log[:, 1:3])
        title += f", inside the true walls {100 * inside.mean():.0f}% of the time"
    ax.set_title(title, loc="left")
    ax.set_xlabel("x (m)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, out_dir, "track.png")
    return result


def timeline(controls, states, boundaries, out_dir, t0):
    from workers.control_worker import unit_to_signed

    fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
    if controls:
        t = np.array([r.produced_ts - t0 for r in controls])
        sig = np.array([unit_to_signed(*r.value) for r in controls])
        axes[0].plot(t, sig[:, 0], color=BLUE, lw=1.4)
        axes[0].set_ylabel("steering\n(+1 = left)")
        axes[1].plot(t, sig[:, 1], color=BLUE, lw=1.4)
        axes[1].set_ylabel("throttle\n(+1 = forward)")
    if states:
        t = np.array([r.produced_ts - t0 for r in states])
        axes[2].plot(t, [r.value.v for r in states], color=BLUE, lw=1.4)
        axes[2].set_ylabel("speed estimate\n(m/s, EKF)")
    for ax in axes:
        if boundaries:
            tb = boundaries[-1].produced_ts - t0
            ax.axvline(tb, color=INK2, lw=1, ls=(0, (4, 3)))
    if boundaries:
        axes[0].text(tb, axes[0].get_ylim()[1], "  map extracted", color=INK2, va="top", fontsize=8)
    axes[-1].set_xlabel("time since recording start (s)")
    axes[0].set_title("Controller commands (ctrl_buf) and state estimate (state_buf)", loc="left")
    fig.tight_layout()
    _save(fig, out_dir, "timeline.png")


def demo_gif(frames, poses, clouds, boundaries, controls, out_dir, t0, n=60, fps=6):
    """Animated side-by-side: the camera frame and the bird's-eye map filling in
    as the car drives (map points, trajectory so far, boundaries once extracted)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image

    from workers.control_worker import unit_to_signed

    pose_ts = np.array([r.produced_ts for r in poses])
    traj = np.array([r.value[0][[0, 2], 3] for r in poses])
    fwd = np.array([r.value[0][[0, 2], 2] for r in poses])
    cloud_ts = np.array([r.produced_ts for r in clouds]) if clouds else np.array([])
    bnd_t = boundaries[-1].produced_ts if boundaries else np.inf
    ctrl_ts = np.array([r.produced_ts for r in controls]) if controls else np.array([])
    lo = np.min(traj, axis=0) - 1.5
    hi = np.max(traj, axis=0) + 1.5

    images = []
    for rec in frames[:: max(1, len(frames) // n)]:
        t = rec.produced_ts
        fig = plt.figure(figsize=(8.4, 3.3), dpi=80)
        FigureCanvasAgg(fig)
        a1 = fig.add_axes([0.0, 0.0, 0.58, 0.86])
        a1.imshow(rec.value[0])
        a1.axis("off")
        a2 = fig.add_axes([0.63, 0.08, 0.35, 0.8])
        a2.set_xlim(lo[0], hi[0])
        a2.set_ylim(lo[1], hi[1])
        a2.set_aspect("equal")
        a2.tick_params(labelsize=6)
        k = np.searchsorted(cloud_ts, t) - 1
        if k >= 0:
            pts = _bev(clouds[k].value)[::4]
            a2.scatter(pts[:, 0], pts[:, 1], s=0.5, color=MUTED, lw=0)
        i = np.searchsorted(pose_ts, t)
        if i > 0:
            a2.plot(traj[:i, 0], traj[:i, 1], color=AQUA, lw=1.5)
            a2.annotate("", xy=traj[i - 1] + 0.8 * fwd[i - 1] / max(np.linalg.norm(fwd[i - 1]), 1e-6),
                        xytext=traj[i - 1], arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
        if t >= bnd_t:
            for line, color in zip(boundaries[-1].value, (BLUE, ORANGE)):
                c = np.vstack([line, line[:1]])
                a2.plot(c[:, 0], c[:, 1], color=color, lw=1.5)
        phase = "driving on the mapped track" if t >= bnd_t else "mapping lap"
        caption = f"t = {t - t0:5.1f} s   {phase}"
        j = np.searchsorted(ctrl_ts, t) - 1
        if j >= 0 and t - ctrl_ts[j] < 0.5:
            st, th = unit_to_signed(*controls[j].value)
            caption += f"   steer {st:+.2f}  throttle {th:+.2f}"
        fig.text(0.01, 0.93, caption, fontsize=9, color=INK, fontweight="bold")
        fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    path = os.path.join(out_dir, "demo.gif")
    frames_p = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for im in images]
    frames_p[0].save(path, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0,
                     optimize=True)
    print(f"wrote {path} ({len(images)} frames, {os.path.getsize(path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("recording", help="recording directory")
    ap.add_argument("--out", default="docs/images", help="output directory (default docs/images)")
    ap.add_argument("--truth", help="config with a synthetic_track section (synthetic runs only)")
    ap.add_argument("--settings", default=os.path.join(REPO_ROOT, "synthetic_slam_settings.yaml"),
                    help="ORB-SLAM3 settings with the camera intrinsics (for the depth_ftg scan)")
    ap.add_argument("--stride", type=int,
                    help="frame sampling for frames.png (default: 8 frames spread over the run)")
    ap.add_argument("--frame", type=int, help="index of the frame for car_view.png (default: middle)")
    ap.add_argument("--gif", action="store_true", help="also write demo.gif (camera + map, animated)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    load = lambda name: load_buffer(args.recording, name) if _has(args.recording, name) else []
    frames, segs, poses = load("frame"), load("segmentation"), load("pose")
    clouds, boundaries, controls, states = load("pointcloud"), load("boundary"), load("control"), load("state")

    truth = Truth(args.truth, frames, poses) if args.truth and poses else None
    # one time origin for every figure: the first recorded sample of any buffer
    t0 = min(b[0].produced_ts for b in (frames, segs, poses, clouds, boundaries, controls, states) if b)
    if frames:
        car_view(frames, segs, args.settings, args.out, args.frame)
        frames_sheet(frames, controls, args.out, args.stride or max(1, len(frames) // 8), t0)
    if poses or clouds:
        map_figure(poses, clouds, boundaries, truth, args.out)
    result = track_figure(boundaries, truth, args.out) if boundaries else None
    if controls or states:
        timeline(controls, states, boundaries, args.out, t0)
    if args.gif and frames and poses:
        demo_gif(frames, poses, clouds, boundaries, controls, args.out, t0)

    print("\n| buffer | records | rate (Hz) | size (MB) |\n| ------ | ------- | --------- | --------- |")
    for name, s in sorted(summarize(args.recording).items()):
        span = (s["last_ts"] - s["first_ts"]) if s["records"] > 1 else 0.0
        rate = (s["records"] - 1) / span if span > 0 else 0.0
        print(f"| `{name}` | {s['records']} | {rate:.1f} | {s['bytes'] / 1e6:.1f} |")
    if truth is not None:
        print(f"\ntrajectory error RMS {100 * np.sqrt(np.mean(truth.ate ** 2)):.1f} cm over {len(truth.ate)} poses")
        if boundaries:
            eo, ei = truth.wall_error(*boundaries[-1].value)
            print(f"boundary error median {100 * np.median(np.r_[eo, ei]):.1f} cm, "
                  f"95th pct {100 * np.percentile(np.r_[eo, ei], 95):.1f} cm, max {100 * np.max(np.r_[eo, ei]):.1f} cm")
    if result is not None:
        print(f"closed loop on the mapped track: {result.summary()}")
    return 0


def _has(recording: str, name: str) -> bool:
    from read_recording import list_files

    return bool(list_files(recording, name))


if __name__ == "__main__":
    sys.exit(main())
