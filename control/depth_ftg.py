"""
Map-free reactive control from the depth image: Follow-the-Gap on a virtual
laser scan synthesised from the depth camera.

Reference: V. Sezer, M. Gokasan, "A novel obstacle avoidance algorithm:
'Follow the Gap Method'", Robotics and Autonomous Systems 60(9), 2012.

Pipeline:
    1. Back-project depth pixels (pinhole intrinsics) into the camera frame
       (x right, y down, z forward) and keep points in an obstacle height
       band above the ground (removes floor and overhead clutter).
    2. Virtual scan: nearest obstacle range per bearing bin (+ bearing = left).
    3. Follow the Gap: safety bubble around the nearest obstacle → best
       free gap (largest visible free area) → heading angle.  As in the
       paper, the gap-centre angle is blended with a goal angle, weighted by
       the nearest-obstacle distance:
       ``phi = (alpha/d_min * phi_gap + beta * phi_goal) / (alpha/d_min + beta)``
       (goal = straight ahead; ``beta = 0`` gives pure gap-centre steering).
       The F1TENTH "deepest point" variant is available via
       ``strategy="deepest"``; on a narrow-FOV camera it tends to hug walls.
    4. Steer at ``phi`` with the pure-pursuit curvature law over a short aim
       distance; speed falls with steering and the car stops close to an
       obstacle.

Assumes a level, forward-facing camera at ``camera_height`` above ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .types import Command, VehicleParams


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_orbslam_settings(cls, path: str) -> "Intrinsics":
        """Read ``Camera1.fx/fy/cx/cy`` from an ORB-SLAM3 settings file."""
        vals = {}
        with open(path) as f:
            for line in f:
                for key in ("fx", "fy", "cx", "cy"):
                    if line.strip().startswith(f"Camera1.{key}:") or line.strip().startswith(f"Camera.{key}:"):
                        vals.setdefault(key, float(line.split(":", 1)[1]))
        return cls(**vals)


@dataclass
class DepthFTGConfig:
    camera_height: float = 0.12       # m above ground (measure on the car)
    min_obstacle_height: float = 0.04  # m — below this is floor
    max_obstacle_height: float = 0.50  # m — above this is ignored
    max_range: float = 4.0            # m
    num_bins: int = 64                # bearing bins across the FOV
    pixel_step: int = 4               # subsample the depth image
    smooth_window: int = 3
    bubble_radius: float = 0.25       # m — car half-width + margin around the nearest obstacle
    min_gap_range: float = 0.6        # m — bins closer than this are not "free"
    strategy: str = "centre"          # "centre" (Sezer & Gokasan) or "deepest"
    alpha: float = 1.0                # gap-angle weight (scaled by 1/d_min)
    beta: float = 0.0                 # goal-angle weight (goal = straight ahead)
    aim_distance: float = 0.7         # m — pure-pursuit look-ahead towards the chosen angle
    cruise_speed: float = 0.6         # m/s
    stop_distance: float = 0.35       # m — obstacle this close ahead → neutral
    speed_per_throttle: float = 3.0   # m/s per unit throttle (same calibration as track control)


def depth_to_scan(
    depth: np.ndarray, K: Intrinsics, cfg: DepthFTGConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Virtual scan from a depth image.

    Returns ``(bearings, ranges)``: ``num_bins`` bearings (rad, + = left,
    ascending) across the camera's horizontal FOV and the nearest obstacle
    range in each bin (``max_range`` if free).
    """
    h, w = depth.shape
    step = cfg.pixel_step
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float64)
    x = (u - K.cx) / K.fx * z
    y = (v - K.cy) / K.fy * z
    height = cfg.camera_height - y                     # y points down
    mask = (z > 0.05) & (z < cfg.max_range) & (height > cfg.min_obstacle_height) & (height < cfg.max_obstacle_height)

    half_fov = np.arctan2(max(K.cx, w - K.cx), K.fx)
    bearings = np.linspace(-half_fov, half_fov, cfg.num_bins)

    def to_bin(xs, zs):
        b = np.arctan2(-xs, zs)                         # + = left
        return np.clip(np.round((b + half_fov) / (2 * half_fov) * (cfg.num_bins - 1)).astype(int),
                       0, cfg.num_bins - 1)

    ranges = np.full(cfg.num_bins, cfg.max_range)
    if mask.any():
        np.minimum.at(ranges, to_bin(x[mask], z[mask]), np.hypot(x[mask], z[mask]))

    # Bins that no valid pixel falls into carry no information (sparse image
    # or invalid depth) — interpolate them rather than report them as free.
    valid = z > 0.05
    covered = np.zeros(cfg.num_bins, dtype=bool)
    covered[to_bin(x[valid], z[valid])] = True
    if covered.any() and not covered.all():
        ranges[~covered] = np.interp(bearings[~covered], bearings[covered], ranges[covered])
    return bearings, ranges


class DepthGapFollower:
    """Follow-the-Gap controller on the depth virtual scan."""

    def __init__(self, params: VehicleParams, K: Intrinsics, cfg: DepthFTGConfig | None = None) -> None:
        self.params = params
        self.K = K
        self.cfg = cfg or DepthFTGConfig()
        self.last_scan: Tuple[np.ndarray, np.ndarray] | None = None
        self.last_target: Tuple[float, float] | None = None   # (bearing, range)

    def reset(self) -> None:
        self.last_scan = self.last_target = None

    def step(self, depth: np.ndarray) -> Command:
        c, p = self.cfg, self.params
        bearings, ranges = depth_to_scan(depth, self.K, c)
        self.last_scan = (bearings, ranges)

        # Obstacle directly ahead within stop_distance → neutral (coast).
        ahead = np.abs(bearings) < np.arctan2(p.width, c.stop_distance)
        if ahead.any() and ranges[ahead].min() < c.stop_distance:
            self.last_target = None
            return Command(0.0, 0.0)

        r = ranges.copy()
        if c.smooth_window > 1:
            r = np.convolve(np.pad(r, c.smooth_window // 2, mode="edge"),
                            np.ones(c.smooth_window) / c.smooth_window, mode="valid")
        nearest = int(np.argmin(r))
        bin_width = bearings[1] - bearings[0]
        bubble = int(np.ceil(np.arctan2(c.bubble_radius, max(r[nearest], 1e-3)) / bin_width))
        r[max(0, nearest - bubble): nearest + bubble + 1] = 0.0
        free = r >= c.min_gap_range

        # Best gap = largest visible free area (sum of ranges), not just the
        # widest run — a nearby obstacle face must not beat genuinely open space.
        best_start, best_len, best_score, start = 0, 0, 0.0, None
        for i, f in enumerate(np.append(free, False)):
            if f and start is None:
                start = i
            elif not f and start is not None:
                score = float(r[start:i].sum())
                if score > best_score:
                    best_start, best_len, best_score = start, i - start, score
                start = None
        if best_len == 0:
            self.last_target = None
            return Command(0.0, 0.0)
        gap = r[best_start: best_start + best_len]
        if c.strategy == "deepest":
            # Deepest point; if several bins tie (range limit), aim at the middle of that run
            deep = np.flatnonzero(gap >= 0.95 * gap.max())
            phi = float(bearings[best_start + int(deep[len(deep) // 2])])
        else:
            phi_gap = 0.5 * float(bearings[best_start] + bearings[best_start + best_len - 1])
            w_gap = c.alpha / max(float(ranges[nearest]), 1e-3)
            phi = (w_gap * phi_gap + c.beta * 0.0) / (w_gap + c.beta)
        self.last_target = (phi, float(gap.mean()))

        # Pure-pursuit curvature towards phi over a short aim distance
        delta = np.arctan2(2.0 * p.wheelbase * np.sin(phi), c.aim_distance)
        steering = float(np.clip(delta / p.max_steering_angle, -1.0, 1.0))
        speed = c.cruise_speed * (1.0 - 0.5 * abs(steering))
        return Command(steering, speed / c.speed_per_throttle).clipped()
