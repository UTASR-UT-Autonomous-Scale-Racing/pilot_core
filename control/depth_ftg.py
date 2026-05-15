"""
Map-free reactive control: virtual laser scan from the depth image, then Follow-the-Gap.

Ref: V. Sezer, M. Gokasan, "A novel obstacle avoidance algorithm: 'Follow the Gap Method'", RAS 60(9), 2012.
Camera frame x right, y down, z forward; bearings + = left.
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
        """Pull Camera1.fx/fy/cx/cy out of an ORB-SLAM3 settings yaml."""
        from .types import resolve_path

        vals = {}
        with open(resolve_path(path)) as f:
            for line in f:
                for key in ("fx", "fy", "cx", "cy"):
                    if line.strip().startswith(f"Camera1.{key}:") or line.strip().startswith(f"Camera.{key}:"):
                        vals.setdefault(key, float(line.split(":", 1)[1]))
        return cls(**vals)


@dataclass
class DepthFTGConfig:
    camera_height: float = 0.12       # m above ground (measure on the car)
    camera_pitch: float = 0.0         # rad, positive = pitched down
    min_obstacle_height: float = 0.04  # m, below this is floor
    max_obstacle_height: float = 0.50  # m, above this is ignored
    max_range: float = 4.0            # m
    num_bins: int = 64                # bearing bins across the FOV
    pixel_step: int = 4               # subsample the depth image
    smooth_window: int = 3
    bubble_radius: float = 0.25       # m, car half-width plus margin
    min_gap_range: float = 0.6        # m, bins closer than this aren't "free"
    alpha: float = 1.0                # gap-angle weight (scaled by 1/d_min)
    beta: float = 0.0                 # goal-angle weight (goal = straight ahead)
    aim_distance: float = 0.7         # m (pure-pursuit look-ahead toward phi)
    cruise_speed: float = 0.6         # m/s
    stop_distance: float = 0.35       # m, anything this close ahead -> neutral
    min_valid_fraction: float = 0.2   # below this share of valid depth pixels -> neutral
    speed_per_throttle: float = 3.0   # m/s per unit throttle


def depth_to_scan(
    depth: np.ndarray, K: Intrinsics, cfg: DepthFTGConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Virtual scan from a depth image -> (bearings, ranges).

    bearings: num_bins ascending angles (rad, + = left) across the FOV;
    ranges: nearest obstacle per bin (m), max_range if none.
    """
    h, w = depth.shape
    step = cfg.pixel_step
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step].astype(np.float64)
    z = np.where(np.isfinite(z), z, 0.0)
    x = (u - K.cx) / K.fx * z
    y = (v - K.cy) / K.fy * z
    if cfg.camera_pitch:
        # rotate into a level frame (y down, z forward along the ground)
        cp, sp = np.cos(cfg.camera_pitch), np.sin(cfg.camera_pitch)
        y, z = y * cp + z * sp, z * cp - y * sp
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

    # bins with no valid pixel tell us nothing: interpolate rather than call them free
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
        sample = depth[::c.pixel_step, ::c.pixel_step]
        if np.mean(np.isfinite(sample) & (sample > 0.05)) < c.min_valid_fraction:
            self.last_scan = self.last_target = None
            return Command(0.0, 0.0)
        bearings, ranges = depth_to_scan(depth, self.K, c)
        self.last_scan = (bearings, ranges)

        # something right in front, inside stop_distance: coast
        ahead = np.abs(bearings) < np.arctan2(p.width, c.stop_distance)
        if ahead.any() and ranges[ahead].min() < c.stop_distance:
            self.last_target = None
            return Command(0.0, 0.0)

        r = ranges.copy()
        if c.smooth_window > 1:
            r = np.convolve(np.pad(r, c.smooth_window // 2, mode="edge"),
                            np.ones(c.smooth_window) / c.smooth_window, mode="valid")
        nearest = int(np.argmin(r))
        if r[nearest] < c.max_range:                    # no bubble in open space
            bin_width = bearings[1] - bearings[0]
            bubble = int(np.ceil(np.arctan2(c.bubble_radius, max(r[nearest], 1e-3)) / bin_width))
            r[max(0, nearest - bubble): nearest + bubble + 1] = 0.0
        free = r >= c.min_gap_range

        # best gap = largest free area (sum of ranges), not widest run, so a near wall can't win
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
        phi_gap = 0.5 * float(bearings[best_start] + bearings[best_start + best_len - 1])
        w_gap = c.alpha / max(float(ranges[nearest]), 1e-3)
        phi = (w_gap * phi_gap + c.beta * 0.0) / (w_gap + c.beta)
        self.last_target = (phi, float(gap.mean()))

        # pure-pursuit curvature toward phi
        delta = np.arctan2(2.0 * p.wheelbase * np.sin(phi), c.aim_distance)
        steering = float(np.clip(delta / p.max_steering_angle, -1.0, 1.0))
        speed = c.cruise_speed * (1.0 - 0.5 * abs(steering))
        return Command(steering, speed / c.speed_per_throttle).clipped()
