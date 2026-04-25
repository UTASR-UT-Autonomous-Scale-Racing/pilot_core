"""
Resample ordered boundary polylines to uniform arc-length spacing, and build
the boundary observation vector from them. Typical use:

    outer = resample_uniform(outer_line, spacing=0.1)
    inner = resample_uniform(inner_line, spacing=0.1)
    obs = boundary_obs(pos, heading_rad, outer, inner, n_ahead=10)  # (44,) float32
"""

from __future__ import annotations

import numpy as np


def smooth_boundary(points: np.ndarray, window: int = 15) -> np.ndarray:
    """Moving average over (N, 2) ordered points, `window` points wide.

    Takes out the zigzag you get from clustered LIDAR wall points.
    """
    if len(points) < window or window <= 1:
        return points.astype(np.float32)
    kernel = np.ones(window) / window
    px = np.convolve(points[:, 0], kernel, mode='same')
    py = np.convolve(points[:, 1], kernel, mode='same')
    return np.stack([px, py], axis=1).astype(np.float32)


def resample_uniform(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an (N, 2) polyline so points are `spacing` metres apart
    along the arc. Returns (M, 2) float32.

    Sampling stops before the end of the polyline, so the final gap can
    come out a bit shorter than `spacing`.
    """
    if len(points) < 2:
        return points.astype(np.float32)

    diffs = np.diff(points, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])

    total_len = arc[-1]
    if total_len == 0.0:
        return points[:1].astype(np.float32)

    targets = np.arange(0.0, total_len, spacing)
    new_x = np.interp(targets, arc, points[:, 0])
    new_y = np.interp(targets, arc, points[:, 1])
    return np.stack([new_x, new_y], axis=1).astype(np.float32)


def boundary_obs(
    pos: np.ndarray,
    heading_rad: float,
    outer: np.ndarray,
    inner: np.ndarray,
    n_ahead: int = 10,
) -> np.ndarray:
    """Vehicle-local RL observation from resampled boundary lines.

    Output is a ((n_ahead+1)*4,) float32 vector laid out as
    [outer_0..outer_n, inner_0..inner_n], each point (x, y) in the vehicle
    frame (forward = +x, left = +y).

    Parameters
    ----------
    pos : (2,) vehicle position, world frame
    heading_rad : world-frame heading, radians
    outer, inner : uniformly resampled boundaries, (N, 2) and (M, 2)
    n_ahead : how many points past the closest one to take, so each
        boundary contributes n_ahead + 1 points. Indexing wraps, i.e. the
        lines are assumed closed.
    """
    idx_o = int(np.linalg.norm(outer - pos, axis=1).argmin())
    idx_i = int(np.linalg.norm(inner - pos, axis=1).argmin())

    n, m = len(outer), len(inner)
    o_pts = outer[np.arange(idx_o, idx_o + n_ahead + 1) % n]   # (n_ahead+1, 2)
    i_pts = inner[np.arange(idx_i, idx_i + n_ahead + 1) % m]   # (n_ahead+1, 2)

    # world -> vehicle frame
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    R_inv = np.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float32)

    o_local = (o_pts - pos) @ R_inv.T   # forward = +x, left = +y
    i_local = (i_pts - pos) @ R_inv.T

    return np.concatenate([o_local.ravel(), i_local.ravel()]).astype(np.float32)
