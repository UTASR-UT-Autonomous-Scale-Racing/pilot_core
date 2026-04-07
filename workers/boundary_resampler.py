"""
boundary_resampler — resample ordered boundary polylines to uniform arc-length spacing.

Usage
-----
    from workers.boundary_resampler import resample_uniform

    outer_resampled = resample_uniform(outer_line, spacing=0.1)
    inner_resampled = resample_uniform(inner_line, spacing=0.1)

Then to build an RL observation from the current vehicle position:

    def boundary_obs(pos, heading_rad, outer, inner, n_ahead=10):
        # Find closest index on each boundary
        idx_o = np.linalg.norm(outer - pos, axis=1).argmin()
        idx_i = np.linalg.norm(inner - pos, axis=1).argmin()

        # Walk forward n_ahead+1 points (wraps around closed track)
        n = len(outer)
        o_pts = outer[np.arange(idx_o, idx_o + n_ahead + 1) % n]  # (11, 2)

        m = len(inner)
        i_pts = inner[np.arange(idx_i, idx_i + n_ahead + 1) % m]  # (11, 2)

        # Transform into vehicle-local frame
        cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
        R_inv = np.array([[cos_h, sin_h], [-sin_h, cos_h]])  # rotate world → vehicle

        o_local = (o_pts - pos) @ R_inv.T   # (11, 2)
        i_local = (i_pts - pos) @ R_inv.T   # (11, 2)

        # Flatten: [o0x,o0y, o1x,o1y, ..., i0x,i0y, ..., i10x,i10y] = 44 floats
        return np.concatenate([o_local.ravel(), i_local.ravel()]).astype(np.float32)
"""

from __future__ import annotations

import numpy as np


def smooth_boundary(points: np.ndarray, window: int = 15) -> np.ndarray:
    """Moving-average smooth to remove zigzag from clustered LIDAR wall points.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2)`` ordered boundary points.
    window : int
        Moving-average kernel size (number of points).
    """
    if len(points) < window or window <= 1:
        return points.astype(np.float32)
    kernel = np.ones(window) / window
    px = np.convolve(points[:, 0], kernel, mode='same')
    py = np.convolve(points[:, 1], kernel, mode='same')
    return np.stack([px, py], axis=1).astype(np.float32)


def resample_uniform(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an ordered polyline to uniformly spaced points by arc length.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 2)`` ordered polyline vertices.
    spacing : float
        Desired arc-length distance between consecutive output points (metres).

    Returns
    -------
    ``(M, 2) float32`` array with consecutive points spaced ``spacing`` apart.
    The last point may be slightly less than ``spacing`` from the second-to-last
    because sampling stops before the end of the polyline.
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
    """Build a vehicle-local RL observation from resampled boundary lines.

    Returns a ``((n_ahead+1)*4,)`` float32 vector:
    ``[outer_0..outer_n, inner_0..inner_n]`` — each point as (x, y) in the
    vehicle's local frame (forward = +x, left = +y).

    Parameters
    ----------
    pos : np.ndarray
        ``(2,)`` current vehicle position in world frame.
    heading_rad : float
        Vehicle heading in the world frame (radians).
    outer : np.ndarray
        ``(N, 2)`` uniformly resampled outer boundary.
    inner : np.ndarray
        ``(M, 2)`` uniformly resampled inner boundary.
    n_ahead : int
        Number of boundary points ahead of the closest point to include.
        Total points per boundary = n_ahead + 1 (closest + n_ahead forward).
    """
    idx_o = int(np.linalg.norm(outer - pos, axis=1).argmin())
    idx_i = int(np.linalg.norm(inner - pos, axis=1).argmin())

    n, m = len(outer), len(inner)
    o_pts = outer[np.arange(idx_o, idx_o + n_ahead + 1) % n]   # (n_ahead+1, 2)
    i_pts = inner[np.arange(idx_i, idx_i + n_ahead + 1) % m]   # (n_ahead+1, 2)

    # Rotate world-frame points into vehicle-local frame
    cos_h, sin_h = np.cos(heading_rad), np.sin(heading_rad)
    R_inv = np.array([[cos_h, sin_h], [-sin_h, cos_h]], dtype=np.float32)

    o_local = (o_pts - pos) @ R_inv.T   # forward = +x, left = +y
    i_local = (i_pts - pos) @ R_inv.T

    return np.concatenate([o_local.ravel(), i_local.ravel()]).astype(np.float32)
