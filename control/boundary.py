"""
Closed track boundaries: smoothing, uniform resampling and the RL boundary observation.
Single implementation shared by training and deployment so observations match.
Boundaries are closed (N, 2) loops; a trailing copy of the first point is dropped.
"""

from __future__ import annotations

import numpy as np


def _open_loop(points: np.ndarray) -> np.ndarray:
    """(N, 2) float64 copy without a duplicated closing point."""
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    if len(pts) > 2 and np.linalg.norm(pts[-1] - pts[0]) < 1e-9:
        pts = pts[:-1]
    return pts


def smooth_boundary(points: np.ndarray, window: int = 15) -> np.ndarray:
    """Circular moving average over a closed (N, 2) boundary, `window` points wide.

    Wrap-padded so points near the seam average with their real neighbours;
    even windows are rounded up to odd so the average stays centred.
    """
    pts = _open_loop(points)
    if window <= 1 or len(pts) < window:
        return pts.astype(np.float32)
    if window % 2 == 0:
        window += 1
    half = window // 2
    padded = np.concatenate([pts[-half:], pts, pts[:half]], axis=0)
    kernel = np.ones(window) / window
    out = np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(2)], axis=1)
    return out.astype(np.float32)


def resample_uniform(points: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed (N, 2) boundary to uniform arc-length spacing -> (M, 2) float32.

    Spacing becomes total / round(total / spacing) so the seam gap matches the others.
    """
    pts = _open_loop(points)
    if len(pts) < 2:
        return pts.astype(np.float32)
    loop = np.concatenate([pts, pts[:1]], axis=0)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    total = float(arc[-1])
    if total == 0.0:
        return pts[:1].astype(np.float32)
    n = max(int(round(total / float(spacing))), 3)
    targets = np.arange(n) * (total / n)
    out = np.stack([np.interp(targets, arc, loop[:, 0]), np.interp(targets, arc, loop[:, 1])], axis=1)
    return out.astype(np.float32)


def prepare_boundary(points: np.ndarray, smooth_window: int, spacing: float) -> np.ndarray:
    """Raw mapped boundary -> smoothed, uniformly resampled closed boundary."""
    return resample_uniform(smooth_boundary(points, smooth_window), spacing)


def boundary_window(line: np.ndarray, pos: np.ndarray, n_ahead: int) -> np.ndarray:
    """World-frame points of `line` from the one closest to `pos` plus
    n_ahead more, wrapping around the closed loop. (n_ahead + 1, 2)."""
    idx = int(np.linalg.norm(line - pos, axis=1).argmin())
    return line[np.arange(idx, idx + n_ahead + 1) % len(line)]


def boundary_obs(
    pos: np.ndarray,
    heading_rad: float,
    outer: np.ndarray,
    inner: np.ndarray,
    n_ahead: int = 10,
) -> np.ndarray:
    """Vehicle-local boundary observation, ((n_ahead+1)*4,) float32.

    Layout [outer_0..outer_n, inner_0..inner_n], each point (x, y) in the
    vehicle frame (forward = +x, left = +y). `outer`/`inner` are prepared
    (resampled) boundaries running in the driving direction.
    """
    pos = np.asarray(pos, dtype=np.float32)
    o_pts = boundary_window(outer, pos, n_ahead)
    i_pts = boundary_window(inner, pos, n_ahead)
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    r_inv = np.array([[c, s], [-s, c]], dtype=np.float32)   # world -> vehicle
    return np.concatenate([((o_pts - pos) @ r_inv.T).ravel(),
                           ((i_pts - pos) @ r_inv.T).ravel()]).astype(np.float32)


def obs_dim(n_ahead: int) -> int:
    """Length of the full policy observation (boundaries + speed)."""
    return (n_ahead + 1) * 4 + 1


def track_observation(
    pos: np.ndarray,
    heading_rad: float,
    speed: float,
    outer: np.ndarray,
    inner: np.ndarray,
    n_ahead: int,
) -> np.ndarray:
    """Full policy observation: boundary_obs(...) followed by speed (m/s)."""
    return np.append(boundary_obs(pos, heading_rad, outer, inner, n_ahead), speed).astype(np.float32)
