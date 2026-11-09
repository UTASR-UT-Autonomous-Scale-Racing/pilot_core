"""
Minimum-curvature racing line inside the mapped track.

The line is the reference path shifted along its normals, P_i + a_i n_i. Squared second
differences of the shifted points (a curvature proxy at uniform spacing) are linear in a,
so each pass is a bounded linear least-squares problem: a_i stays between the walls minus
a margin, measured by casting the normal against the boundaries. A few passes, each
re-linearised about the previous line, let it move further than one pass's normals allow.

Ref: F. Braghin et al., "Race driving along the minimum curvature path", 2008;
A. Heilmeier et al., "Minimum curvature trajectory planning and control for an
autonomous race car", 2019 (the same QP, with splines).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree

from .track import TrackModel, _resample_closed, _smooth_closed


def _normals(pts: np.ndarray) -> np.ndarray:
    """Unit left normals of a closed polyline (central differences)."""
    d = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return np.column_stack([-d[:, 1], d[:, 0]])


def _ray_distances(origins: np.ndarray, dirs: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray,
                   max_dist: float) -> np.ndarray:
    """Distance from each origin along its dir to the nearest wall segment (max_dist if none)."""
    e = seg_b - seg_a                                       # (M, 2)
    out = np.full(len(origins), max_dist)
    for k in range(0, len(origins), 256):                   # chunks bound the (n, M) arrays
        o, d = origins[k:k + 256, None, :], dirs[k:k + 256, None, :]
        w = seg_a[None] - o                                 # (n, M, 2)
        den = d[..., 0] * e[None, :, 1] - d[..., 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (w[..., 0] * e[None, :, 1] - w[..., 1] * e[None, :, 0]) / den
            u = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / den
        hit = (np.abs(den) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, max_dist)
        out[k:k + 256] = np.minimum(t.min(axis=1), max_dist)
    return out


def min_curvature_line(
    track: TrackModel,
    outer: np.ndarray,
    inner: np.ndarray,
    margin: float = 0.3,
    spacing: float = 0.2,
    passes: int = 4,
    smooth_window: int = 3,
    out_spacing: float = 0.05,
) -> np.ndarray:
    """(N, 2) closed racing line in the direction of track.points.

    margin: m kept from either wall (car half-width plus a safety band);
    spacing: optimisation sample spacing (m); passes: re-linearisations; out_spacing:
    spacing of the returned points, sampled from a periodic cubic spline through the
    optimised ones (a linear resample would leave curvature spikes at every knot)."""
    walls = [_resample_closed(np.asarray(b, dtype=np.float64)[:, :2], 0.05) for b in (outer, inner)]
    seg_a = np.vstack([w for w in walls])
    seg_b = np.vstack([np.roll(w, -1, axis=0) for w in walls])
    wall_pts = seg_a
    wall_tree = cKDTree(wall_pts)
    pts = _resample_closed(track.points, spacing)
    for _ in range(passes):
        n = len(pts)
        # second-difference operator on a closed loop
        D = np.zeros((n, n))
        idx = np.arange(n)
        D[idx, (idx - 1) % n], D[idx, idx], D[idx, (idx + 1) % n] = 1.0, -2.0, 1.0
        nrm = _normals(pts)
        left = _ray_distances(pts, nrm, seg_a, seg_b, 5.0) - margin
        right = _ray_distances(pts, -nrm, seg_a, seg_b, 5.0) - margin
        # a squeezed sample (narrower than 2 * margin) stays where it is
        lo, hi = np.minimum(-right, 0.0), np.maximum(left, 0.0)
        lo, hi = np.where(lo < hi, lo, -1e-6), np.where(lo < hi, hi, 1e-6)
        # D (pts + a * nrm) = 0 in least squares, x and y rows stacked
        A = np.vstack([D * nrm[:, 0], D * nrm[:, 1]])
        b = -np.concatenate([D @ pts[:, 0], D @ pts[:, 1]])
        # the normal ray misses walls off to the side (a sharp inner corner): where the
        # nearest wall is closer than the margin, pull that side's bound in and solve again
        for _ in range(8):
            a = lsq_linear(A, b, bounds=(lo, hi), method="bvls").x
            new = pts + a[:, None] * nrm
            d, j = wall_tree.query(new)
            short = d < margin - 0.01
            if not short.any():
                break
            side = np.sign(((wall_pts[j] - new) * nrm).sum(axis=1))
            push = (margin - d + 0.02)[short]
            left_wall, right_wall = short & (side > 0), short & (side <= 0)
            hi[left_wall] = np.maximum(a[left_wall] - push[side[short] > 0], lo[left_wall] + 1e-6)
            lo[right_wall] = np.minimum(a[right_wall] + push[side[short] <= 0], hi[right_wall] - 1e-6)
        pts = _resample_closed(new, spacing)
    pts = _smooth_closed(pts, smooth_window)
    closed = np.vstack([pts, pts[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    spline = CubicSpline(s, closed, bc_type="periodic")
    return spline(np.arange(0.0, s[-1], out_spacing))
