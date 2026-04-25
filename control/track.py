"""
TrackModel: centre-line built once from the mapped boundaries, then queried
every control step.

Gives arc length s, curvature kappa(s), half-width, and a windowed project()
that carries the index between steps, so a query costs O(window) instead of
a global search.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .types import wrap_angle


@dataclass
class Projection:
    idx: int               # index of the nearest centre-line sample
    s: float               # arc length of the projected point (m)
    lateral_error: float   # m, signed, + means the car is left of the path
    path_heading: float    # tangent heading at the projection (rad)
    curvature: float       # path curvature at the projection (1/m, + = left turn)


def _resample_closed(pts: np.ndarray, spacing: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = seg > 1e-9
    closed = np.vstack([closed[:-1][keep], closed[-1:]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    n = max(int(round(s[-1] / spacing)), 8)
    s_new = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(s_new, s, closed[:, 0]), np.interp(s_new, s, closed[:, 1])])


def _smooth_closed(pts: np.ndarray, window: int) -> np.ndarray:
    """Moving average, wraps around."""
    if window <= 1:
        return pts
    k = np.ones(window) / window
    pad = window // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])
    out = np.column_stack([np.convolve(ext[:, i], k, mode="valid") for i in range(2)])
    return out[: len(pts)]


class TrackModel:
    """Closed-loop centre-line model.

    Parameters
    ----------
    outer, inner : np.ndarray
        (N, 2) / (M, 2) boundary points in the BEV plane, each ordered along
        the track (MappingWorker already gives them in that order).
    spacing : float
        Centre-line sample spacing, metres.
    smooth_window : int
        Samples in the circular smoothing window, applied before curvature.
    """

    def __init__(
        self,
        outer: np.ndarray,
        inner: np.ndarray,
        spacing: float = 0.05,
        smooth_window: int = 9,
    ) -> None:
        outer = np.asarray(outer, dtype=np.float64)[:, :2]
        inner = np.asarray(inner, dtype=np.float64)[:, :2]
        if len(outer) < 3 or len(inner) < 3:
            raise ValueError("TrackModel needs at least 3 points per boundary")

        # centre = midpoint of each outer sample and its nearest inner point
        outer_rs = _resample_closed(outer, spacing)
        inner_tree = cKDTree(_resample_closed(inner, spacing / 2))
        _, j = inner_tree.query(outer_rs)
        mid = 0.5 * (outer_rs + inner_tree.data[j])
        centre = _resample_closed(_smooth_closed(mid, smooth_window), spacing)

        self._set_centre(centre)
        # half-width is just distance to the nearest wall
        walls = np.vstack([_resample_closed(outer, spacing / 2), _resample_closed(inner, spacing / 2)])
        self.half_width, _ = cKDTree(walls).query(self.points)

    def _set_centre(self, centre: np.ndarray) -> None:
        self.points = centre
        n = len(centre)
        nxt = np.roll(centre, -1, axis=0)
        seg = np.linalg.norm(nxt - centre, axis=1)
        self.length = float(seg.sum())
        self.s = np.concatenate([[0.0], np.cumsum(seg[:-1])])
        # Central differences over roughly a +/-0.25 m stencil. Doing second
        # differences at the raw sample spacing blows up the resampling kinks.
        k = max(1, int(round(0.25 / max(self.length / n, 1e-6))))
        fwd, back = np.roll(centre, -k, axis=0), np.roll(centre, k, axis=0)
        d1 = (fwd - back) / (2.0 * k)
        d2 = (fwd - 2.0 * centre + back) / (k * k)
        self.heading = np.arctan2(d1[:, 1], d1[:, 0])
        denom = np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)
        self.curvature = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / denom
        self._tree = cKDTree(centre)
        self._n = n

    def reverse(self) -> None:
        hw = self.half_width[::-1].copy()
        self._set_centre(self.points[::-1].copy())
        self.half_width = hw

    def align_direction(self, x: float, y: float, yaw: float) -> bool:
        """Point the centre-line the way the car is facing. True if it had to reverse."""
        p = self.project(x, y)
        if abs(wrap_angle(p.path_heading - yaw)) > np.pi / 2:
            self.reverse()
            return True
        return False

    def project(self, x: float, y: float, hint: int | None = None, window: int = 40) -> Projection:
        """Project (x, y) onto the centre-line.

        Pass the previous idx as hint and only a window around it gets
        searched. If the best match lands on the window edge (pose jump,
        usually) we fall back to the global KD-tree query.
        """
        pos = np.array([x, y])
        if hint is None:
            _, idx = self._tree.query(pos)
        else:
            rng = (np.arange(-window, window + 1) + hint) % self._n
            d = np.linalg.norm(self.points[rng] - pos, axis=1)
            k = int(np.argmin(d))
            idx = int(rng[k])
            if k in (0, len(rng) - 1):
                _, idx = self._tree.query(pos)
        idx = int(idx)

        # refine using the tangent at the nearest sample
        p0 = self.points[idx]
        h = self.heading[idx]
        t = np.array([np.cos(h), np.sin(h)])
        along = float(np.dot(pos - p0, t))
        lateral = float(t[0] * (pos - p0)[1] - t[1] * (pos - p0)[0])
        s = (self.s[idx] + along) % self.length
        return Projection(idx, s, lateral, float(h), float(self.curvature[idx]))

    def index_at(self, s: float) -> int:
        """Sample index at or just before arc length s. Wraps."""
        return int(np.searchsorted(self.s, s % self.length, side="right") - 1) % self._n

    def point_at(self, s: float) -> np.ndarray:
        """Interpolated centre-line point at s (wraps)."""
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side="right") - 1) % self._n
        j = (i + 1) % self._n
        seg = self.length - self.s[i] if j == 0 else self.s[j] - self.s[i]
        a = 0.0 if seg <= 0 else (s - self.s[i]) / seg
        return (1 - a) * self.points[i] + a * self.points[j]
