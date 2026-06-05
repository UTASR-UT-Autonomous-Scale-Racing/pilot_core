"""
Approximate 2D LiDAR scan from the drivable-track mask and the aligned depth image, for
the controllers that only take a scan (the follow-the-gap mapping drive) on a car that has
no LiDAR.

Per image column: start at the lowest drivable pixel (the car's own body can fill the
bottom of the view), go up while the pixels stay drivable, and the first non-drivable
pixel is the track edge. The edge is placed with the depth at the edge pixel (a wall's
base, or the boundary line), else of the floor just below it, back-projected through the
intrinsics and levelled by the camera pitch. Points are binned into the LiDAR's beams (nearest per beam).

Scan convention as control/lidar_gap.py: 1080 ranges, -135 deg (right) to +135 deg (left),
0 = straight ahead. inf = free as far as the camera sees, NaN = unknown (outside the
camera's view, or no depth): LidarGapFollower treats NaN as blocked when choosing a gap
and ignores it when pushing away from walls.

Camera frame x right, y down, z forward; depth = optical z (m), 0 / non-finite = none.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth_ftg import Intrinsics
from .lidar_gap import ANGLE_INCREMENT, ANGLE_MIN

N_BEAMS = 1080


@dataclass
class SegScanConfig:
    camera_pitch: float = 0.0      # rad, positive = pitched down
    camera_height: float = 0.12    # m above the floor (only for floor_fallback)
    origin_forward: float = 0.0    # m, camera position ahead of the scan origin (negative = behind)
    max_range: float = 10.0        # m, edges farther than this are free (inf)
    edge_rows: int = 3             # rows at/above the edge (else just below it) whose depth places it
    # no valid depth at an edge: intersect that pixel's ray with the flat floor instead
    # (camera_height, camera_pitch). Off = such columns are unknown (NaN)
    floor_fallback: bool = True
    fill_beams: int = 6            # unknown runs up to this many beams between known ones are interpolated


def _floor_points(u, v, K: Intrinsics, cfg: SegScanConfig):
    """Pixel rays -> (fwd, left) where they meet the floor; inf where they never do."""
    xn, yn = (u - K.cx) / K.fx, (v - K.cy) / K.fy
    cp, sp = np.cos(cfg.camera_pitch), np.sin(cfg.camera_pitch)
    d_fwd, d_up = cp - yn * sp, -sp - yn * cp
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(d_up < -1e-6, cfg.camera_height / -d_up, np.inf)
    return t * d_fwd, t * -xn


def seg_edge_scan(mask: np.ndarray, depth: np.ndarray, K: Intrinsics, cfg: SegScanConfig | None = None,
                  return_debug: bool = False):
    """mask (H, W), nonzero = drivable; depth (H, W) m, aligned with it -> (1080,) float32 ranges.

    return_debug=True also returns a dict: per-column edge row (-1 = none), and how each
    column's range was obtained ("depth", "floor", "free", "blocked", "unknown") as counts.
    """
    cfg = cfg or SegScanConfig()
    depth = np.asarray(depth, dtype=np.float64)
    h, w = depth.shape
    m = np.asarray(mask)
    if m.shape != (h, w):
        import cv2

        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    drv = m > 0
    valid = np.isfinite(depth) & (depth > 0.05)

    # rows counted from the bottom: b = 0 is the lowest image row
    f = drv[::-1]
    rows = np.arange(h)[:, None]
    has_drv = f.any(axis=0)
    first = np.argmax(f, axis=0)                          # lowest drivable pixel
    above = ~f & (rows > first[None, :])
    has_edge = has_drv & above.any(axis=0)
    edge_b = np.where(has_edge, np.argmax(above, axis=0), -1)

    cols = np.arange(w)
    edge_v = np.where(has_edge, h - 1 - edge_b, -1)       # image row of the edge pixel
    fwd = np.full(w, np.nan)
    left = np.full(w, np.nan)
    how = np.full(w, "unknown", dtype=object)

    cp, sp = np.cos(cfg.camera_pitch), np.sin(cfg.camera_pitch)

    def place(u, v, z):
        x, y = (u - K.cx) / K.fx * z, (v - K.cy) / K.fy * z
        return z * cp - y * sp, -x

    for u in cols[has_edge]:
        e = edge_v[u]
        # the edge pixel and the few above it (a wall's base, or the boundary line itself);
        # failing that, the floor just below the edge, which reads a little short
        at = np.arange(e, max(-1, e - cfg.edge_rows), -1)
        at = at[valid[at, u]]
        below = np.arange(e + 1, min(h, e + 1 + cfg.edge_rows))
        below = below[drv[below, u] & valid[below, u]]
        vs = at if len(at) else below
        if len(vs):
            f_, l_ = place(u, vs.astype(np.float64), depth[vs, u])
            k = int(np.argmin(np.hypot(f_, l_)))          # nearest: the base of the edge
            fwd[u], left[u] = f_[k], l_[k]
            how[u] = "depth"
        elif cfg.floor_fallback:
            ff, ll = _floor_points(np.array([u]), np.array([e + 1.0]), K, cfg)
            if np.isfinite(ff[0]):
                fwd[u], left[u] = ff[0], ll[0]
                how[u] = "floor"
            else:
                how[u] = "free"          # the edge is at or above the horizon
    # drivable all the way to the top of the image: free as far as the camera sees
    how[has_drv & ~has_edge] = "free"
    # nothing drivable in the column: the nearest thing with depth blocks it
    for u in cols[~has_drv]:
        vs = np.flatnonzero(valid[:, u])
        if len(vs):
            v = vs[np.argmin(depth[vs, u])]
            fwd[u], left[u] = place(u, v, depth[v, u])
            how[u] = "blocked"

    # bin into beams, relative to the scan origin
    fwd_o = fwd + cfg.origin_forward
    r = np.hypot(fwd_o, left)
    bearing = np.arctan2(left, fwd_o)
    # the bearing of free / unknown columns: the ray through the column centre at the horizon
    xn = (cols - K.cx) / K.fx
    col_bearing = np.where(np.isfinite(bearing), bearing, np.arctan2(-xn, cp))
    beam = np.round((col_bearing - ANGLE_MIN) / ANGLE_INCREMENT).astype(int)
    ok = (beam >= 0) & (beam < N_BEAMS)

    ranges = np.full(N_BEAMS, np.nan)
    hit = ok & np.isfinite(r) & (r <= cfg.max_range)
    far = ok & ((np.isfinite(r) & (r > cfg.max_range)) | (how == "free"))
    ranges[np.unique(beam[far])] = np.inf
    best = np.full(N_BEAMS, np.inf)
    np.minimum.at(best, beam[hit], r[hit])
    got = np.isfinite(best)
    ranges[got] = best[got]

    # short unknown runs between known beams (columns are coarser than beams at the edges)
    known = ~np.isnan(ranges)
    if known.sum() >= 2 and cfg.fill_beams > 0:
        idx = np.flatnonzero(known)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= cfg.fill_beams + 1:
                ra, rb = ranges[a], ranges[b]
                if np.isfinite(ra) and np.isfinite(rb):
                    ranges[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [ra, rb])
                elif np.isinf(ra) and np.isinf(rb):
                    ranges[a + 1:b] = np.inf
                else:
                    ranges[a + 1:b] = min(ra, rb)     # free next to a wall: keep the wall
    out = ranges.astype(np.float32)
    if not return_debug:
        return out
    kinds = ("depth", "floor", "free", "blocked", "unknown")
    return out, {"edge_row": edge_v, "counts": {k: int(np.sum(how == k)) for k in kinds}}
