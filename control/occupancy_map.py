"""
Track walls from segmentation-edge scans, as an occupancy grid.

Each scan is a ray fan from the scan origin to the drivable area's edge (control/seg_scan.py).
A ray frees the floor it crosses; an edge only counts as wall when it was seen from close by,
so an edge placed short from far away (dark corners, the far end of a straight) is cleared
again by the floor seen up close. The drivable region is the floor seen free, connected to the
driven path and within `reach` of it; its outer contour and its hole are the two walls.

Scans taken while the car stands still, or around a disturbance (a wall hit, a respawn), are
left out: they repeat one view of a wall many times.

Used live by MappingWorker (`wall_source: occupancy`) and offline by scripts/seg_map.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import _resample_closed, _smooth_closed


@dataclass
class OccupancyConfig:
    res: float = 0.05            # m, grid cell
    beam_step: int = 6           # every n-th beam is used
    pad: float = 3.0             # m of grid around the driven path
    min_move: float = 0.03       # m driven between scans used
    drop_before: float = 1.5     # s before a disturbance whose scans are dropped
    drop_after: float = 2.5      # s after it
    edge_range: float = 2.5      # m; edges farther than this are not walls
    free_range: float = 1.6      # m of each ray that frees the floor
    ratio: float = 0.35          # occupied / (occupied + free) of a wall cell
    min_occ: float = 0.6         # smoothed edge count of a wall cell
    min_free: float = 2.0        # smoothed ray count of a free cell
    reach: float = 0.9           # m from the driven path the track can extend
    seal: int = 3                # cells; wall gaps up to twice this are closed
    car_radius: int = 3          # cells around the driven path that cannot be wall
    close: int = 10              # cells; notches in the outer wall narrower than twice this are filled
    grow: int = 8                # cells the region grows to reach the walls
    trim: int = 6                # cells; necks narrower than twice this are cut
    smooth: int = 11             # points, moving average of the wall contours
    spacing: float = 0.1         # m between output boundary points


def signed_area(p: np.ndarray) -> float:
    return 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


def scan_to_samples(ranges: np.ndarray, angles: np.ndarray, origin: np.ndarray, heading: float,
                    cfg: OccupancyConfig) -> np.ndarray:
    """One scan in its origin's frame -> (N, 5) [edge x, edge y, range, origin x, origin y], world."""
    ok = np.isfinite(ranges) & (ranges > 0.05) & (ranges < cfg.edge_range + 2.5)
    ok[np.arange(len(ranges)) % cfg.beam_step != 0] = False
    c, s = np.cos(heading), np.sin(heading)
    fx, fy = ranges[ok] * np.cos(angles[ok]), ranges[ok] * np.sin(angles[ok])
    ox, oy = float(origin[0]), float(origin[1])
    n = int(ok.sum())
    return np.column_stack([ox + fx * c - fy * s, oy + fx * s + fy * c, ranges[ok],
                            np.full(n, ox), np.full(n, oy)])


def occupancy(samples: np.ndarray, x0: float, y0: float, shape: tuple, cfg: OccupancyConfig):
    """(free, occupied) smoothed hit counts on the grid."""
    import cv2

    free, occ = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    h, w = shape

    def add(grid, x, y):
        i, j = ((y - y0) / cfg.res).astype(int), ((x - x0) / cfg.res).astype(int)
        inside = (i >= 0) & (i < h) & (j >= 0) & (j < w)
        np.add.at(grid, (i[inside], j[inside]), 1)

    ex, ey, r, ox, oy = samples.T
    for k in np.linspace(0.2, 0.92, 10):        # along each ray, stopping short of the edge
        m = r * k < cfg.free_range
        add(free, ox[m] + (ex[m] - ox[m]) * k, oy[m] + (ey[m] - oy[m]) * k)
    m = r < cfg.edge_range
    add(occ, ex[m], ey[m])
    return cv2.GaussianBlur(free, (0, 0), 1.5), cv2.GaussianBlur(occ, (0, 0), 1.0)


def _largest_on_path(mask: np.ndarray, path_cells: np.ndarray) -> np.ndarray:
    import cv2

    n, labels = cv2.connectedComponents(mask)
    on_path = labels[path_cells[:, 1], path_cells[:, 0]]
    on_path = on_path[on_path > 0]
    if len(on_path) == 0:
        raise ValueError("no free floor connected to the driven path")
    return (labels == np.bincount(on_path, minlength=n).argmax()).astype(np.uint8)


def walls_from_grid(free: np.ndarray, occ: np.ndarray, path_cells: np.ndarray, cfg: OccupancyConfig):
    """(outer, inner) contours in cells (x, y) and the drivable region. Rays stop at walls,
    so a gap in a wall does not let the region out."""
    import cv2

    wall = (occ / (occ + free + 1e-6) > cfg.ratio) & (occ > cfg.min_occ)
    # no wall where the car drove: a lighting edge across the floor looks like one, and the
    # camera only looks ahead, so nothing else clears it
    driven = np.ones(free.shape, np.uint8)
    driven[path_cells[:, 1], path_cells[:, 0]] = 0
    distance = cv2.distanceTransform(driven, cv2.DIST_L2, 5)
    wall &= distance > cfg.car_radius
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.seal + 1, 2 * cfg.seal + 1))
    seen = ((cv2.dilate(wall.astype(np.uint8), disc) == 0) & (free > cfg.min_free)).astype(np.uint8)
    region = _largest_on_path(seen, path_cells)
    # out to the walls (rays stop short of them), no further than reach from the driven path;
    # then cut off whatever got out through a gap in a wall
    solid = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(cfg.grow):
        region = cv2.dilate(region, solid) & (~wall).astype(np.uint8)
    region &= (distance < cfg.reach / cfg.res).astype(np.uint8)
    trim = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.trim + 1, 2 * cfg.trim + 1))
    region = _largest_on_path(cv2.morphologyEx(region, cv2.MORPH_OPEN, trim), path_cells)
    contours, _ = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    by_area = sorted(contours, key=lambda c: -abs(cv2.contourArea(c)))
    if len(by_area) < 2:
        raise ValueError("the drivable region has no hole: the lap is not closed")
    # the outer wall has no thin spikes into the track, the inner one has (hairpin tips):
    # only the outer contour is closed
    outside = cv2.drawContours(np.zeros_like(region), by_area[:1], -1, 1, cv2.FILLED)
    fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cfg.close + 1, 2 * cfg.close + 1))
    outside, _ = cv2.findContours(cv2.morphologyEx(outside, cv2.MORPH_CLOSE, fill),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outer = max(outside, key=lambda c: abs(cv2.contourArea(c)))
    return outer[:, 0].astype(np.float64), by_area[1][:, 0].astype(np.float64), region


class OccupancyTrackMapper:
    """Collects scans along a lap and extracts the two walls."""

    def __init__(self, cfg: OccupancyConfig | None = None) -> None:
        self.cfg = cfg or OccupancyConfig()
        self.reset()

    def reset(self) -> None:
        self._samples: list[tuple[float, np.ndarray]] = []
        self._disturbances: list[float] = []
        self._last_origin: np.ndarray | None = None
        self.grid = None

    @property
    def n_samples(self) -> int:
        return sum(len(s) for _, s in self._samples)

    def mark_disturbance(self, t: float) -> None:
        """A wall hit or a respawn at time t: scans around it are not used."""
        self._disturbances.append(float(t))

    def add_scan(self, t: float, ranges: np.ndarray, angles: np.ndarray, origin: np.ndarray, heading: float) -> bool:
        """A scan from `origin` (world x, y) facing `heading`. False if it was skipped
        (the car has not moved far enough since the last one used)."""
        origin = np.asarray(origin, dtype=np.float64)[:2]
        if self._last_origin is not None and np.linalg.norm(origin - self._last_origin) < self.cfg.min_move:
            return False
        self._last_origin = origin
        self._samples.append((float(t), scan_to_samples(np.asarray(ranges, np.float64), angles, origin,
                                                        heading, self.cfg)))
        return True

    def _kept(self) -> np.ndarray:
        d = np.asarray(self._disturbances)
        keep = [s for t, s in self._samples
                if not np.any((t > d - self.cfg.drop_before) & (t < d + self.cfg.drop_after))]
        return np.vstack(keep) if keep else np.empty((0, 5))

    def extract(self, path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Walls (outer, inner), (N, 2) world coordinates, both ordered along the driven
        direction of `path` ((T, 2) positions, the lap first). Raises ValueError if there is
        no closed track in the data."""
        cfg = self.cfg
        path = np.asarray(path, dtype=np.float64)[:, :2]
        samples = self._kept()
        if len(samples) == 0 or len(path) < 10:
            raise ValueError(f"not enough data ({len(samples)} edge samples, {len(path)} path points)")
        x0, y0 = path.min(axis=0) - cfg.pad
        w, h = np.ceil((path.max(axis=0) + cfg.pad - [x0, y0]) / cfg.res).astype(int)
        free, occ = occupancy(samples, x0, y0, (h, w), cfg)
        cells = np.unique(((path - [x0, y0]) / cfg.res).astype(int), axis=0)
        cells = cells[(cells[:, 0] >= 0) & (cells[:, 0] < w) & (cells[:, 1] >= 0) & (cells[:, 1] < h)]
        outer, inner, region = walls_from_grid(free, occ, cells, cfg)
        self.grid = (region, x0, y0, w, h)
        turn = signed_area(path[::10])
        lines = []
        for b in (outer, inner):
            b = (b + 0.5) * cfg.res + [x0, y0]
            b = _resample_closed(_smooth_closed(b, cfg.smooth), cfg.spacing)
            lines.append(b if signed_area(b) * turn > 0 else b[::-1])
        return lines[0], lines[1]
