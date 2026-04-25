"""
Follow-the-Gap on the mapped track boundaries.

There's no real LiDAR here: we fake a 1-D polar scan from the BEV outer/inner
boundary polylines and run the usual FTG steps on it. Smooth, clamp to
max_range, zero a safety bubble around the closest hit, take the longest run
of non-zero beams, aim at its deepest point (or its centre), and scale that
bearing to a steering command.

Sezer & Gokasan, "A novel obstacle avoidance algorithm: 'Follow the Gap
Method'", Robotics and Autonomous Systems 60(9), 2012.

Same interface as TrackPIDController: (steering, throttle) in [-1, 1],
+1 steer = left, optional Gaussian noise on both.
"""

from __future__ import annotations

import numpy as np


def _angle_diff(a: float, b: float) -> float:
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _wrap_angle(a: float) -> float:
    """-> [-pi, pi)"""
    return (a + np.pi) % (2 * np.pi) - np.pi


class TrackFTGController:
    """Follow-the-Gap track follower.

    step() has the same signature as TrackPIDController.step so the two are
    drop-in replacements for each other.

    Parameters
    ----------
    num_beams : int
        Virtual scan resolution.
    fov_deg : float
        Total FOV centred on the heading. 270 by default, roughly what a
        real LiDAR sees.
    max_range : float
        Metres. Anything further is clamped.
    bubble_radius_deg : float
        Half-width of the safety bubble.
    smooth_window : int
        Moving-average kernel length.
    target_throttle : float
        Throttle on a straight, in [-1, 1].
    throttle_corner_scale : float
        throttle = base * (1 - scale * |steer|)
    noise_std : float
        Steering noise std (throttle gets half).
    gap_strategy : str
        "deepest" aims at the farthest beam in the gap, "centroid" at the
        middle of it.
    """

    def __init__(
        self,
        num_beams: int = 360,
        fov_deg: float = 270.0,
        max_range: float = 5.0,
        bubble_radius_deg: float = 30.0,
        smooth_window: int = 5,
        target_throttle: float = 0.3,
        throttle_corner_scale: float = 0.5,
        noise_std: float = 0.05,
        gap_strategy: str = "deepest",
    ) -> None:
        self.num_beams = num_beams
        self.fov = np.deg2rad(fov_deg)
        self.max_range = max_range
        self.bubble_radius = np.deg2rad(bubble_radius_deg)
        self.smooth_window = smooth_window
        self.target_throttle = target_throttle
        self.throttle_corner_scale = throttle_corner_scale
        self.noise_std = noise_std
        self.gap_strategy = gap_strategy

        # beam angles relative to the heading
        half_fov = self.fov / 2.0
        self._beam_angles = np.linspace(-half_fov, half_fov, num_beams)
        self._beam_step = self._beam_angles[1] - self._beam_angles[0] if num_beams > 1 else self.fov

    def step(
        self,
        current_bev: np.ndarray,
        heading_rad: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> tuple[float, float]:
        """Arguments as in TrackPIDController.step (BEV position, heading,
        ordered outer/inner boundary vertices). Returns (steering, throttle)
        clipped to [-1, 1].
        """
        ranges = self._build_virtual_scan(current_bev, heading_rad, outer_line, inner_line)
        ranges = self._preprocess(ranges)
        ranges = self._apply_bubble(ranges)

        start_i, end_i = self._find_max_gap(ranges)
        best_idx = self._find_best_point(start_i, end_i, ranges)

        steer_angle = self._beam_angles[best_idx]

        # +-half_fov -> +-1, positive bearing is left
        half_fov = self.fov / 2.0
        raw_steer = float(np.clip(steer_angle / half_fov, -1.0, 1.0))

        # slow down in corners
        raw_throttle = self.target_throttle * (
            1.0 - self.throttle_corner_scale * abs(raw_steer)
        )

        steer = float(np.clip(
            raw_steer + np.random.normal(0, self.noise_std), -1.0, 1.0
        ))
        throttle = float(np.clip(
            raw_throttle + np.random.normal(0, self.noise_std * 0.5), -1.0, 1.0
        ))

        return steer, throttle

    def reset(self) -> None:
        """Nothing to do, FTG is purely reactive."""

    def _build_virtual_scan(
        self,
        origin: np.ndarray,
        heading: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> np.ndarray:
        """Boundary vertices -> 1-D range array.

        Each vertex is binned by its bearing from the heading, and each bin
        keeps the nearest distance. Bins with no hits stay at max_range.
        """
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float64)
        half_fov = self.fov / 2.0

        for pts in (outer_line, inner_line):
            if pts.size == 0:
                continue
            deltas = pts - origin  # (N, 2)
            dists = np.linalg.norm(deltas, axis=1)

            bearings = np.arctan2(deltas[:, 1], deltas[:, 0]) - heading
            bearings = (bearings + np.pi) % (2 * np.pi) - np.pi

            for bearing, dist in zip(bearings, dists):
                if abs(bearing) > half_fov:
                    continue
                idx = int(round((bearing + half_fov) / self._beam_step))
                idx = np.clip(idx, 0, self.num_beams - 1)
                if dist < ranges[idx]:
                    ranges[idx] = dist

        return ranges

    def _preprocess(self, ranges: np.ndarray) -> np.ndarray:
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            ranges = np.convolve(ranges, kernel, mode="same")
        return np.clip(ranges, 0.0, self.max_range)

    def _apply_bubble(self, ranges: np.ndarray) -> np.ndarray:
        """Zero every beam within bubble_radius of the closest hit."""
        closest_idx = int(np.argmin(ranges))
        n_bubble = max(1, int(round(self.bubble_radius / self._beam_step)))
        lo = max(0, closest_idx - n_bubble)
        hi = min(self.num_beams, closest_idx + n_bubble + 1)
        ranges = ranges.copy()
        ranges[lo:hi] = 0.0
        return ranges

    @staticmethod
    def _find_max_gap(ranges: np.ndarray) -> tuple[int, int]:
        """(start, end) of the longest run of non-zero beams, end inclusive."""
        is_free = ranges > 0.0
        best_start, best_len = 0, 0
        cur_start, cur_len = 0, 0

        for i, free in enumerate(is_free):
            if free:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
            else:
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
                cur_len = 0
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        if best_len == 0:
            # everything blocked, just aim straight ahead
            mid = len(ranges) // 2
            return mid, mid

        return best_start, best_start + best_len - 1

    def _find_best_point(
        self, start_i: int, end_i: int, ranges: np.ndarray
    ) -> int:
        if self.gap_strategy == "centroid":
            return (start_i + end_i) // 2
        # anything else -> deepest
        gap_slice = ranges[start_i: end_i + 1]
        return int(start_i + np.argmax(gap_slice))

    # debug / visualisation helpers

    @property
    def beam_angles(self) -> np.ndarray:
        """Beam angles in rad, relative to the heading."""
        return self._beam_angles.copy()

    def get_virtual_scan(
        self,
        current_bev: np.ndarray,
        heading_rad: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> np.ndarray:
        """The smoothed, bubble-masked scan that step() would act on, for overlays."""
        raw = self._build_virtual_scan(current_bev, heading_rad, outer_line, inner_line)
        processed = self._preprocess(raw)
        return self._apply_bubble(processed)
