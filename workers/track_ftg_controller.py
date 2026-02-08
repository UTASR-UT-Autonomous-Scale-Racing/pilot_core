"""
TrackFTGController — Follow-the-Gap reactive controller that steers
the vehicle through the widest open corridor between track boundaries
projected onto a 2-D bird's-eye-view plane.

Instead of raw LiDAR ranges the algorithm synthesises a virtual
1-D scan from the BEV boundary polylines (outer / inner track edges).
The classic Follow-the-Gap pipeline then runs on that scan:

  1. Build a virtual polar range array from boundary points.
  2. Smooth the ranges (moving-average window).
  3. Reject / clamp ranges beyond a max horizon.
  4. Find the closest obstacle and zero out a safety bubble around it.
  5. Find the longest contiguous gap of non-zero ranges.
  6. Pick the best steering point inside that gap (deepest or centroid).
  7. Convert the chosen bearing into a normalised steering command.

Output: (steering, throttle) both in [-1, 1], with optional Gaussian
noise for stochastic sampling — matching TrackPIDController's API.
"""

from __future__ import annotations

import numpy as np


def _angle_diff(a: float, b: float) -> float:
    """Signed shortest angular difference in radians."""
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class TrackFTGController:
    """Follow-the-Gap controller for autonomous track following.

    Shares the same ``step(current_bev, heading_rad, outer_line, inner_line)``
    signature as :class:`TrackPIDController` so the two can be swapped
    without changing calling code.

    Parameters
    ----------
    num_beams : int
        Resolution of the virtual scan (default 360).
    fov_deg : float
        Total field-of-view in degrees centred on the heading
        (default 270 — like a typical LiDAR).
    max_range : float
        Clamp / reject distances beyond this (BEV units, default 5.0).
    bubble_radius_deg : float
        Half-width of the safety bubble in degrees (default 30).
    smooth_window : int
        Moving-average kernel size for range pre-processing (default 5).
    target_throttle : float
        Constant throttle baseline in [-1, 1] (default 0.3).
    throttle_corner_scale : float
        Throttle is reduced proportionally to |steering| by this factor
        (default 0.5).  ``throttle = base * (1 - scale * |steer|)``.
    noise_std : float
        Gaussian noise std for stochastic sampling (default 0.05).
    gap_strategy : str
        ``"deepest"`` — steer toward the farthest point in the gap.
        ``"centroid"`` — steer toward the angular centre of the gap.
        (default ``"deepest"``).
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

        # Pre-compute angular bins (relative to heading = 0)
        half_fov = self.fov / 2.0
        self._beam_angles = np.linspace(-half_fov, half_fov, num_beams)
        # Width of one angular bin (used for bubble masking)
        self._beam_step = self._beam_angles[1] - self._beam_angles[0] if num_beams > 1 else self.fov

    # ------------------------------------------------------------------
    # Public API (matches TrackPIDController)
    # ------------------------------------------------------------------

    def step(
        self,
        current_bev: np.ndarray,
        heading_rad: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> tuple[float, float]:
        """Compute one Follow-the-Gap control step.

        Parameters
        ----------
        current_bev : np.ndarray
            ``(2,)`` current vehicle position in BEV (X, Z).
        heading_rad : float
            Vehicle heading angle in the BEV (X, Z) plane.
        outer_line : np.ndarray
            ``(N, 2)`` ordered outer boundary vertices.
        inner_line : np.ndarray
            ``(M, 2)`` ordered inner boundary vertices.

        Returns
        -------
        (steering, throttle) — both clipped to [-1, 1].
        """
        # 1. Build virtual range scan from boundary polylines
        ranges = self._build_virtual_scan(current_bev, heading_rad, outer_line, inner_line)

        # 2. Pre-process (smooth + clamp)
        ranges = self._preprocess(ranges)

        # 3. Safety bubble around closest point
        ranges = self._apply_bubble(ranges)

        # 4. Find widest gap
        start_i, end_i = self._find_max_gap(ranges)

        # 5. Choose best aim point in the gap
        best_idx = self._find_best_point(start_i, end_i, ranges)

        # 6. Convert beam index → steering angle relative to heading
        steer_angle = self._beam_angles[best_idx]

        # Normalise: map [-half_fov, +half_fov] → [-1, +1]
        half_fov = self.fov / 2.0
        raw_steer = float(np.clip(steer_angle / half_fov, -1.0, 1.0))

        # 7. Throttle: slow down in corners
        raw_throttle = self.target_throttle * (
            1.0 - self.throttle_corner_scale * abs(raw_steer)
        )

        # 8. Stochastic sampling
        steer = float(np.clip(
            raw_steer + np.random.normal(0, self.noise_std), -1.0, 1.0
        ))
        throttle = float(np.clip(
            raw_throttle + np.random.normal(0, self.noise_std * 0.5), -1.0, 1.0
        ))

        return steer, throttle

    def reset(self) -> None:
        """No internal state to reset (reactive controller)."""
        pass

    # ------------------------------------------------------------------
    # Virtual LiDAR scan from BEV boundary polylines
    # ------------------------------------------------------------------

    def _build_virtual_scan(
        self,
        origin: np.ndarray,
        heading: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> np.ndarray:
        """Project boundary points into a 1-D range array.

        For every boundary vertex compute the bearing relative to the
        vehicle heading and the distance, then populate the closest-hit
        range in the corresponding angular bin.
        """
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float64)
        half_fov = self.fov / 2.0

        for pts in (outer_line, inner_line):
            if pts.size == 0:
                continue
            deltas = pts - origin  # (N, 2)
            dists = np.linalg.norm(deltas, axis=1)

            # Bearing of each point relative to heading
            bearings = np.arctan2(deltas[:, 1], deltas[:, 0]) - heading
            # Wrap to [-pi, pi)
            bearings = (bearings + np.pi) % (2 * np.pi) - np.pi

            for bearing, dist in zip(bearings, dists):
                if abs(bearing) > half_fov:
                    continue  # outside FOV
                # Map bearing → beam index
                idx = int(round((bearing + half_fov) / self._beam_step))
                idx = np.clip(idx, 0, self.num_beams - 1)
                # Keep the nearest hit
                if dist < ranges[idx]:
                    ranges[idx] = dist

        return ranges

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess(self, ranges: np.ndarray) -> np.ndarray:
        """Smooth with a moving average and clamp to max_range."""
        if self.smooth_window > 1:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            ranges = np.convolve(ranges, kernel, mode="same")
        return np.clip(ranges, 0.0, self.max_range)

    # ------------------------------------------------------------------
    # Safety bubble
    # ------------------------------------------------------------------

    def _apply_bubble(self, ranges: np.ndarray) -> np.ndarray:
        """Zero out beams within ``bubble_radius`` of the closest hit."""
        closest_idx = int(np.argmin(ranges))
        # Number of beams covered by the bubble radius
        n_bubble = max(1, int(round(self.bubble_radius / self._beam_step)))
        lo = max(0, closest_idx - n_bubble)
        hi = min(self.num_beams, closest_idx + n_bubble + 1)
        ranges = ranges.copy()
        ranges[lo:hi] = 0.0
        return ranges

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    @staticmethod
    def _find_max_gap(ranges: np.ndarray) -> tuple[int, int]:
        """Return (start, end) indices of the longest contiguous non-zero run."""
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
        # Final run
        if cur_len > best_len:
            best_start, best_len = cur_start, cur_len

        if best_len == 0:
            # Fallback: entire scan blocked — point straight ahead
            mid = len(ranges) // 2
            return mid, mid

        return best_start, best_start + best_len - 1

    def _find_best_point(
        self, start_i: int, end_i: int, ranges: np.ndarray
    ) -> int:
        """Choose the aim point inside the gap.

        ``"deepest"``  — index with the largest range value.
        ``"centroid"`` — angular midpoint of the gap.
        """
        if self.gap_strategy == "centroid":
            return (start_i + end_i) // 2
        # default: deepest
        gap_slice = ranges[start_i: end_i + 1]
        return int(start_i + np.argmax(gap_slice))

    # ------------------------------------------------------------------
    # Introspection helpers (useful for debug / visualisation)
    # ------------------------------------------------------------------

    @property
    def beam_angles(self) -> np.ndarray:
        """Angular position of each virtual beam (radians, relative to heading)."""
        return self._beam_angles.copy()

    def get_virtual_scan(
        self,
        current_bev: np.ndarray,
        heading_rad: float,
        outer_line: np.ndarray,
        inner_line: np.ndarray,
    ) -> np.ndarray:
        """Return the preprocessed + bubble-masked virtual scan without
        computing a control action.  Useful for visualisation overlays.
        """
        raw = self._build_virtual_scan(current_bev, heading_rad, outer_line, inner_line)
        processed = self._preprocess(raw)
        return self._apply_bubble(processed)
