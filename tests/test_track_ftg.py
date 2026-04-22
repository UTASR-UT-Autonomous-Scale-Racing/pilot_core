"""
test_track_ftg.py — TrackFTGController, the reactive Follow-the-Gap
controller that steers from a virtual scan of the BEV track boundaries.

Covers:
    - Construction / default parameters
    - step() returns (steering, throttle) in [-1, 1]
    - Virtual scan generation from BEV boundary points
    - Pre-processing (smoothing + clamp)
    - Safety bubble zeroing
    - Max-gap detection
    - Best-point selection (deepest & centroid strategies)
    - Corner-throttle scaling
    - Reset is a no-op (reactive controller)
    - Edge cases: empty boundaries, vehicle on a boundary, single-point lines
"""

from __future__ import annotations

import numpy as np
import pytest

from workers.track_ftg_controller import TrackFTGController, _angle_diff


# --- Helpers ---

def _oval_track(n_outer: int = 80, n_inner: int = 80):
    """Return (outer, inner) boundary arrays for a simple oval track."""
    t_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    t_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
    outer = np.column_stack([3.0 * np.cos(t_outer), 2.0 * np.sin(t_outer)])
    inner = np.column_stack([2.0 * np.cos(t_inner), 1.0 * np.sin(t_inner)])
    return outer, inner


def _straight_corridor(length: float = 10.0, width: float = 2.0, n: int = 50):
    """Parallel straight boundaries along the X axis."""
    xs = np.linspace(0, length, n)
    outer = np.column_stack([xs, np.full(n, width / 2)])
    inner = np.column_stack([xs, np.full(n, -width / 2)])
    return outer, inner


# --- Construction ---

class TestConstruction:

    def test_defaults(self):
        c = TrackFTGController()
        assert c.num_beams == 360
        assert pytest.approx(np.rad2deg(c.fov), abs=0.5) == 270.0
        assert c.max_range == 5.0
        assert c.gap_strategy == "deepest"

    def test_custom_params(self):
        c = TrackFTGController(
            num_beams=180, fov_deg=180.0, max_range=3.0,
            bubble_radius_deg=20, smooth_window=3,
            target_throttle=0.5, throttle_corner_scale=0.8,
            noise_std=0.0, gap_strategy="centroid",
        )
        assert c.num_beams == 180
        assert c.gap_strategy == "centroid"
        assert c.target_throttle == 0.5

    def test_beam_angles_shape(self):
        c = TrackFTGController(num_beams=100)
        assert c.beam_angles.shape == (100,)


# --- _angle_diff helper ---

class TestAngleDiff:

    @pytest.mark.parametrize("a, b, expected", [
        (0.0, 0.0, 0.0),
        (np.pi, 0.0, np.pi),       # or -pi, both are equivalent wraps
        (0.1, -0.1, pytest.approx(0.2, abs=1e-9)),
        (3.0, -3.0, pytest.approx(3.0 - (-3.0) - 2 * np.pi, abs=1e-9)),
    ])
    def test_known_values(self, a, b, expected):
        result = _angle_diff(a, b)
        assert -np.pi <= result <= np.pi
        if expected != pytest.approx(np.pi, abs=1e-9):
            assert result == expected


# --- step() basic contract ---

class TestStepContract:

    def test_returns_tuple_of_two_floats(self):
        c = TrackFTGController(noise_std=0.0)
        outer, inner = _oval_track()
        pos = np.array([0.0, 0.0])
        steer, throttle = c.step(pos, 0.0, outer, inner)
        assert isinstance(steer, float)
        assert isinstance(throttle, float)

    def test_output_range(self):
        c = TrackFTGController(noise_std=0.0)
        outer, inner = _oval_track()
        for heading in np.linspace(-np.pi, np.pi, 12):
            s, t = c.step(np.array([0.0, 0.0]), heading, outer, inner)
            assert -1.0 <= s <= 1.0, f"steering {s} out of range"
            assert -1.0 <= t <= 1.0, f"throttle {t} out of range"

    def test_straight_corridor_steers_within_bounds(self):
        """On a straight corridor, centred vehicle heading along X → valid output."""
        c = TrackFTGController(noise_std=0.0, num_beams=360)
        outer, inner = _straight_corridor(length=10.0, width=2.0)
        pos = np.array([1.0, 0.0])
        steer, throttle = c.step(pos, 0.0, outer, inner)
        assert -1.0 <= steer <= 1.0
        assert throttle > 0.0  # should still be moving forward


# --- Virtual scan ---

class TestVirtualScan:

    def test_scan_shape(self):
        c = TrackFTGController(num_beams=180, noise_std=0.0)
        outer, inner = _oval_track()
        scan = c.get_virtual_scan(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert scan.shape == (180,)

    def test_scan_values_within_max_range(self):
        c = TrackFTGController(max_range=5.0, noise_std=0.0)
        outer, inner = _oval_track()
        scan = c.get_virtual_scan(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert np.all(scan <= 5.0)
        assert np.all(scan >= 0.0)

    def test_closer_boundary_gives_shorter_range(self):
        """Inner boundary is closer → some beams should read well below max."""
        c = TrackFTGController(max_range=10.0, num_beams=360, noise_std=0.0)
        outer, inner = _oval_track()
        scan = c._build_virtual_scan(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert scan.min() < 3.0  # inner ellipse radius ~ 1-2 units


# --- Pre-processing ---

class TestPreprocess:

    def test_clamps_to_max_range(self):
        c = TrackFTGController(max_range=3.0, smooth_window=1)
        raw = np.array([1.0, 5.0, 2.0, 10.0])
        result = c._preprocess(raw)
        assert np.all(result <= 3.0)

    def test_smoothing_reduces_spikes(self):
        c = TrackFTGController(smooth_window=5, max_range=100.0)
        raw = np.zeros(20)
        raw[10] = 50.0  # single spike
        result = c._preprocess(raw)
        assert result[10] < 50.0  # spike should be attenuated

    def test_no_smoothing_when_window_1(self):
        c = TrackFTGController(smooth_window=1, max_range=100.0)
        raw = np.array([1.0, 2.0, 3.0])
        result = c._preprocess(raw)
        np.testing.assert_array_equal(result, raw)


# --- Safety bubble ---

class TestBubble:

    def test_bubble_zeros_around_closest(self):
        c = TrackFTGController(num_beams=20, bubble_radius_deg=36, noise_std=0.0)
        # beam_step ≈ fov/num_beams; make ranges with a clear closest point
        ranges = np.full(20, 5.0)
        ranges[10] = 0.5  # closest hit
        result = c._apply_bubble(ranges)
        assert result[10] == 0.0
        # At least a few neighbours should also be zeroed
        assert np.sum(result == 0.0) > 1

    def test_bubble_does_not_modify_original(self):
        c = TrackFTGController(num_beams=10, noise_std=0.0)
        ranges = np.full(10, 3.0)
        ranges[5] = 0.2
        original = ranges.copy()
        _ = c._apply_bubble(ranges)
        np.testing.assert_array_equal(ranges, original)


# --- Gap detection ---

class TestMaxGap:

    def test_single_gap(self):
        ranges = np.array([0, 0, 1, 1, 1, 0, 0], dtype=float)
        start, end = TrackFTGController._find_max_gap(ranges)
        assert (start, end) == (2, 4)

    def test_two_gaps_picks_longer(self):
        ranges = np.array([0, 1, 0, 1, 1, 1, 0], dtype=float)
        start, end = TrackFTGController._find_max_gap(ranges)
        assert (start, end) == (3, 5)

    def test_all_free(self):
        ranges = np.ones(10)
        start, end = TrackFTGController._find_max_gap(ranges)
        assert start == 0
        assert end == 9

    def test_all_blocked(self):
        ranges = np.zeros(10)
        start, end = TrackFTGController._find_max_gap(ranges)
        # Fallback: midpoint
        assert start == end == 5

    def test_gap_at_end(self):
        ranges = np.array([0, 0, 0, 1, 1], dtype=float)
        start, end = TrackFTGController._find_max_gap(ranges)
        assert (start, end) == (3, 4)


# --- Best point selection ---

class TestBestPoint:

    def test_deepest_strategy(self):
        c = TrackFTGController(gap_strategy="deepest", noise_std=0.0)
        ranges = np.array([0, 0, 1.0, 3.0, 2.0, 0], dtype=float)
        idx = c._find_best_point(2, 4, ranges)
        assert idx == 3  # deepest

    def test_centroid_strategy(self):
        c = TrackFTGController(gap_strategy="centroid", noise_std=0.0)
        idx = c._find_best_point(2, 6, np.ones(10))
        assert idx == 4  # midpoint of [2..6]


# --- Throttle scaling ---

class TestThrottleScaling:

    def test_full_steer_reduces_throttle(self):
        c = TrackFTGController(
            target_throttle=0.5, throttle_corner_scale=0.5, noise_std=0.0
        )
        # Force steering to a large value by placing boundaries to one side
        outer = np.array([[0.5, 2.0], [1.0, 2.0], [2.0, 2.0]])
        inner = np.array([[0.5, 1.5], [1.0, 1.5], [2.0, 1.5]])
        pos = np.array([0.0, 0.0])
        _, throttle = c.step(pos, 0.0, outer, inner)
        # Hard steering must reduce throttle well below the 0.5 base
        assert throttle < 0.45

    def test_straight_throttle_near_base(self):
        c = TrackFTGController(
            target_throttle=0.4, throttle_corner_scale=0.5, noise_std=0.0
        )
        outer, inner = _straight_corridor()
        pos = np.array([1.0, 0.0])
        _, throttle = c.step(pos, 0.0, outer, inner)
        assert throttle > 0.0


# --- Reset ---

class TestReset:

    def test_reset_is_callable(self):
        c = TrackFTGController()
        c.reset()  # should not raise

    def test_step_after_reset(self):
        c = TrackFTGController(noise_std=0.0)
        outer, inner = _oval_track()
        c.step(np.array([0.0, 0.0]), 0.0, outer, inner)
        c.reset()
        s, t = c.step(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert -1.0 <= s <= 1.0
        assert -1.0 <= t <= 1.0


# --- Edge cases ---

class TestEdgeCases:

    def test_empty_outer(self):
        c = TrackFTGController(noise_std=0.0)
        _, inner = _oval_track()
        outer = np.empty((0, 2))
        s, t = c.step(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert -1.0 <= s <= 1.0

    def test_empty_inner(self):
        c = TrackFTGController(noise_std=0.0)
        outer, _ = _oval_track()
        inner = np.empty((0, 2))
        s, t = c.step(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert -1.0 <= s <= 1.0

    def test_single_point_boundaries(self):
        c = TrackFTGController(noise_std=0.0)
        outer = np.array([[3.0, 0.0]])
        inner = np.array([[1.0, 0.0]])
        s, t = c.step(np.array([0.0, 0.0]), 0.0, outer, inner)
        assert -1.0 <= s <= 1.0

    def test_vehicle_on_boundary(self):
        """Vehicle sitting right on a boundary point should not crash."""
        c = TrackFTGController(noise_std=0.0)
        outer, inner = _oval_track()
        pos = outer[0].copy()
        s, t = c.step(pos, 0.0, outer, inner)
        assert -1.0 <= s <= 1.0

    def test_deterministic_when_noise_zero(self):
        c = TrackFTGController(noise_std=0.0, num_beams=180)
        outer, inner = _oval_track()
        pos = np.array([0.0, 0.0])
        results = [c.step(pos, 0.0, outer, inner) for _ in range(5)]
        for s, t in results:
            assert s == results[0][0]
            assert t == results[0][1]
