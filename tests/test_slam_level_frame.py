"""SLAMWorker's floor-aligned world frame, fitted from the synthetic depth
(camera pitched 12 deg down, 0.15 m above the floor)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from synthetic.closed_loop_generator import ClosedLoopFrameGenerator
from workers.slam_worker import SLAMWorker, level_rotation, read_settings_intrinsics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(ROOT, "synthetic_slam_settings.yaml")


def test_level_rotation_is_proper_and_keeps_y_down():
    n = np.array([0.1, 0.95, 0.3])
    R = level_rotation(n)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(R @ (n / np.linalg.norm(n)), [0, 1, 0], atol=1e-6)
    assert (R @ [0, 0, 1])[0] == pytest.approx(0.0, abs=1e-6)   # forward stays in the Y-Z plane


def test_settings_intrinsics_match_synthetic_camera():
    K = read_settings_intrinsics(SETTINGS)
    gen = ClosedLoopFrameGenerator(n_frames=10)
    assert (K.width, K.height) == (gen.W, gen.H)
    assert K.fx == pytest.approx(gen.fx) and K.cy == pytest.approx(gen.cy)


def test_floor_fit_levels_the_pitched_camera():
    gen = ClosedLoopFrameGenerator(n_frames=100, straight_len=2.0, turn_radius=1.0)
    w = SLAMWorker(name="slam", level_frames=3)
    w._logger = MagicMock()
    w._K = read_settings_intrinsics(SETTINGS)
    x, z, heading = gen._ellipse_path(0.0)
    for _ in range(3):
        _, depth = gen._render_frame_fast(x, z, heading)
        w._update_level(depth)
    R = w._R_level
    assert R is not None
    # Camera frame at map init = ORB-SLAM3 world frame. The ground-truth
    # camera-to-world rotation is R_yaw @ R_pitch; levelling must undo the pitch.
    p = gen.pitch
    R_pitch = np.array([[1, 0, 0], [0, np.cos(p), np.sin(p)], [0, -np.sin(p), np.cos(p)]])
    np.testing.assert_allclose(R[:3, :3], R_pitch, atol=0.02)


def test_plane_above_the_camera_is_not_the_floor():
    # an upside-down camera sees the floor (or a ceiling) above it: never level on that
    gen = ClosedLoopFrameGenerator(n_frames=100, straight_len=2.0, turn_radius=1.0)
    w = SLAMWorker(name="slam", level_frames=3, level_timeout_frames=3)
    w._logger = MagicMock()
    w._K = read_settings_intrinsics(SETTINGS)
    x, z, heading = gen._ellipse_path(0.0)
    _, depth = gen._render_frame_fast(x, z, heading)
    for _ in range(3):
        w._update_level(np.ascontiguousarray(depth[::-1, ::-1]))   # camera rotated 180°
    assert w._plane_fits == []
    np.testing.assert_allclose(w._R_level, np.eye(4))          # falls back to the unlevelled frame
