"""control/pose_degrade.py: the sim's ground-truth pose published like a SLAM pose."""

from __future__ import annotations

import numpy as np
import pytest

from control.pose_degrade import PoseDegradeConfig, PoseDegrader


def _poses(n: int = 2000, dt: float = 0.02):
    for i in range(n):
        T = np.eye(4, dtype=np.float32)
        T[0, 3] = 0.05 * i
        yield i * dt, T


def test_poses_are_late_in_order_and_keep_their_capture_time():
    d = PoseDegrader(PoseDegradeConfig(enabled=True, dropout_rate=0.0, seed=1))
    out, delays = [], []
    for t, T in _poses():
        d.add(t, T)
        for tc, pose in d.due(t):
            out.append(tc)
            delays.append(t - tc)
    assert out == sorted(out)
    delays = np.array(delays)
    assert np.median(delays) == pytest.approx(0.033 + 0.0084, abs=0.02)   # 20 ms message steps
    assert delays.min() >= 0.033


def test_noise_and_drift_are_bounded():
    d = PoseDegrader(PoseDegradeConfig(enabled=True, dropout_rate=0.0, seed=2))
    err = []
    for t, T in _poses(n=600):                          # 12 s, about one lap
        d.add(t, T)
        for tc, pose in d.due(t + 1.0):
            err.append(abs(float(pose[0, 3]) - 0.05 * round(tc / 0.02)))
    assert 0.005 < np.median(err) < 0.2


def test_dropouts_remove_stretches_of_poses():
    cfg = PoseDegradeConfig(enabled=True, dropout_rate=0.5, dropout_s=0.3, seed=3)
    d = PoseDegrader(cfg)
    for t, T in _poses(n=3000):
        d.add(t, T)
    d.due(1e9)
    lost = d.dropped / (d.dropped + d.published)
    expected = 0.5 * 0.3 / (1 + 0.5 * 0.3)
    assert lost == pytest.approx(expected, abs=0.05)


def test_the_bias_stays_bounded_over_a_long_race():
    d = PoseDegrader(PoseDegradeConfig(enabled=True, dropout_rate=0.0, xy_sigma=0.0, seed=4))
    bias = []
    for i in range(6000):                               # 240 s at 25 Hz
        d.add(0.04 * i, np.eye(4, dtype=np.float32))
        bias.append(float(np.hypot(*d._bias)))
    assert np.sqrt(np.mean(np.square(bias))) == pytest.approx(0.07 * np.sqrt(2), rel=0.3)
    assert max(bias) < 0.4
