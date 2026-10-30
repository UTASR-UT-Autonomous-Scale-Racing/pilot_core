"""control.ImuEKF: IMU-driven planar EKF with late SLAM poses (rear-IMU state estimation)."""

from __future__ import annotations

import numpy as np
import pytest

from control import ImuEKF, ImuEstimatorConfig, PoseEKF


def _circle(t: np.ndarray, v: float = 3.0, r: float = 1.0):
    """IMU point on a circle at speed v, yaw rate r, no slip: pose and body-frame IMU."""
    yaw = r * t
    R = v / r
    x, y = R * np.sin(yaw), R * (1 - np.cos(yaw))
    imu = np.column_stack([np.zeros_like(t), np.full_like(t, v * r), np.full_like(t, r)])   # ax, ay, gz
    return x, y, yaw, imu


def _run(ekf, pose_period=1 / 30, latency=0.03, gap=None, T=6.0):
    t_imu = np.arange(0, T, 0.01)
    x, y, yaw, imu = _circle(t_imu)
    t_pose = np.arange(0, T, pose_period)
    px, py, pyaw, _ = _circle(t_pose)
    events = [(t, 0, i) for i, t in enumerate(t_imu)]
    events += [(t + latency, 1, i) for i, t in enumerate(t_pose) if gap is None or not (gap[0] <= t < gap[1])]
    errs = []
    for t_arr, kind, i in sorted(events):
        if kind == 0:
            if hasattr(ekf, "update_imu"):
                ekf.update_imu(t_imu[i], *imu[i])
            st = ekf.predict_to(t_imu[i])
            if st is not None and t_imu[i] > 1.0:
                errs.append((t_imu[i], np.hypot(st.x - x[i], st.y - y[i])))
        else:
            ekf.update_pose(t_pose[i], px[i], py[i], pyaw[i])
    return np.array(errs)


def test_tracks_a_circle_with_late_poses():
    e = _run(ImuEKF())
    assert e[:, 1].max() < 0.02


def test_bridges_a_pose_gap_better_than_the_pose_only_ekf():
    gap = (3.0, 3.5)
    imu = _run(ImuEKF(), gap=gap)
    pose = _run(PoseEKF(), gap=gap)
    in_gap = lambda e: e[(e[:, 0] >= gap[0]) & (e[:, 0] < gap[1] + 0.1), 1]
    assert in_gap(imu).max() < 0.03
    assert in_gap(pose).max() > 3 * in_gap(imu).max()


def test_a_late_pose_gives_the_same_state_as_an_on_time_one():
    a, b = ImuEKF(), ImuEKF()
    t = np.arange(0, 2, 0.01)
    x, y, yaw, imu = _circle(t)
    for ekf in (a, b):
        ekf.update_pose(0.0, x[0], y[0], yaw[0])
    for i in range(1, 100):                       # a: pose at t=0.5 applied in order
        a.update_imu(t[i], *imu[i])
        if i == 50:
            a.update_pose(t[50], x[50], y[50], yaw[50])
    for i in range(1, 100):                       # b: the same pose arrives at t=0.99
        b.update_imu(t[i], *imu[i])
    b.update_pose(t[50], x[50], y[50], yaw[50])
    np.testing.assert_allclose(a.x, b.x, atol=1e-9)
    np.testing.assert_allclose(a.P, b.P, atol=1e-9)


def test_estimates_sideslip():
    """Body drifting sideways at 0.5 m/s while going 3 m/s straight: vy is observed through poses
    (without the no-slip pseudo-measurement, which is for a rear axle)."""
    ekf = ImuEKF(ImuEstimatorConfig(vy_std=None))
    for k in range(300):
        t = k * 0.01
        ekf.update_imu(t, 0.0, 0.0, 0.0)
        if k % 3 == 0:
            ekf.update_pose(t, 3.0 * t, 0.5 * t, 0.0)
    assert ekf.x[4] == pytest.approx(0.5, abs=0.05)
    assert ekf.sideslip == pytest.approx(np.arctan2(0.5, 3.0), abs=0.02)


def test_invalid_without_poses():
    ekf = ImuEKF(ImuEstimatorConfig(stale_after=0.5))
    assert ekf.predict_to(0.0) is None
    ekf.update_pose(0.0, 0.0, 0.0, 0.0)
    assert ekf.predict_to(0.4) is not None and ekf.predict_to(0.6) is None


def test_rejects_an_outlier_pose_and_reanchors_on_a_lasting_offset():
    ekf = ImuEKF()
    t = np.arange(0, 3, 0.01)
    x, y, yaw, imu = _circle(t)
    for i, ti in enumerate(t):
        ekf.update_imu(ti, *imu[i])
        if i % 3 == 0:
            off = 1.0 if 1.5 <= ti < 1.6 else 0.0          # a 0.1 s burst of poses 1 m off
            ekf.update_pose(ti, x[i] + off, y[i], yaw[i])
    assert ekf.rejected > 0
    assert np.hypot(ekf.x[0] - x[-1], ekf.x[1] - y[-1]) < 0.05
    ekf = ImuEKF(ImuEstimatorConfig(gate_reset_s=0.5))
    for i, ti in enumerate(t):                              # SLAM moves its frame by 1 m for good
        ekf.update_imu(ti, *imu[i])
        if i % 3 == 0:
            ekf.update_pose(ti, x[i] + (1.0 if ti >= 1.5 else 0.0), y[i], yaw[i])
    assert abs(ekf.x[0] - (x[-1] + 1.0)) < 0.05


def test_a_pose_before_any_imu_then_a_late_pose():
    """The first pose can come before the first IMU sample; replaying from it must work."""
    ekf = ImuEKF()
    ekf.update_pose(0.0, 0.0, 0.0, 0.0)
    for k in range(1, 20):
        ekf.update_imu(k * 0.01, 0.0, 0.0, 0.0)
    ekf.update_pose(0.005, 0.0, 0.0, 0.0)          # late, between the first pose and the first IMU sample
    assert np.isfinite(ekf.x).all()
