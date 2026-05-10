"""MappingWorker boundary extraction: lap trimming, direction, wall filters,
map reset; plus LidarProjectionWorker's voxel filter."""

from __future__ import annotations

import multiprocessing as mp
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import PointCloudBuffer, PoseBuffer, TrackBoundaryBuffer
from workers.mapping_worker import (
    MappingWorker, clean_trajectory, extract_track_boundaries, floor_heights, signed_area,
)

A, B = 4.0, 2.5          # centre-line ellipse semi-axes (BEV)
HALF = 0.6               # track half-width


def _lap(n=200, cw=False, laps=1.0):
    phi = np.linspace(0, 2 * np.pi * laps, int(n * laps), endpoint=False)
    if cw:
        phi = -phi
    return np.column_stack([A * np.cos(phi), B * np.sin(phi)])


def _walls(n=400):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.column_stack([(A + HALF) * np.cos(th), (B + HALF) * np.sin(th)])
    inner = np.column_stack([(A - HALF) * np.cos(th), (B - HALF) * np.sin(th)])
    return outer, inner


def _mean_radius(line):
    return float(np.median(np.linalg.norm(line / [A, B], axis=1)))


def test_clean_trajectory_drops_stationary_and_trims_lap():
    lap = _lap(laps=1.4)
    traj = np.vstack([np.repeat(lap[:1], 20, 0), lap])   # parked at the start first
    out, closed = clean_trajectory(traj)
    assert closed
    assert len(out) < len(lap)                           # trimmed to about one lap
    assert np.linalg.norm(out[-1] - out[0]) < 0.5
    assert abs(len(out) - 200) <= 3


@pytest.mark.parametrize("cw", [False, True])
def test_outer_inner_correct_for_both_directions(cw):
    traj, closed = clean_trajectory(_lap(cw=cw, laps=1.2))
    assert (signed_area(traj) > 0) == (not cw)
    outer, inner = _walls()
    o, i, ccw = extract_track_boundaries(np.vstack([outer, inner]), None, traj, closed)
    assert ccw == (not cw)
    assert _mean_radius(o) > 1.0 > _mean_radius(i)


def test_filters_far_points_floor_and_points_behind_the_wall():
    traj, closed = clean_trajectory(_lap(laps=1.2))
    outer, inner = _walls()
    th = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    behind = np.column_stack([(A + HALF + 0.3) * np.cos(th), (B + HALF + 0.3) * np.sin(th)])
    far = np.column_stack([(A + 3.0) * np.cos(th), (B + 3.0) * np.sin(th)])
    floor = _lap(n=400)                                   # points on the driven line, on the floor
    pts = np.vstack([outer, inner, behind, far, floor])
    height = np.concatenate([np.full(1200, 0.2), np.full(400, 0.2), np.zeros(400)])
    o, i, _ = extract_track_boundaries(pts, height, traj, closed)
    # outer is the real wall, not the points behind it or far away
    r = np.linalg.norm(o / [A + HALF, B + HALF], axis=1)
    assert np.median(r) == pytest.approx(1.0, abs=0.03)
    ri = np.linalg.norm(i / [A - HALF, B - HALF], axis=1)
    assert np.median(ri) == pytest.approx(1.0, abs=0.03)   # the floor points were dropped


def test_strays_in_front_of_the_wall_and_gaps_in_a_thin_wall():
    # what a real SLAM map looks like: a dense outer wall, an inner wall seen only
    # in patches, and sparse stray points between the car and both walls
    rng = np.random.default_rng(0)
    traj, closed = clean_trajectory(_lap(laps=1.2))
    th = rng.uniform(0, 2 * np.pi, 4000)
    outer = np.column_stack([(A + HALF) * np.cos(th), (B + HALF) * np.sin(th)])
    th_in = rng.uniform(0, 2 * np.pi, 1500)
    th_in = th_in[np.sin(3 * th_in) > -0.3]               # three unseen stretches
    inner = np.column_stack([(A - HALF) * np.cos(th_in), (B - HALF) * np.sin(th_in)])
    walls = np.vstack([outer, inner]) + rng.normal(0, 0.02, (len(outer) + len(inner), 2))
    th_s = rng.uniform(0, 2 * np.pi, 600)
    r = rng.uniform(-HALF + 0.1, HALF - 0.1, 600)         # anywhere across the track
    strays = np.column_stack([(A + r) * np.cos(th_s), (B + r) * np.sin(th_s)])
    o, i, _ = extract_track_boundaries(np.vstack([walls, strays]), None, traj, closed)
    for line, (a, b) in ((o, (A + HALF, B + HALF)), (i, (A - HALF, B - HALF))):
        # distance to the true ellipse, approximately (fine for these proportions)
        err = np.abs(np.linalg.norm(line / [a, b], axis=1) - 1.0) * min(a, b)
        assert np.percentile(err, 95) < 0.08
    assert len(i) == len(o)                               # the gaps were bridged


def test_floor_heights_absorb_a_tilted_floor():
    rng = np.random.default_rng(1)
    xy = rng.uniform(-4, 4, (3000, 2))
    tilt = 0.02 * xy[:, 0]                                # ~1.1 deg, 8 cm across the map
    floor = np.column_stack([xy, tilt + rng.normal(0, 0.005, 3000)])
    wall = np.column_stack([rng.uniform(-4, 4, (500, 2)), rng.uniform(0.1, 0.5, 500)])
    wall[:, 2] += 0.02 * wall[:, 0]
    h = floor_heights(np.vstack([floor, wall]), floor_guess=0.0)
    assert h is not None
    assert np.abs(h[:3000]).max() < 0.03                  # floor at ~0 everywhere
    assert np.all(h[3000:] > 0.08)
    assert floor_heights(wall, floor_guess=5.0) is None   # no floor near the guess


def _mapper(**kw):
    uid = uuid.uuid4().hex[:8]
    pb = PoseBuffer(slots=4, name=f"mb_p_{uid}", create=True)
    pcb = PointCloudBuffer(slots=2, name=f"mb_pc_{uid}", create=True)
    kfb = PointCloudBuffer(slots=2, name=f"mb_kf_{uid}", create=True, max_points=2000)
    bb = TrackBoundaryBuffer(slots=2, name=f"mb_b_{uid}", create=True)
    w = MappingWorker(
        name="mapping", pose_buffer=pb, pointcloud_buffer=pcb, keyframe_buffer=kfb,
        boundary_buffer=bb, loop_event=mp.Event(), mapping_done_event=mp.Event(),
        map_reset_event=mp.Event(), boundary_min_points=10, **kw,
    )
    w._logger = MagicMock()
    w.setup()
    return w, (pb, pcb, kfb, bb)


def _to3d(xz, y=0.0):
    return np.column_stack([xz[:, 0], np.full(len(xz), y), xz[:, 1]]).astype(np.float32)


def test_worker_uses_keyframe_trajectory_and_clockwise_lap(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    w, bufs = _mapper()
    pb, pcb, kfb, bb = bufs
    try:
        outer, inner = _walls()
        pcb.write(_to3d(np.vstack([outer, inner]), y=-0.1))  # 0.25 m above the floor
        # live poses are junk (drifted); the keyframes are the corrected lap
        for k in range(5):
            T = np.eye(4, dtype=np.float32)
            T[0, 3] = 20.0 + k
            pb.write(T)
            w.step()
        kfb.write(_to3d(_lap(n=80, cw=True, laps=1.1)))
        w._loop_event.set()
        w.step()
        assert w._boundaries_extracted
        _, o, i = bb.read_latest()
        assert _mean_radius(o) > 1.0 > _mean_radius(i)
    finally:
        for b in bufs:
            b.unlink()


def test_map_reset_drops_recorded_data(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    w, bufs = _mapper()
    pb, pcb, kfb, bb = bufs
    try:
        pcb.write(np.random.randn(100, 3).astype(np.float32))
        pb.write(np.eye(4, dtype=np.float32))
        w.step()
        assert w._bev_trajectory and w._bev_wall_pts
        w._map_reset_event.set()
        w._loop_event.set()
        w.step()                                   # reset wins, stale data never used
        assert not w._map_reset_event.is_set()
        assert not w._bev_trajectory and not w._bev_wall_pts
        assert not w._boundaries_extracted
    finally:
        for b in bufs:
            b.unlink()


def test_lidar_voxel_filter_first_point_wins():
    from workers.lidar_projection_worker import LidarProjectionWorker

    w = LidarProjectionWorker()
    w._voxel_size = 0.1
    pts = np.array([[0.01, 0.01, 0], [0.02, 0.03, 0], [0.5, -0.5, 0], [-0.01, 0, 0]], np.float32)
    assert w._add_to_voxel_map(pts) == 3          # first two share a voxel
    np.testing.assert_allclose(sorted(map(tuple, w._voxel_pts)),
                               sorted(map(tuple, pts[[0, 2, 3]])))
    assert w._add_to_voxel_map(np.array([[0.05, 0.05, 0.0], [1.0, 1.0, 1.0]], np.float32)) == 1
    assert len(w._voxel_pts) == 4
