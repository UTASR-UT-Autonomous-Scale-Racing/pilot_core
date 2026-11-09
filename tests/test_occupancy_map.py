"""control/occupancy_map.py and MappingWorker's occupancy wall source: a ring track (walls at
r = 2.0 and 3.0 m) scanned 45 deg either side of ahead, lap driven at r = 2.5 m (from there
the inner wall is first seen 37 deg off the heading).
Walls land within about 0.1 m, as on the compete track (docs/results.md)."""

from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from control.lidar_gap import ANGLE_INCREMENT, ANGLE_MIN
from control.occupancy_map import OccupancyConfig, OccupancyTrackMapper
from control.track import TrackModel

ANG = ANGLE_MIN + np.arange(1080) * ANGLE_INCREMENT
R_IN, R_OUT, R_PATH = 2.0, 3.0, 2.5
VIEW = np.radians(45)


def _ray_circle(o: np.ndarray, d: np.ndarray, r: float) -> np.ndarray:
    """Nearest positive hit distance of rays o + t d (d unit, (N, 2)) with the circle |p| = r."""
    b = d @ o
    c = o @ o - r * r
    disc = b * b - c
    with np.errstate(invalid="ignore"):
        root = np.sqrt(disc)
        t1, t2 = -b - root, -b + root
    t = np.where(t1 > 1e-6, t1, np.where(t2 > 1e-6, t2, np.inf))
    return np.where(disc >= 0, t, np.inf)


def _scan(x: float, y: float, heading: float) -> np.ndarray:
    o = np.array([x, y])
    a = heading + ANG
    d = np.column_stack([np.cos(a), np.sin(a)])
    r = np.minimum(_ray_circle(o, d, R_IN), _ray_circle(o, d, R_OUT))
    r[np.abs(ANG) > VIEW] = np.nan
    return r.astype(np.float32)


def _lap(n: int = 240, laps: float = 1.2):
    th = np.linspace(0.0, 2 * np.pi * laps, int(n * laps), endpoint=False)
    xs, ys = R_PATH * np.cos(th), R_PATH * np.sin(th)
    heading = th + np.pi / 2          # counter-clockwise
    return xs, ys, heading


def _radii(line: np.ndarray) -> np.ndarray:
    return np.hypot(line[:, 0], line[:, 1])


def test_walls_of_a_ring_track():
    m = OccupancyTrackMapper(OccupancyConfig(reach=0.7))
    xs, ys, hs = _lap()
    for i, (x, y, h) in enumerate(zip(xs, ys, hs)):
        m.add_scan(0.05 * i, _scan(x, y, h), ANG, (x, y), h)
    outer, inner = m.extract(np.column_stack([xs, ys]))
    assert np.median(np.abs(_radii(outer) - R_OUT)) < 0.12
    assert np.median(np.abs(_radii(inner) - R_IN)) < 0.12
    track = TrackModel(outer, inner)
    assert track.length == pytest.approx(2 * np.pi * R_PATH, rel=0.05)


def test_scans_around_a_disturbance_and_standing_still_are_left_out():
    m = OccupancyTrackMapper()
    assert m.add_scan(0.0, _scan(R_PATH, 0.0, np.pi / 2), ANG, (R_PATH, 0.0), np.pi / 2)
    assert not m.add_scan(0.1, _scan(R_PATH, 0.0, np.pi / 2), ANG, (R_PATH, 0.01), np.pi / 2)  # 1 cm
    assert m.add_scan(0.2, _scan(R_PATH, 0.1, np.pi / 2), ANG, (R_PATH, 0.1), np.pi / 2)
    m.mark_disturbance(1.0)
    assert len(m._kept()) == 0                                # both within 1.5 s before it


def test_an_open_path_has_no_track():
    m = OccupancyTrackMapper()
    xs, ys, hs = _lap(laps=0.3)
    for i, (x, y, h) in enumerate(zip(xs, ys, hs)):
        m.add_scan(0.05 * i, _scan(x, y, h), ANG, (x, y), h)
    with pytest.raises(ValueError):
        m.extract(np.column_stack([xs, ys]))


def test_mapping_worker_maps_the_ring_from_scans():
    from shared_buffer import LidarBuffer, PoseBuffer
    from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer
    from workers import MappingWorker

    u = uuid.uuid4().hex[:8]
    pb = PoseBuffer(slots=512, name=f"om_p_{u}", create=True)
    sb = LidarBuffer(slots=512, name=f"om_s_{u}", create=True)
    bb = TrackBoundaryBuffer(slots=2, name=f"om_b_{u}", create=True)
    loop, done = mp.Event(), mp.Event()
    w = MappingWorker(name="mapping", pose_buffer=pb, boundary_buffer=bb, loop_event=loop,
                      mapping_done_event=done, wall_source="occupancy", scan_buffer=sb,
                      scan_offset=[0.0, 0.0, 0.0], bev_axes=[0, 1], forward_axis=0,
                      occupancy={"reach": 0.7})
    w._logger = MagicMock()
    try:
        w.setup()
        xs, ys, hs = _lap()
        t0 = time.monotonic()
        for i, (x, y, h) in enumerate(zip(xs, ys, hs)):
            T = np.eye(4, dtype=np.float32)
            c, s = np.cos(h), np.sin(h)
            T[:2, :2] = [[c, -s], [s, c]]
            T[0, 3], T[1, 3] = x, y
            ts = t0 + 0.05 * i
            pb.write(T, ts=ts)
            sb.write(_scan(x, y, h), ts=ts)
            w.step()
        loop.set()
        w.step()
        assert done.is_set()
        _, outer, inner = bb.read_latest()
        assert np.median(np.abs(_radii(outer) - R_OUT)) < 0.12
        assert np.median(np.abs(_radii(inner) - R_IN)) < 0.12
    finally:
        for b in (pb, sb, bb):
            b.unlink()


def test_seg_scan_worker_memory_in_the_slam_frame():
    """On the car the pose is SLAM's camera frame (X right, Y down, Z forward): a remembered edge
    lands where it was seen, and the raw output carries no remembered edges."""
    from shared_buffer import FrameBuffer, LidarBuffer, PoseBuffer
    from shared_buffer.segment_buffer import SegmentationBuffer
    from workers.seg_scan_worker import SegScanWorker

    h, w_ = 216, 384
    u = uuid.uuid4().hex[:8]
    sb = SegmentationBuffer(slots=4, name=f"om_g_{u}", create=True, max_payload=h * w_ + 64)
    fb = FrameBuffer(slots=4, name=f"om_f_{u}", max_payload=16 + h * w_ * 7)
    out = LidarBuffer(slots=4, name=f"om_o_{u}")
    raw = LidarBuffer(slots=4, name=f"om_r_{u}")
    pb = PoseBuffer(slots=8, name=f"om_q_{u}", create=True)
    worker = SegScanWorker(name="seg_scan", segment_buffer=sb, frame_buffer=fb, output_buffer=out,
                           raw_output_buffer=raw, pose_buffer=pb, memory_s=2.0, bev_axes=[0, 2], forward_axis=2,
                           seg_scan={"camera_settings": "sim_camera/orbslam_settings.yaml",
                                     "camera_pitch": 0.1833, "camera_height": 0.17})
    worker._logger = MagicMock()
    worker.setup()
    try:
        mask = np.zeros((h, w_), np.uint8)
        mask[150:, :] = 1                                   # an edge across the view
        blank = np.zeros((h, w_), np.float32)
        # camera facing +Z, then turned 90 deg left (towards -X in the camera's right-handed frame)
        for k, yaw in enumerate((0.0, np.pi / 2)):
            T = np.eye(4, dtype=np.float32)
            c, s = np.cos(yaw), np.sin(yaw)
            # rotation about the camera's down axis (Y): forward Z turns towards -X for a left turn
            T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, -s, s, c
            ts = 1.0 + k
            pb.write(T, ts=ts)
            fb.write(np.zeros((h, w_, 3), np.uint8), blank, ts=ts)
            sb.write(mask, ts=ts)
            worker.step()
        _, with_memory = out.read_latest()
        _, without = raw.read_latest()
        right = int(round((-np.pi / 2 - ANGLE_MIN) / ANGLE_INCREMENT))
        # the first edge is now to the right of the car, at its distance
        near = with_memory[right - 15:right + 16]
        assert np.isfinite(near).any()
        assert np.nanmin(near) == pytest.approx(float(np.nanmin(without[np.abs(ANG) < 0.05])), abs=0.25)
        assert not np.isfinite(without[right - 15:right + 16]).any()
    finally:
        for b in (sb, fb, out, raw, pb):
            b.unlink()
