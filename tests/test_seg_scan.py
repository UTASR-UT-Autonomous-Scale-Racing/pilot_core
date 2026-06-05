"""control/seg_scan.py: the segmentation-edge scan that stands in for a LiDAR on the car,
the gap follower on a scan that only covers the camera's view, and its ControlWorker /
start.py wiring. Scenes are rendered analytically (flat floor, vertical walls)."""

from __future__ import annotations

import os
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from control.depth_ftg import Intrinsics
from control.lidar_gap import ANGLE_INCREMENT, ANGLE_MIN, LidarGapConfig, LidarGapFollower
from control.seg_scan import SegScanConfig, seg_edge_scan
from control.types import VehicleParams
from tests.conftest import REPO_ROOT

W, H = 640, 400
K = Intrinsics.from_orbslam_settings("orbslam_oakd_rgbd.yaml")
PITCH, CAM_H = np.radians(8.0), 0.15
ANG = ANGLE_MIN + np.arange(1080) * ANGLE_INCREMENT


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _scene(wall_fwd: float | None = None, max_depth: float = 20.0):
    """(mask, depth) for a level floor and optionally a wall across the road wall_fwd m ahead
    of the camera. mask = floor pixels; depth = optical z, 0 where nothing is within max_depth."""
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    yn = (v - K.cy) / K.fy                  # the wall is across the road: only the row matters
    cp, sp = np.cos(PITCH), np.sin(PITCH)
    d_fwd, d_up = cp - yn * sp, -sp - yn * cp
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(d_up < -1e-6, CAM_H / -d_up, np.inf)
        t_wall = np.where(d_fwd > 1e-6, wall_fwd / d_fwd, np.inf) if wall_fwd else np.full(d_up.shape, np.inf)
    floor = t_floor < t_wall
    depth = np.where(floor, t_floor, t_wall)
    depth = np.where(np.isfinite(depth) & (depth < max_depth), depth, 0.0).astype(np.float32)
    return floor.astype(np.uint8), depth


def _cfg(**kw) -> SegScanConfig:
    return SegScanConfig(camera_pitch=PITCH, camera_height=CAM_H, **kw)


def test_wall_ahead_is_placed_at_its_distance():
    mask, depth = _scene(wall_fwd=2.0)
    r = seg_edge_scan(mask, depth, K, _cfg())
    ahead = np.abs(ANG) < np.radians(30)
    fwd = r[ahead] * np.cos(ANG[ahead])
    assert np.all(np.isfinite(fwd))
    np.testing.assert_allclose(fwd, 2.0, atol=0.03)


def test_open_road_is_free_and_outside_the_view_unknown():
    mask, depth = _scene(wall_fwd=None)
    r = seg_edge_scan(mask, depth, K, _cfg())
    half_fov = np.arctan2(max(K.cx, W - K.cx), K.fx)
    assert np.all(np.isinf(r[np.abs(ANG) < half_fov - np.radians(2)]))
    assert np.all(np.isnan(r[np.abs(ANG) > half_fov + np.radians(2)]))


def test_no_depth_at_the_edge_is_unknown_without_the_floor_fallback():
    mask, depth = _scene(wall_fwd=None)
    r = seg_edge_scan(mask, depth, K, _cfg(floor_fallback=False))
    assert np.all(np.isnan(r))


def test_the_car_body_at_the_bottom_of_the_view_is_skipped():
    mask, depth = _scene(wall_fwd=2.0)
    mask[-80:, 200:440] = 0          # not drivable, like the car's own body in view
    depth[-80:, 200:440] = 0.0
    r = seg_edge_scan(mask, depth, K, _cfg())
    ahead = np.abs(ANG) < np.radians(10)
    np.testing.assert_allclose(r[ahead] * np.cos(ANG[ahead]), 2.0, atol=0.03)


def test_mask_at_another_resolution_is_resized():
    mask, depth = _scene(wall_fwd=2.0)
    small = mask[::2, ::2]
    a, b = seg_edge_scan(mask, depth, K, _cfg()), seg_edge_scan(small, depth, K, _cfg())
    ahead = np.abs(ANG) < np.radians(30)
    np.testing.assert_allclose(a[ahead], b[ahead], atol=0.05)


def test_origin_offset_moves_the_ranges():
    mask, depth = _scene(wall_fwd=2.0)
    r = seg_edge_scan(mask, depth, K, _cfg(origin_forward=-0.3))     # camera 0.3 m behind the origin
    i = int(np.argmin(np.abs(ANG)))
    assert r[i] == pytest.approx(1.7, abs=0.03)


def test_gap_follower_stays_inside_a_partial_scan():
    # camera-like scan: known only within +-35 deg, open ahead, a wall close on the right
    r = np.full(1080, np.nan)
    view = np.abs(ANG) < np.radians(35)
    r[view] = 3.0
    r[view & (ANG < np.radians(-25))] = 0.8
    g = LidarGapFollower(VehicleParams.load("vehicle_params_roboracer.yaml"), LidarGapConfig())
    t = g.target_angle(r)
    assert np.radians(-35) <= t <= np.radians(35)
    assert t > 0                    # away from the wall on the right
    assert np.isfinite(g.wall_push(r))


def test_gap_follower_with_nothing_free_aims_at_the_deepest_known_beam():
    r = np.full(1080, np.nan)
    view = np.abs(ANG) < np.radians(30)
    r[view] = 0.5
    r[view & (ANG > np.radians(20))] = 0.9
    t = LidarGapFollower(VehicleParams.load("vehicle_params_roboracer.yaml"), LidarGapConfig()).target_angle(r)
    assert np.radians(20) <= t <= np.radians(30)


# ---------------------------------------------------------------- ControlWorker

def _control_rig(segment: bool = True):
    from control import VehicleState
    from shared_buffer import ControlBuffer, FrameBuffer, SegmentationBuffer, StateBuffer, TrackBoundaryBuffer
    from workers import ControlWorker

    bufs = {
        "frame": FrameBuffer(slots=4, name=f"ss_f_{_uid()}", create=True),
        "seg": SegmentationBuffer(slots=4, name=f"ss_s_{_uid()}", create=True),
        "state": StateBuffer(slots=4, name=f"ss_st_{_uid()}", create=True),
        "boundary": TrackBoundaryBuffer(slots=2, name=f"ss_b_{_uid()}", create=True),
        "ctrl": ControlBuffer(slots=8, name=f"ss_c_{_uid()}", create=True),
    }
    w = ControlWorker(
        name="control", mode="pure_pursuit", output_buffer=bufs["ctrl"],
        state_buffer=bufs["state"], boundary_buffer=bufs["boundary"],
        mapping_done_event=MagicMock(is_set=MagicMock(return_value=False)),
        track_config={"mapping_drive": {"speed": 0.6, "min_speed": 0.4}},
        vehicle_params="vehicle_params_roboracer.yaml",
        frame_buffer=bufs["frame"], segment_buffer=bufs["seg"] if segment else None,
        seg_scan={"camera_settings": "orbslam_oakd_rgbd.yaml", "camera_pitch": float(PITCH),
                  "camera_height": CAM_H},
    )
    w._logger = MagicMock()
    bufs["state"].write(VehicleState(t=time.monotonic(), x=0.0, y=0.0, yaw=0.0, v=0.5, yaw_rate=0.0))
    return w, bufs


def test_control_worker_drives_the_mapping_lap_on_the_segmentation_scan():
    w, bufs = _control_rig()
    try:
        w.setup()
        mask, depth = _scene(wall_fwd=3.0)
        rgb = np.zeros((H, W, 3), np.uint8)
        ts = time.monotonic()
        bufs["frame"].write(rgb, depth, ts=ts)
        bufs["frame"].write(rgb, depth, ts=ts + 0.03)      # a newer frame arrived during inference
        bufs["seg"].write(mask, ts=ts)                      # VisionWorker stamps the frame's time
        w.step()
        assert bufs["ctrl"].latest_seq() == 1
        _, steer, throttle = bufs["ctrl"].read_latest()
        assert 0.0 <= steer <= 1.0 and throttle > 0.5      # drives forward
    finally:
        for b in bufs.values():
            b.unlink()


def test_a_mask_whose_frame_has_left_the_ring_is_skipped():
    w, bufs = _control_rig()
    try:
        w.setup()
        mask, depth = _scene(wall_fwd=3.0)
        bufs["frame"].write(np.zeros((H, W, 3), np.uint8), depth, ts=time.monotonic())
        bufs["seg"].write(mask, ts=time.monotonic() - 5.0)
        w.step()
        assert bufs["ctrl"].latest_seq() == 0
    finally:
        for b in bufs.values():
            b.unlink()


def test_mapping_drive_without_lidar_or_segmentation_is_an_error():
    w, bufs = _control_rig(segment=False)
    try:
        with pytest.raises(RuntimeError, match="segment_buffer"):
            w.setup()
    finally:
        for b in bufs.values():
            b.unlink()


# ---------------------------------------------------------------- start.py wiring

def _build(config_name, **overrides):
    import start

    config = start.load_config(os.path.join(REPO_ROOT, config_name))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return start.build_pipeline(config)


@pytest.fixture()
def _unique_buffer_names(monkeypatch):
    import start

    suffix = uuid.uuid4().hex[:6]
    for cls_name in ("FrameBuffer", "IMUBuffer", "SegmentationBuffer", "PoseBuffer", "ControlBuffer",
                     "PointCloudBuffer", "TrackBoundaryBuffer", "LidarBuffer", "StateBuffer"):
        cls = getattr(start, cls_name)

        def make(*args, _cls=cls, **kwargs):
            kwargs["name"] = f"{kwargs['name']}_{suffix}"
            return _cls(*args, **kwargs)

        monkeypatch.setattr(start, cls_name, make)


def test_car_mapping_drive_needs_vision(_unique_buffer_names):
    with pytest.raises(SystemExit, match="vision.enabled"):
        _build("config_cpu.yaml", control_mode="pure_pursuit", enable_slam=True,
               track_control={"mapping_drive": {"speed": 0.5}})


def test_car_mapping_drive_runs_on_the_segmentation_scan(_unique_buffer_names):
    p = _build("config_cpu.yaml", control_mode="pure_pursuit", enable_slam=True,
               track_control={"mapping_drive": {"speed": 0.5}}, vision={"enabled": True})
    try:
        ctrl = next(w for w in p.workers if w.name == "control")
        vision = next(w for w in p.workers if w.name == "vision")
        assert ctrl.config["lidar_buffer"] is None
        assert ctrl.config["segment_buffer"] is vision.config["output_buffer"] is not None
        assert ctrl.config["seg_scan"]["camera_height"] == 0.12      # depth_ftg's, one camera mount
    finally:
        for b in p.buffers:
            b.unlink()


def test_sim_uses_the_lidar_unless_told_otherwise(_unique_buffer_names):
    p = _build("config_autodrive.yaml")
    try:
        ctrl = next(w for w in p.workers if w.name == "control")
        assert ctrl.config["lidar_buffer"] is not None
    finally:
        for b in p.buffers:
            b.unlink()
    p = _build("config_autodrive.yaml", seg_scan={"source": "segmentation"}, vision={"enabled": True})
    try:
        ctrl = next(w for w in p.workers if w.name == "control")
        assert ctrl.config["lidar_buffer"] is None and ctrl.config["segment_buffer"] is not None
        assert ctrl.config["seg_scan"]["origin_forward"] == pytest.approx(-0.2883)
    finally:
        for b in p.buffers:
            b.unlink()
