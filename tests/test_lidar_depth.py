"""LiDAR -> depth adapter for ORB-SLAM3 on the AutoDRIVE sim camera (workers/lidar_depth.py):
geometry against hand-computed depth, and a car-body mask that hides the body without
leaving corners on its outline."""

from __future__ import annotations

import math

import numpy as np
import pytest

from workers.lidar_depth import ANGLE_INCREMENT, ANGLE_MIN, CameraModel, LidarDepth, mask_rgb


def _wall_ahead(cam: CameraModel, dist_from_camera: float) -> np.ndarray:
    """1080 ranges of a straight wall perpendicular to x, dist_from_camera ahead of the camera."""
    a = ANGLE_MIN + np.arange(1080) * ANGLE_INCREMENT
    x_wall = cam.cam_x + dist_from_camera - cam.lidar_x            # in the LiDAR frame
    with np.errstate(divide="ignore"):
        r = np.where(np.cos(a) > 1e-3, x_wall / np.cos(a), np.inf)
    return np.where(r < 10, r, np.inf)


def test_wall_and_floor_depth_match_the_geometry():
    cam = CameraModel(width=192, height=108, vfov_deg=40.0, pitch_deg=0.0, cam_height=0.2, wall_height=0.3)
    fn = LidarDepth(cam)
    depth = fn(_wall_ahead(cam, 3.0))
    fx, fy, cx, cy = cam.intrinsics()
    u, v = int(round(cx)), int(round(cy))
    # level camera: the centre ray meets the wall at its own height -> optical depth 3 m
    assert depth[v, u] == pytest.approx(3.0, abs=0.02)
    # a ray 10 deg down hits the floor at 0.2 / tan(10 deg) = 1.13 m, before the wall
    v_floor = int(round(cy + fy * math.tan(math.radians(10))))
    assert depth[v_floor, u] == pytest.approx(0.2 / math.tan(math.radians(10)), rel=0.03)
    # above the wall top (0.3 m at 3 m: 1.9 deg up) there is no depth
    v_sky = int(round(cy - fy * math.tan(math.radians(4))))
    assert depth[v_sky, u] == 0.0
    # a wall straight across the view is at the same optical depth in every column
    assert np.allclose(depth[v, 10:-10], 3.0, atol=0.03)


def test_pitch_moves_the_horizon_down_and_no_scan_means_no_wall():
    level = LidarDepth(CameraModel(pitch_deg=0.0))
    pitched = LidarDepth(CameraModel(pitch_deg=10.0))
    empty = np.full(1080, np.inf)
    d0, d1 = level(empty), pitched(empty)
    # only floor; looking down, the floor starts higher up in the image
    first_floor = lambda d: int(np.argmax((d > 0).any(axis=1)))     # noqa: E731
    assert first_floor(d1) < first_floor(d0)
    assert (d0[: first_floor(d0)] == 0).all()


def test_mask_hides_the_body_without_corners():
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    h, w = 108, 192
    scene = (rng.integers(40, 60, (h, w, 3))).astype(np.uint8)     # textured scene
    mask = np.zeros((h, w), bool)
    mask[60:, 40:150] = True
    img = scene.copy()
    img[mask] = (200, 200, 200)                                     # the car: a bright block
    out = mask_rgb(img, mask, clahe=False)
    orb = cv2.ORB_create(nfeatures=1000, fastThreshold=7, edgeThreshold=5)
    near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    near &= ~cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    def on_outline(im):
        return sum(near[int(k.pt[1]), int(k.pt[0])] for k in orb.detect(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), None))

    with_block, filled, no_car = on_outline(img), on_outline(out), on_outline(scene)
    assert with_block > 2 * no_car + 10          # a plain block puts corners on its outline
    # the fill removes most of them; a few stay where its smooth sides meet the textured scene
    assert filled - no_car <= (with_block - no_car) / 4


def test_autodrive_worker_writes_rgbd_from_lidar():
    """sim_depth on: a sim message becomes an upscaled RGB-D frame for SLAMWorker."""
    import logging
    import uuid

    from shared_buffer import FrameBuffer
    from tests.fake_autodrive import FakeAutoDRIVE
    from workers import AutoDRIVEWorker
    from workers.autodrive_worker import autodrive_to_internal

    fb = FrameBuffer(slots=2, name=f"rgbd_{uuid.uuid4().hex[:6]}", max_payload=16 + 384 * 216 * 7)
    w = AutoDRIVEWorker(frame_buffer=fb, sim_depth={"camera": "sim_camera/camera.yaml", "scale": 2})
    w._logger = logging.getLogger("test.rgbd")
    try:
        from workers.lidar_depth import CameraModel, LidarDepth
        w._frame_buf = fb
        w._depth_fn = LidarDepth(CameraModel.from_yaml("sim_camera/camera.yaml", scale=2))
        w._depth_clahe = True
        w._write_rgbd(autodrive_to_internal(FakeAutoDRIVE(port=1).telemetry()))
        _, rgb, depth = fb.read_latest()
        assert rgb.shape == (216, 384, 3) and depth.shape == (216, 384)
        assert (depth > 0).mean() > 0.2                       # floor / walls below the horizon
        assert not depth[w._depth_fn.mask].any()              # nothing on the car body
        assert depth.max() <= 10.0
    finally:
        fb.unlink()
