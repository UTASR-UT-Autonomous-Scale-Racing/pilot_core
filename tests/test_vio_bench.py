"""scripts/vio_bench: car recording -> rgbdi sequence, AprilTag sightings and the marker metrics."""

from __future__ import annotations

import os
import sys
import uuid
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "vio_bench"))

import car_to_rgbdi  # noqa: E402
import markers  # noqa: E402
import rgbdi  # noqa: E402
from shared_buffer import FrameBuffer, RawIMUBuffer  # noqa: E402
from workers.buffer_recorder_worker import BufferRecorderWorker  # noqa: E402

K = np.array([[400.0, 0, 320.0], [0, 400.0, 200.0], [0, 0, 1]])


def _render_tag(tag_id: int, size: float, R: np.ndarray, t: np.ndarray, shape=(400, 640)) -> np.ndarray:
    """Grey image of one tag36h11 at pose (R, t) in the camera frame (tag x right, y up)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    s = 400
    tag = cv2.aruco.generateImageMarker(d, tag_id, s, borderBits=1)
    corners3d = markers.tag_object_points(size) @ R.T + t
    img_pts = (corners3d @ K.T)[:, :2] / (corners3d @ K.T)[:, 2:]
    e = (-0.5, s - 0.5)                              # the marker image's outer pixel edges
    H = cv2.getPerspectiveTransform(np.float32([[e[0], e[0]], [e[1], e[0]], [e[1], e[1]], [e[0], e[1]]]),
                                    img_pts.astype(np.float32))
    canvas = np.full(shape, 255, np.uint8)
    warped = cv2.warpPerspective(tag, H, (shape[1], shape[0]), borderValue=255)
    mask = cv2.warpPerspective(np.full_like(tag, 255), H, (shape[1], shape[0])) > 0
    canvas[mask] = warped[mask]
    return canvas


def test_tag_is_located_in_the_camera_frame():
    R = np.diag([1.0, -1.0, -1.0])                    # facing the camera
    R = cv2.Rodrigues(np.array([0.0, 0.4, 0.0]))[0] @ R   # and turned 23 deg
    t = np.array([0.3, 0.1, 1.5])
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), params)
    found = markers.detect_frame(_render_tag(7, 0.16, R, t), det, K, 0.16)
    assert [f[0] for f in found] == [7]
    _, p, side, reproj = found[0]
    assert np.linalg.norm(p - t) < 0.02
    assert side > 30 and reproj < 1.0


def _sightings(tags: dict, t_ns: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    rows = []
    for t, T in zip(t_ns, T_wc):
        Tcw = np.linalg.inv(T)
        for i, p in tags.items():
            rows.append([t, i, *(Tcw @ np.append(p, 1.0))[:3], 50.0, 0.2])
    return np.array(rows)


def _two_laps():
    """Two passes 10 s apart past the same tags along a straight; T_world_cam per frame."""
    t = np.concatenate([np.arange(0, 1, 0.1), np.arange(10, 11, 0.1)])
    T = np.tile(np.eye(4), (len(t), 1, 1))
    T[:, 0, 3] = np.concatenate([np.arange(10) * 0.1, np.arange(10) * 0.1])
    tags = {0: np.array([0.5, 0.2, 2.0]), 1: np.array([-0.4, 0.0, 2.5]), 2: np.array([0.3, -0.3, 3.0])}
    return (t * 1e9).astype(np.int64), T, tags


def test_perfect_trajectory_scores_zero_in_any_world_frame():
    t, T, tags = _two_laps()
    det = _sightings(tags, t, T)
    W = np.eye(4)
    W[:3, :3] = cv2.Rodrigues(np.array([0.1, 1.2, -0.3]))[0]
    W[:3, 3] = [5, -2, 1]                             # the result's own world frame
    r = markers.score_result(det, t, W @ T, tags, closed=False, pass_gap=2.0)
    assert r["sightings_tracked"] == len(det)
    assert r["repeatability_m"]["rms"] < 1e-9
    assert r["absolute_m"]["rmse"] < 1e-6


def test_drift_between_laps_shows_in_repeatability_and_growth():
    t, T, tags = _two_laps()
    det = _sightings(tags, t, T)
    est = T.copy()
    est[10:, 0, 3] += 0.3                              # lap 2 believes it is 0.3 m further along
    r = markers.score_result(det, t, est, tags, closed=False, pass_gap=2.0)
    assert r["repeatability_m"]["rms"] == pytest.approx(0.15, abs=1e-6)
    assert r["drift_from_first_pass_m"][1] == pytest.approx(0.3, abs=1e-6)
    assert r["absolute_m"]["rmse"] > 0.1
    # frames the result did not track are not scored
    r = markers.score_result(det, t[:10], est[:10], None, closed=True, pass_gap=2.0)
    assert r["sightings_tracked"] == 30 and "repeatability_m" not in r
    assert r["closure_m"] == pytest.approx(0.9)


SETTINGS = """%YAML:1.0
File.version: "1.0"
Camera1.fx: 50.0
Camera1.fy: 50.0
Camera1.cx: 31.5
Camera1.cy: 19.5
Camera.width: 64
Camera.height: 40
Camera.fps: 30
IMU.T_b_c1: !!opencv-matrix
   rows: 4
   cols: 4
   dt: f
   data: [1, 0, 0, 0.05, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
IMU.NoiseGyro: 0.01
IMU.NoiseAcc: 0.1
IMU.GyroWalk: 1e-06
IMU.AccWalk: 0.0001
IMU.Frequency: 200.0
"""


def test_car_recording_converts_to_a_sequence(tmp_path, monkeypatch):
    u = uuid.uuid4().hex[:8]
    fb = FrameBuffer(slots=8, name=f"vb_f_{u}", max_payload=16 + 64 * 40 * 7)
    ib = RawIMUBuffer(slots=64, name=f"vb_i_{u}")
    rec = tmp_path / "rec"
    try:
        w = BufferRecorderWorker(name="recorder", output_dir=str(rec), buffers={"frame": fb, "raw_imu": ib})
        w._logger = MagicMock()
        w.setup()
        t0 = 500.0
        for k in range(40):                            # 200 Hz IMU, a frame every 6th sample
            t = t0 + k * 0.005
            ib.write(np.array([0, 0, 9.81, 0, 0, 0.1 * k], np.float32), ts=t)
            if k % 6 == 0:
                rgb = np.full((40, 64, 3), k, np.uint8)
                fb.write(rgb, np.full((40, 64), 1.25, np.float32), ts=t)
            w.step()
        w.teardown()
    finally:
        fb.unlink()
        ib.unlink()
    settings = tmp_path / "s.yaml"
    settings.write_text(SETTINGS)
    out = tmp_path / "seq"
    monkeypatch.setattr(sys, "argv", ["car_to_rgbdi.py", str(rec), "--settings", str(settings), "--out", str(out)])
    assert car_to_rgbdi.main() == 0

    seq = rgbdi.load(str(out))
    assert len(seq.t_ns) == 7
    assert seq.t_ns[0] == int(round(t0 * 1e9))
    assert seq.calib["fx"] == 50.0 and seq.calib["imu"]["rate_hz"] == 200.0
    np.testing.assert_allclose(seq.T_imu_cam[0, 3], 0.05)
    assert (seq.imu[:, 0] >= seq.t_ns[0] - 5e7).all()
    np.testing.assert_allclose(seq.imu[:3, 3], 9.81, atol=1e-5)
    depth = cv2.imread(seq.depth[0], cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and int(depth[0, 0]) == 1250
    rgb = cv2.imread(seq.rgb[2], cv2.IMREAD_COLOR)
    assert int(rgb[0, 0, 0]) == 12


def test_markers_cli_on_a_rendered_sequence(tmp_path, monkeypatch):
    """Two passes along a straight past three tags; a perfect and a drifting result."""
    import json

    import yaml

    tags = {0: np.array([-0.5, 0.0, 1.4]), 1: np.array([0.0, -0.2, 1.6]), 2: np.array([0.5, 0.1, 1.3])}
    face = np.diag([1.0, -1.0, -1.0])                  # tags face the camera (-z)
    t_ns, T = [], []
    for k in range(12):                                # passes at 0-0.55 s and 10-10.55 s
        t = (k % 6) * 0.1 + (10.0 if k >= 6 else 0.0)
        Twc = np.eye(4)
        Twc[0, 3] = 0.02 * (k % 6)
        t_ns.append(int(t * 1e9))
        T.append(Twc)
    (tmp_path / "seq" / "rgb").mkdir(parents=True)
    (tmp_path / "seq" / "depth").mkdir()
    with open(tmp_path / "seq" / "frames.csv", "w") as f:
        f.write("t_ns,rgb,depth\n")
        for t, Twc in zip(t_ns, T):
            img = np.full((400, 640), 255, np.uint8)
            Tcw = np.linalg.inv(Twc)
            for i, p in tags.items():
                one = _render_tag(i, 0.16, Tcw[:3, :3] @ face, (Tcw @ np.append(p, 1))[:3])
                img = np.minimum(img, one)
            cv2.imwrite(str(tmp_path / "seq" / "rgb" / f"{t}.png"), img)
            cv2.imwrite(str(tmp_path / "seq" / "depth" / f"{t}.png"), np.zeros((400, 640), np.uint16))
            f.write(f"{t},rgb/{t}.png,depth/{t}.png\n")
    (tmp_path / "seq" / "imu.csv").write_text("t_ns,ax,ay,az,gx,gy,gz\n0,0,0,9.81,0,0,0\n")
    calib = {"width": 640, "height": 400, "fx": 400.0, "fy": 400.0, "cx": 320.0, "cy": 200.0,
             "T_imu_cam": np.eye(4).tolist(), "imu": {"rate_hz": 200.0}}
    (tmp_path / "seq" / "calib.yaml").write_text(yaml.safe_dump(calib))
    est = [Ti.copy() for Ti in T]
    for Ti in est[6:]:
        Ti[0, 3] += 0.2                                 # the second pass is 0.2 m off
    for name, poses in (("perfect", T), ("drift", est)):
        d = tmp_path / "res" / name
        d.mkdir(parents=True)
        rgbdi.write_poses(str(d / "poses.csv"), np.array(t_ns), np.array(poses))
        (d / "meta.json").write_text(json.dumps({"mode": "test"}))
    survey = {"family": "36h11", "size_m": 0.16, "tags": {i: p.tolist() for i, p in tags.items()}}
    (tmp_path / "markers.yaml").write_text(yaml.safe_dump(survey))
    monkeypatch.setattr(sys, "argv", ["markers.py", str(tmp_path / "seq"), str(tmp_path / "res" / "perfect"),
                                      str(tmp_path / "res" / "drift"), "--survey", str(tmp_path / "markers.yaml"),
                                      "--out", str(tmp_path / "out")])
    assert markers.main() == 0
    perfect, drift = json.loads((tmp_path / "out" / "markers.json").read_text())
    assert perfect["tags_seen"] == 3 and perfect["sightings_tracked"] == 36
    assert perfect["repeatability_m"]["rms"] < 0.01
    assert perfect["absolute_m"]["rmse"] < 0.02
    assert drift["drift_from_first_pass_m"]["1"] == pytest.approx(0.2, abs=0.02)


def test_rendered_depth_is_exact():
    """render_track.Scene: floor and wall depths along known rays, and a textured image."""
    import render_track

    inner = np.array([[30.0, -1.0], [31.0, -1.0], [31.0, 1.0], [30.0, 1.0]])     # out of the way
    outer = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])       # wall 5 m ahead
    scene = render_track.Scene(inner, outer)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([[0, -1, 0], [0, 0, -1], [1, 0, 0]])   # optical z along world x, level
    T[:3, 3] = [0.0, 0.0, 0.2]                                           # 0.2 m above the floor
    K = (400.0, 400.0, 319.5, 199.5)
    depth = scene.cast(T, K, 640, 400, colour=False)
    assert depth[199 + 41, 319] == pytest.approx(0.2 * 400 / (41 - 0.5), rel=1e-9)   # floor
    assert depth[199 - 4, 319] == pytest.approx(5.0, rel=1e-9)                       # wall
    assert depth[0, 319] == 0.0                                                        # above the wall
    img = scene.cast(T, K, 640, 400, colour=True)
    assert img.shape == (400, 640) and img[250:].std() > 20                           # textured floor
