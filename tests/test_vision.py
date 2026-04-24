"""
test_vision.py — drivable-track segmentation (vision/): candidate networks,
dataset/loss/metric, training + ONNX export, the training-free ground-plane
segmenter, legacy-output conversion and the VisionWorker modes.

The network tests need torch and torchvision; the ONNX tests also need onnx
and onnxruntime.  They are skipped when those are missing.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from control.depth_ftg import Intrinsics
from vision.data import bce_dice_loss, find_pairs, load_pair, track_iou
from vision.segmenters import GroundPlaneSegmenter, to_track_mask

K = Intrinsics(396.0, 395.8, 327.0, 189.0)      # OAK-D RGB at 640×400
H, W = 400, 640


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _floor_depth(cam_height=0.12, pitch=0.0, wall_at=None):
    """Depth (m) of a flat floor, optionally with a wall ``wall_at`` m ahead."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    ray = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u)], axis=-1)
    c, s = np.cos(pitch), np.sin(pitch)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])          # camera pitched down by `pitch`
    ray_w = ray @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(ray_w[..., 1] > 1e-6, cam_height / ray_w[..., 1], np.inf)
        t = t_floor
        if wall_at is not None:
            t_wall = np.where(ray_w[..., 2] > 1e-6, wall_at / ray_w[..., 2], np.inf)
            t = np.minimum(t_floor, t_wall)
    depth = np.where(np.isfinite(t), t * ray[..., 2], 0.0)
    return depth.astype(np.float32)


# --- ground plane ---

class TestGroundPlane:

    def test_flat_floor_is_drivable(self):
        mask = GroundPlaneSegmenter(K).get_mask(_floor_depth())
        below = mask[int(K.cy) + 20:]
        assert below.mean() > 0.9                       # floor in front of the car
        assert mask[: int(K.cy) - 10].sum() == 0        # nothing above the horizon

    def test_wall_blocks_free_space(self):
        depth = _floor_depth(wall_at=2.0)
        mask = GroundPlaneSegmenter(K).get_mask(depth)
        wall_rows = depth > 0
        near = (depth > 0) & (depth < 1.8)
        assert mask[near & (np.arange(H)[:, None] > K.cy + 5)].mean() > 0.9
        assert mask[(depth > 1.95) & wall_rows].mean() < 0.05   # wall and beyond are not track

    def test_handles_camera_pitch(self):
        seg = GroundPlaneSegmenter(K)
        depth = _floor_depth(pitch=np.deg2rad(10))
        mask = seg.get_mask(depth)
        floor = (depth > 0) & (depth < seg.cfg.max_range)       # all floor within range
        assert seg.plane is not None
        assert mask[floor].mean() > 0.9 and mask[~floor].mean() < 0.05

    def test_no_depth_gives_empty_mask(self):
        assert GroundPlaneSegmenter(K).get_mask(np.zeros((H, W), np.float32)).sum() == 0


# --- legacy outputs, dataset, metrics ---

@pytest.mark.parametrize("output, expected_sum", [
    (np.array([[True, False], [False, False]]), 1),
    (np.array([[0.0, 2.5], [1.0, 0.0]], np.float32), 2),          # depth with non-track zeroed
    ((np.array([[[0, 0, 0], [9, 0, 0]], [[0, 0, 0], [0, 0, 0]]], np.uint8), 0.6, 0.5), 1),  # Threshold tuple
])
def test_to_track_mask(output, expected_sum):
    mask = to_track_mask(output, (2, 2))
    assert mask.dtype == np.uint8 and mask.shape == (2, 2) and mask.sum() == expected_sum


def test_to_track_mask_resizes_and_passes_none():
    assert to_track_mask(np.ones((4, 4)), (8, 8)).shape == (8, 8)
    assert to_track_mask(None, (8, 8)) is None


def test_track_iou():
    a = np.zeros((4, 4), np.uint8)
    b = a.copy()
    assert track_iou(a, b) == 1.0
    a[:2] = 1
    b[1:3] = 1
    assert track_iou(a, b) == pytest.approx(4 / 12)


def _write_dataset(root, n=8, h=48, w=64, seed=0):
    """Tiny fixture dataset: the track is the bright lower-left region."""
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "images"))
    os.makedirs(os.path.join(root, "masks"))
    for i in range(n):
        mask = np.zeros((h, w), np.uint8)
        cut = rng.integers(w // 3, 2 * w // 3)
        mask[h // 2:, :cut] = 255
        img = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)
        img[mask > 0] += 150
        cv2.imwrite(os.path.join(root, "images", f"f{i}.png"), img)
        cv2.imwrite(os.path.join(root, "masks", f"f{i}.png"), mask)
    cv2.imwrite(os.path.join(root, "images", "unlabelled.png"), img)      # no mask → skipped
    return root


def test_dataset_layout(tmp_path):
    root = _write_dataset(str(tmp_path / "ds"))
    pairs = find_pairs(root)
    assert len(pairs) == 8
    rgb, mask = load_pair(*pairs[0], (32, 40))
    assert rgb.shape == (32, 40, 3) and set(np.unique(mask)) <= {0, 1}


def test_find_pairs_empty(tmp_path):
    os.makedirs(tmp_path / "images")
    os.makedirs(tmp_path / "masks")
    with pytest.raises(FileNotFoundError):
        find_pairs(str(tmp_path))


# --- networks ---

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from vision.models import MODELS, build_model  # noqa: E402


@pytest.mark.parametrize("name", MODELS)
def test_model_output_shape(name):
    net = build_model(name).eval()
    with torch.no_grad():
        y = net(torch.randn(2, 3, 96, 160))
    assert y.shape == (2, 1, 96, 160)


def test_unknown_model():
    with pytest.raises(ValueError):
        build_model("resnet9000")


def test_loss_is_finite_and_prefers_correct_prediction():
    target = torch.zeros(1, 1, 8, 8)
    target[..., 4:, :] = 1
    good = (target * 2 - 1) * 8
    assert float(bce_dice_loss(good, target)) < float(bce_dice_loss(-good, target))


def test_training_learns_fixture(tmp_path):
    from vision.training import TrackSegNet, evaluate, train

    pairs = find_pairs(_write_dataset(str(tmp_path / "ds"), n=12))
    torch.manual_seed(0)                  # weight init must not depend on earlier tests
    model = TrackSegNet("fast_scnn")
    before = evaluate(model, pairs[:4], (48, 64))
    model = train(model, pairs[4:], pairs[:4], (48, 64), epochs=25, batch_size=4, lr=3e-3, log=lambda *_: None)
    after = evaluate(model, pairs[:4], (48, 64))
    assert after > max(before, 0.8)


@pytest.mark.parametrize("name", ["fast_scnn", "lraspp"])
def test_onnx_export_matches_torch(name, tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from vision.segmenters import OnnxTrackSegmenter
    from vision.training import TrackSegNet, export_onnx

    torch.manual_seed(0)
    model = TrackSegNet(name).eval()
    path = export_onnx(model, str(tmp_path / f"{name}.onnx"), (96, 160))
    x = torch.rand(1, 3, 96, 160)
    with torch.no_grad():
        ref = torch.sigmoid(model(x)).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"rgb": x.numpy()})[0]
    np.testing.assert_allclose(out, ref, atol=1e-4)

    seg = OnnxTrackSegmenter(path)
    mask = seg.get_mask(None, np.zeros((H, W, 3), np.uint8))
    assert mask.shape == (H, W) and mask.dtype == np.uint8


# --- VisionWorker ---

def _run_worker(**kwargs):
    from shared_buffer import FrameBuffer, SegmentationBuffer
    from workers import VisionWorker

    fb = FrameBuffer(slots=2, name=f"vs_f_{_uid()}", create=True)
    sb = SegmentationBuffer(slots=2, name=f"vs_s_{_uid()}", create=True)
    try:
        w = VisionWorker(name="vision", input_buffer=fb, output_buffer=sb, **kwargs)
        w._logger = MagicMock()
        w.setup()
        fb.write(np.zeros((H, W, 3), np.uint8), _floor_depth(wall_at=2.0))
        w.step()
        result = sb.read_latest()
        assert result is not None
        return result[1]
    finally:
        fb.unlink()
        sb.unlink()


def test_worker_ground_plane_mode():
    mask = _run_worker(mode="ground_plane", camera_settings="orbslam_oakd_rgbd.yaml")
    assert mask.shape == (H, W) and set(np.unique(mask)) <= {0, 1} and mask.sum() > 0


def test_worker_pilot_vision_mode_outputs_mask():
    mask = _run_worker(mode="pilot_vision", model_config="pilot_vision/models/default/config.yaml")
    assert set(np.unique(mask)) <= {0, 1}


def test_worker_onnx_mode(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from vision.training import TrackSegNet, export_onnx

    path = export_onnx(TrackSegNet("fast_scnn"), str(tmp_path / "seg.onnx"), (96, 160))
    mask = _run_worker(mode="onnx", onnx_model=path)
    assert mask.shape == (H, W) and set(np.unique(mask)) <= {0, 1}


def test_worker_rejects_unknown_mode():
    from workers import VisionWorker

    w = VisionWorker(name="vision", input_buffer=None, output_buffer=None, mode="bogus")
    w._logger = MagicMock()
    with pytest.raises(ValueError):
        w.setup()
