"""Shared fixtures: generated sensor data and shm buffers that get unlinked on teardown."""

from __future__ import annotations

import multiprocessing as mp
import os
import uuid

import numpy as np
import pytest

# Absolute path of the repository root, so tests don't depend on the cwd:
#   from tests.conftest import REPO_ROOT   or   def test_x(repo_root): ...
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def repo_root() -> str:
    """Absolute path of the repository root (same as ``REPO_ROOT``)."""
    return REPO_ROOT


# unique shm names so parallel runs don't collide

def _uid() -> str:
    return uuid.uuid4().hex[:8]


# generated data

@pytest.fixture()
def rgb_frame() -> np.ndarray:
    """640x400 BGR uint8, same shape as the OAK-D output."""
    return np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)


@pytest.fixture()
def depth_frame() -> np.ndarray:
    """Depth in metres, float32, uniform in [0.2, 5.0]"""
    return np.random.uniform(0.2, 5.0, (400, 640)).astype(np.float32)


@pytest.fixture()
def imu_vector() -> np.ndarray:
    """imu_buf layout: [qw, qx, qy, qz, ax, ay, az, s7, s8]"""
    return np.random.randn(9).astype(np.float32)


@pytest.fixture()
def segmentation_mask() -> np.ndarray:
    """Fake ResNet18 output. Classes: 0=background, 1=road, 2=wall."""
    rng = np.random.default_rng(42)
    mask = np.zeros((400, 640), dtype=np.int32)
    # road in the middle, wall top and bottom
    mask[100:300, :] = 1
    mask[:100, :] = 2
    mask[300:, :] = 2
    noise_idx = rng.integers(0, 400 * 640, size=5000)
    mask.flat[noise_idx] = rng.integers(0, 3, size=5000)
    return mask


@pytest.fixture()
def pose_4x4() -> np.ndarray:
    """Random valid SE(3) pose, float32."""
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=42).as_matrix().astype(np.float32)
    t = np.random.randn(3).astype(np.float32) * 2.0
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@pytest.fixture()
def point_cloud() -> np.ndarray:
    return np.random.randn(500, 3).astype(np.float32) * 5.0


@pytest.fixture()
def track_boundaries() -> tuple[np.ndarray, np.ndarray]:
    """Oval outer/inner boundaries in BEV (X, Z)."""
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    outer = np.column_stack([
        8.0 * np.cos(theta),
        5.0 * np.sin(theta),
    ]).astype(np.float32)
    inner = np.column_stack([
        5.0 * np.cos(theta),
        3.0 * np.sin(theta),
    ]).astype(np.float32)
    return outer, inner


# Buffers. Each one gets unlinked on teardown

@pytest.fixture()
def frame_buffer():
    from shared_buffer import FrameBuffer
    buf = FrameBuffer(slots=4, name=f"test_frame_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


@pytest.fixture()
def imu_buffer():
    from shared_buffer import IMUBuffer
    buf = IMUBuffer(slots=8, name=f"test_imu_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


@pytest.fixture()
def segmentation_buffer():
    from shared_buffer import SegmentationBuffer
    buf = SegmentationBuffer(slots=4, name=f"test_seg_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


@pytest.fixture()
def pose_buffer():
    from shared_buffer import PoseBuffer
    buf = PoseBuffer(slots=4, name=f"test_pose_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


@pytest.fixture()
def control_buffer():
    from shared_buffer import ControlBuffer
    buf = ControlBuffer(slots=4, name=f"test_ctrl_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


@pytest.fixture()
def pointcloud_buffer():
    from shared_buffer import PointCloudBuffer
    buf = PointCloudBuffer(slots=4, name=f"test_pc_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


@pytest.fixture()
def boundary_buffer():
    from shared_buffer import TrackBoundaryBuffer
    buf = TrackBoundaryBuffer(slots=2, name=f"test_bnd_{_uid()}", create=True)
    yield buf
    try:
        buf.unlink()
    except Exception:
        pass


# multiprocessing

@pytest.fixture()
def spawn_ctx():
    return mp.get_context("spawn")


@pytest.fixture()
def mp_event():
    ctx = mp.get_context("spawn")
    return ctx.Event()
