"""
conftest.py — Shared fixtures for pilot_core test suite.

Provides:
    - Unique buffer names (avoid shm collisions between parallel tests)
    - Synthetic camera frames (640×400 RGB + Depth)
    - Synthetic IMU data (16-float vector)
    - Synthetic segmentation masks (ResNet18-style)
    - Synthetic poses (4×4 float32)
    - Synthetic point clouds (N×3 float32)
    - Synthetic track boundaries (outer/inner lines)
    - Multiprocessing spawn context and Event factories
"""

from __future__ import annotations

import multiprocessing as mp
import os
import uuid
from typing import Generator

import numpy as np
import pytest

# 
# Unique naming helpers — prevent shm collisions across parallel workers / reruns
# 

def _uid() -> str:
    """Short unique suffix for shared-memory names."""
    return uuid.uuid4().hex[:8]


# 
# Synthetic data factories
# 

@pytest.fixture()
def rgb_frame() -> np.ndarray:
    """640×400 BGR uint8 frame (like OAK-D output)."""
    return np.random.randint(0, 255, (400, 640, 3), dtype=np.uint8)


@pytest.fixture()
def depth_frame() -> np.ndarray:
    """640×400 depth in metres (float32), range [0.2, 5.0]."""
    return np.random.uniform(0.2, 5.0, (400, 640)).astype(np.float32)


@pytest.fixture()
def imu_vector() -> np.ndarray:
    """16-float IMU vector matching IMUWorker/IMUBuffer schema."""
    return np.random.randn(16).astype(np.float32)


@pytest.fixture()
def segmentation_mask() -> np.ndarray:
    """640×400 segmentation mask (int32) — simulates ResNet18 output.

    Class IDs: 0=background, 1=road, 2=wall.
    """
    rng = np.random.default_rng(42)
    mask = np.zeros((400, 640), dtype=np.int32)
    # Central strip = road
    mask[100:300, :] = 1
    # Top & bottom strips = wall
    mask[:100, :] = 2
    mask[300:, :] = 2
    # Sprinkle noise
    noise_idx = rng.integers(0, 400 * 640, size=5000)
    mask.flat[noise_idx] = rng.integers(0, 3, size=5000)
    return mask


@pytest.fixture()
def pose_4x4() -> np.ndarray:
    """Random valid SE(3) 4×4 pose matrix (float32)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=42).as_matrix().astype(np.float32)
    t = np.random.randn(3).astype(np.float32) * 2.0
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


@pytest.fixture()
def point_cloud() -> np.ndarray:
    """Synthetic 3D point cloud (500 points)."""
    return np.random.randn(500, 3).astype(np.float32) * 5.0


@pytest.fixture()
def track_boundaries() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic oval outer + inner track boundaries in BEV (X,Z)."""
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


# 
# Buffer fixtures (auto-cleanup via finalizer)
# 

@pytest.fixture()
def frame_buffer():
    """Fresh FrameBuffer with unique shm name — auto-unlinked."""
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


# 
# Multiprocessing helpers
# 

@pytest.fixture()
def spawn_ctx():
    """Return a 'spawn' multiprocessing context."""
    return mp.get_context("spawn")


@pytest.fixture()
def mp_event():
    """Create a fresh multiprocessing Event (spawn context)."""
    ctx = mp.get_context("spawn")
    return ctx.Event()
