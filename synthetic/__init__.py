"""
Fake camera/depth/IMU data so the stack runs without hardware (tests, PILOT_SYNTHETIC=1, dev container).
For SLAM tests use closed_loop_generator (ground-truth poses).
"""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np

# focal length (px) of the synthetic cameras at 640 px width; must match
# synthetic_slam_settings.yaml (640x400, cx=320, cy=200)
SYNTH_FX = 456.99


class SyntheticFrameGenerator:
    """Straight corridor seen from an FPV camera, with a roughly matching depth map.

    wall_distance, corridor_length in m; speed (m/s) only sets the texture scroll rate.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 400,
        wall_distance: float = 1.5,
        corridor_length: float = 10.0,
        speed: float = 2.0,
    ) -> None:
        self.width = width
        self.height = height
        self.wall_distance = wall_distance
        self.corridor_length = corridor_length
        self.speed = speed

        # pinhole intrinsics, same as synthetic_slam_settings.yaml at 640 px
        # wide (the depth model below is not a true projection though)
        self.fx = SYNTH_FX * width / 640.0
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.camera_height = 0.15  # m

        self._frame_idx = 0
        self._t0 = time.monotonic()

    def next_fast(self) -> Tuple[np.ndarray, np.ndarray]:
        """Next (rgb uint8 HxWx3, depth float32 HxW m); depth is a rough model, not a true projection."""
        self._frame_idx += 1
        t = time.monotonic() - self._t0

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), self.corridor_length, dtype=np.float32)

        vp_x = self.width // 2
        vp_y = self.height // 2 - int(self.height * 0.05)

        rows = np.arange(self.height)
        cols = np.arange(self.width)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')

        # everything below the vanishing point is floor or wall
        floor_mask = row_grid > (vp_y + 20)
        frac = np.where(
            floor_mask,
            (row_grid - vp_y).astype(np.float32) / max(self.height - vp_y, 1),
            0.01,
        )

        floor_depth = np.clip(
            self.camera_height / np.maximum(frac * 0.3, 0.01),
            0.0,
            self.corridor_length,
        )

        wall_x_frac = (col_grid - vp_x).astype(np.float32) / max(self.width / 2, 1)
        wall_boundary = 0.15 / np.maximum(frac, 0.01) * 0.5

        left_wall = floor_mask & (wall_x_frac < -wall_boundary)
        right_wall = floor_mask & (wall_x_frac > wall_boundary)
        floor_only = floor_mask & ~left_wall & ~right_wall

        wall_depth = np.clip(
            self.wall_distance / np.maximum(np.abs(wall_x_frac), 0.01) * 0.5,
            0.0,
            self.corridor_length,
        )

        depth[floor_only] = floor_depth[floor_only]
        depth[left_wall] = wall_depth[left_wall]
        depth[right_wall] = wall_depth[right_wall]

        rgb[floor_only] = [60, 60, 60]
        rgb[left_wall] = [100, 120, 140]
        rgb[right_wall] = [100, 120, 140]
        sky_mask = ~floor_mask
        rgb[sky_mask] = [50, 50, 50]

        # dashed centre line, scrolls with t * speed
        scroll = int((t * self.speed * 30) % 40)
        centre_band = floor_only & (np.abs(col_grid - vp_x) < 3)
        dash = ((row_grid + scroll) % 20) < 10
        rgb[centre_band & dash] = [180, 180, 180]

        return rgb, depth


class SyntheticIMUGenerator:
    """IMU for straight-line driving, IMUBuffer layout
    [qw, qx, qy, qz, ax, ay, az, throttle_raw, steering_raw] (pulse widths in µs).

    accel_noise is the m/s^2 std on gravity-on-z accel; throttle/steering are held constant.
    """

    def __init__(
        self,
        throttle: float = 1500.0,
        steering: float = 1461.0,
        accel_noise: float = 0.05,
    ) -> None:
        self.throttle = throttle
        self.steering = steering
        self.accel_noise = accel_noise
        self._step = 0

    def next(self) -> np.ndarray:
        self._step += 1

        qw = 1.0
        qx = np.random.normal(0, 0.001)
        qy = np.random.normal(0, 0.001)
        qz = np.random.normal(0, 0.001)

        # at rest, gravity on +z
        ax = np.random.normal(0, self.accel_noise)
        ay = np.random.normal(0, self.accel_noise)
        az = 9.81 + np.random.normal(0, self.accel_noise)

        return np.array(
            [qw, qx, qy, qz, ax, ay, az, self.throttle, self.steering],
            dtype=np.float32,
        )
