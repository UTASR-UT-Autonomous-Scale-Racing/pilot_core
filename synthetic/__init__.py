"""
synthetic — Procedural generators for synthetic FPV camera frames, depth maps,
and IMU data.  Used when no OAK-D camera or real hardware is available
(e.g. in the dev-local Docker container for testing).

Generators produce data that mimics the view from an FPV camera going
straight down a corridor with track walls on each side.

Public API
----------
SyntheticFrameGenerator
    .next()  → (rgb, depth)       # uint8 (H,W,3), float32 (H,W) metres
SyntheticIMUGenerator
    .next()  → imu_vector          # float32 (16,)
"""

from __future__ import annotations

import math
import time
from typing import Tuple

import cv2
import numpy as np


class SyntheticFrameGenerator:
    """Generate synthetic FPV RGB + depth frames.

    Simulates a camera looking down a straight corridor with walls on the
    left and right.  The floor is dark grey, walls are lighter, and the
    depth map reflects the 3D geometry.

    Parameters
    ----------
    width : int
        Frame width in pixels (default 640).
    height : int
        Frame height in pixels (default 400).
    wall_distance : float
        Distance to each wall in metres (default 1.5).
    corridor_length : float
        Visible corridor length in metres (default 10.0).
    speed : float
        Simulated forward speed in m/s — controls texture scrolling
        (default 2.0).
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

        # Camera intrinsics (simple pinhole)
        self.fx = width * 0.8
        self.fy = width * 0.8
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.camera_height = 0.15  # 15 cm (scale car)

        self._frame_idx = 0
        self._t0 = time.monotonic()

    def next(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the next (rgb, depth) pair.

        Returns
        -------
        rgb : np.ndarray
            Shape ``(H, W, 3)`` dtype ``uint8`` BGR frame.
        depth : np.ndarray
            Shape ``(H, W)`` dtype ``float32`` depth in metres.
        """
        self._frame_idx += 1
        t = time.monotonic() - self._t0

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), self.corridor_length, dtype=np.float32)

        # Vanishing point
        vp_x = self.width // 2
        vp_y = self.height // 2 - int(self.height * 0.05)

        # --- Floor (dark grey trapezoid) ---
        floor_colour = (60, 60, 60)
        floor_pts = np.array([
            [0, self.height],
            [self.width, self.height],
            [vp_x + 40, vp_y + 20],
            [vp_x - 40, vp_y + 20],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [floor_pts], floor_colour)

        # --- Left wall ---
        left_wall_colour = (100, 120, 140)
        scroll = int((t * self.speed * 30) % 40)
        left_pts = np.array([
            [0, self.height],
            [0, vp_y - 30],
            [vp_x - 40, vp_y + 20],
            [int(self.width * 0.15), self.height],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [left_pts], left_wall_colour)
        # Vertical stripe texture on left wall
        for i in range(0, self.height, 40):
            y = (i + scroll) % self.height
            if y < vp_y:
                continue
            x_start = 0
            frac = (y - vp_y) / max(self.height - vp_y, 1)
            x_end = int(frac * self.width * 0.15)
            cv2.line(rgb, (x_start, y), (x_end, y), (80, 100, 120), 1)

        # --- Right wall ---
        right_wall_colour = (100, 120, 140)
        right_pts = np.array([
            [self.width, self.height],
            [self.width, vp_y - 30],
            [vp_x + 40, vp_y + 20],
            [int(self.width * 0.85), self.height],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [right_pts], right_wall_colour)
        # Vertical stripe texture on right wall
        for i in range(0, self.height, 40):
            y = (i + scroll) % self.height
            if y < vp_y:
                continue
            frac = (y - vp_y) / max(self.height - vp_y, 1)
            x_start = self.width - int(frac * self.width * 0.15)
            x_end = self.width
            cv2.line(rgb, (x_start, y), (x_end, y), (80, 100, 120), 1)

        # --- Sky / far end ---
        sky_colour = (50, 50, 50)
        sky_pts = np.array([
            [0, 0],
            [self.width, 0],
            [self.width, vp_y - 30],
            [vp_x + 40, vp_y + 20],
            [vp_x - 40, vp_y + 20],
            [0, vp_y - 30],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [sky_pts], sky_colour)

        # --- Generate depth map from geometry ---
        depth = self._generate_depth_map(vp_x, vp_y)

        return rgb, depth

    def _generate_depth_map(self, vp_x: int, vp_y: int) -> np.ndarray:
        """Compute a per-pixel depth map for the corridor scene.

        Uses a simple geometric model:
        - Floor pixels: depth from camera height and pixel row
        - Wall pixels: depth from wall distance and pixel column
        - Sky/far pixels: corridor_length
        """
        depth = np.full((self.height, self.width), self.corridor_length, dtype=np.float32)

        for row in range(self.height):
            for col in range(self.width):
                # Below vanishing point → floor or wall
                if row > vp_y + 20:
                    frac = (row - vp_y) / max(self.height - vp_y, 1)
                    floor_depth = self.camera_height / max(frac * 0.3, 0.01)
                    floor_depth = min(floor_depth, self.corridor_length)

                    # Check if pixel is in left wall region
                    wall_x_frac = (col - vp_x) / max(self.width / 2, 1)
                    left_boundary = -0.15 / max(frac, 0.01) * 0.5
                    right_boundary = 0.15 / max(frac, 0.01) * 0.5

                    if wall_x_frac < left_boundary:
                        # Left wall
                        wall_depth = self.wall_distance / max(abs(wall_x_frac), 0.01) * 0.5
                        depth[row, col] = min(wall_depth, self.corridor_length)
                    elif wall_x_frac > right_boundary:
                        # Right wall
                        wall_depth = self.wall_distance / max(abs(wall_x_frac), 0.01) * 0.5
                        depth[row, col] = min(wall_depth, self.corridor_length)
                    else:
                        depth[row, col] = floor_depth

        return depth

    def next_fast(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fast vectorized version of next() — no per-pixel loop.

        Returns same format as next() but uses numpy broadcasting for speed.
        """
        self._frame_idx += 1
        t = time.monotonic() - self._t0

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        depth = np.full((self.height, self.width), self.corridor_length, dtype=np.float32)

        vp_x = self.width // 2
        vp_y = self.height // 2 - int(self.height * 0.05)

        # Create coordinate grids
        rows = np.arange(self.height)
        cols = np.arange(self.width)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')

        # Floor region: below vanishing point
        floor_mask = row_grid > (vp_y + 20)
        frac = np.where(
            floor_mask,
            (row_grid - vp_y).astype(np.float32) / max(self.height - vp_y, 1),
            0.01,
        )

        # Floor depth
        floor_depth = np.clip(
            self.camera_height / np.maximum(frac * 0.3, 0.01),
            0.0,
            self.corridor_length,
        )

        # Wall regions (perspective)
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

        # RGB rendering (simplified but fast)
        # Floor
        rgb[floor_only] = [60, 60, 60]
        # Left wall
        rgb[left_wall] = [100, 120, 140]
        # Right wall
        rgb[right_wall] = [100, 120, 140]
        # Sky
        sky_mask = ~floor_mask
        rgb[sky_mask] = [50, 50, 50]

        # Add dashed centre line on floor
        scroll = int((t * self.speed * 30) % 40)
        centre_band = floor_only & (np.abs(col_grid - vp_x) < 3)
        dash = ((row_grid + scroll) % 20) < 10
        rgb[centre_band & dash] = [180, 180, 180]

        return rgb, depth


class SyntheticIMUGenerator:
    """Generate synthetic IMU data simulating straight-line driving.

    Produces 16-float vectors:
    [qw, qx, qy, qz, ax, ay, az, gx, gy, gz, mx, my, mz, throttle_raw, steering_raw, timestamp]

    For straight driving: quaternion ≈ identity, small accelerometer noise,
    throttle ≈ constant, steering ≈ centre.

    Parameters
    ----------
    throttle : float
        Simulated raw throttle (default 1500 — neutral PWM).
    steering : float
        Simulated raw steering (default 1461 — neutral PWM).
    accel_noise : float
        Gaussian noise std for accelerometer readings in m/s² (default 0.05).
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
        """Generate a single 16-float IMU vector.

        Returns
        -------
        np.ndarray — shape ``(16,)`` dtype ``float32``.
        """
        self._step += 1

        # Identity quaternion with tiny perturbation
        qw = 1.0
        qx = np.random.normal(0, 0.001)
        qy = np.random.normal(0, 0.001)
        qz = np.random.normal(0, 0.001)

        # Accelerometer: gravity on Z + noise
        ax = np.random.normal(0, self.accel_noise)
        ay = np.random.normal(0, self.accel_noise)
        az = 9.81 + np.random.normal(0, self.accel_noise)

        # Gyro + mag placeholders for synthetic straight driving
        gx = np.random.normal(0, 0.02)
        gy = np.random.normal(0, 0.02)
        gz = np.random.normal(0, 0.02)
        mx = np.random.normal(0, 0.1)
        my = np.random.normal(0, 0.1)
        mz = np.random.normal(0, 0.1)

        timestamp = float(time.time())

        return np.array(
            [
                qw, qx, qy, qz,
                ax, ay, az,
                gx, gy, gz,
                mx, my, mz,
                self.throttle, self.steering,
                timestamp,
            ],
            dtype=np.float32,
        )
