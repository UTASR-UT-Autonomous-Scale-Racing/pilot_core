"""
Fake camera/depth/IMU data so the stack runs without hardware: tests,
PILOT_SYNTHETIC=1, and the dev Docker container.

SyntheticFrameGenerator fakes an FPV camera driving straight down a corridor
between two walls. next() / next_fast() return (rgb uint8 HxWx3, depth
float32 HxW in metres). SyntheticIMUGenerator.next() returns a (9,) float32
vector in the IMUBuffer layout.

For SLAM tests you probably want closed_loop_generator instead; it has a
closed elliptical track with ground-truth poses.
"""

from __future__ import annotations

import time
from typing import Tuple

import cv2
import numpy as np


class SyntheticFrameGenerator:
    """Straight corridor seen from an FPV camera: dark grey floor, lighter
    walls either side, and a depth map that roughly matches.

    wall_distance and corridor_length are in metres. speed (m/s) only
    affects how fast the wall texture scrolls.
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

        # pinhole intrinsics
        self.fx = width * 0.8
        self.fy = width * 0.8
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.camera_height = 0.15  # m

        self._frame_idx = 0
        self._t0 = time.monotonic()

    def next(self) -> Tuple[np.ndarray, np.ndarray]:
        """Next (rgb, depth) pair. Slow (per-pixel depth loop), see next_fast."""
        self._frame_idx += 1
        t = time.monotonic() - self._t0

        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        vp_x = self.width // 2
        vp_y = self.height // 2 - int(self.height * 0.05)

        # floor
        floor_colour = (60, 60, 60)
        floor_pts = np.array([
            [0, self.height],
            [self.width, self.height],
            [vp_x + 40, vp_y + 20],
            [vp_x - 40, vp_y + 20],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [floor_pts], floor_colour)

        # left wall
        left_wall_colour = (100, 120, 140)
        scroll = int((t * self.speed * 30) % 40)
        left_pts = np.array([
            [0, self.height],
            [0, vp_y - 30],
            [vp_x - 40, vp_y + 20],
            [int(self.width * 0.15), self.height],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [left_pts], left_wall_colour)
        # stripes scroll with t * speed so there's apparent motion
        for i in range(0, self.height, 40):
            y = (i + scroll) % self.height
            if y < vp_y:
                continue
            x_start = 0
            frac = (y - vp_y) / max(self.height - vp_y, 1)
            x_end = int(frac * self.width * 0.15)
            cv2.line(rgb, (x_start, y), (x_end, y), (80, 100, 120), 1)

        # right wall
        right_wall_colour = (100, 120, 140)
        right_pts = np.array([
            [self.width, self.height],
            [self.width, vp_y - 30],
            [vp_x + 40, vp_y + 20],
            [int(self.width * 0.85), self.height],
        ], dtype=np.int32)
        cv2.fillPoly(rgb, [right_pts], right_wall_colour)
        for i in range(0, self.height, 40):
            y = (i + scroll) % self.height
            if y < vp_y:
                continue
            frac = (y - vp_y) / max(self.height - vp_y, 1)
            x_start = self.width - int(frac * self.width * 0.15)
            x_end = self.width
            cv2.line(rgb, (x_start, y), (x_end, y), (80, 100, 120), 1)

        # sky / far end of the corridor
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

        depth = self._generate_depth_map(vp_x, vp_y)

        return rgb, depth

    def _generate_depth_map(self, vp_x: int, vp_y: int) -> np.ndarray:
        """Rough per-pixel depth. Not a real projection: floor depth falls
        off with row, wall depth with column, and anything above the
        vanishing point is just corridor_length.
        """
        depth = np.full((self.height, self.width), self.corridor_length, dtype=np.float32)

        for row in range(self.height):
            for col in range(self.width):
                if row > vp_y + 20:
                    frac = (row - vp_y) / max(self.height - vp_y, 1)
                    floor_depth = self.camera_height / max(frac * 0.3, 0.01)
                    floor_depth = min(floor_depth, self.corridor_length)

                    wall_x_frac = (col - vp_x) / max(self.width / 2, 1)
                    left_boundary = -0.15 / max(frac, 0.01) * 0.5
                    right_boundary = 0.15 / max(frac, 0.01) * 0.5

                    if wall_x_frac < left_boundary:
                        wall_depth = self.wall_distance / max(abs(wall_x_frac), 0.01) * 0.5
                        depth[row, col] = min(wall_depth, self.corridor_length)
                    elif wall_x_frac > right_boundary:
                        wall_depth = self.wall_distance / max(abs(wall_x_frac), 0.01) * 0.5
                        depth[row, col] = min(wall_depth, self.corridor_length)
                    else:
                        depth[row, col] = floor_depth

        return depth

    def next_fast(self) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorised next(). Same depth model and output format, but the RGB
        is flat-shaded with a scrolling dashed centre line instead of wall
        stripes.
        """
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
    """IMU for straight-line driving, in the IMUBuffer layout
    [qw, qx, qy, qz, ax, ay, az, throttle_raw, steering_raw].

    Quaternion is ~identity, accel is gravity on z plus Gaussian noise
    (accel_noise, m/s^2 std), and throttle/steering are held constant. The
    steering default of 1461 is the neutral pulse width in µs.
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
