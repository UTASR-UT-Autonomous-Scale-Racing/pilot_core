"""closed_loop_generator -- Synthetic RGBD + IMU frame generator for ORB-SLAM3.

The camera follows an elliptical path between two elliptical track walls
(inner and outer) sharing the same centre.  The camera looks forward along
the direction of travel.  IMU readings are derived analytically from the
elliptical trajectory.

Public API
----------
ClosedLoopFrameGenerator
    .generate_all() -> list[(rgb, depth, pose_4x4, timestamp, imu)]
"""

from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np


class ClosedLoopFrameGenerator:
    """Generate a full closed-loop RGBD + IMU sequence for ORB-SLAM3.

    The camera moves along an elliptical path between two pre-rendered
    elliptical track walls (inner smaller, outer larger), looking forward
    along the direction of travel.  IMU readings are modelled from
    analytical derivatives of the elliptical trajectory.

    Parameters
    ----------
    width, height : int
        Frame dimensions (default 640x480).
    n_frames : int
        Total frames in one full loop (default 200).
    straight_len : float
        Controls semi-major axis: a = straight_len/2 + turn_radius.
    turn_radius : float
        Controls semi-minor axis: b = turn_radius.
    wall_distance : float
        Half-width of the track (distance from camera path to each wall).
    wall_height : float
        Height of the track walls in metres (default 0.5).
    camera_height : float
        Camera height above ground plane in metres (default 0.15).
    camera_pitch_deg : float
        Downward pitch of the camera in degrees (default 12).
    fps : float
        Simulated frame rate (default 25.0).
    seed : int
        Random seed for reproducible texture (default 42).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        n_frames: int = 200,
        straight_len: float = 4.0,
        turn_radius: float = 2.0,
        wall_distance: float = 0.8,
        wall_height: float = 1.5,
        camera_height: float = 0.15,
        camera_pitch_deg: float = 12.0,
        fps: float = 25.0,
        seed: int = 42,
    ) -> None:
        self.W = width
        self.H = height
        self.n_frames = n_frames
        self.straight = straight_len
        self.radius = turn_radius
        self.wall_dist = wall_distance
        self.wall_height = wall_height
        self.cam_h = camera_height
        self.pitch = math.radians(camera_pitch_deg)
        self.fps = fps
        self.rng = np.random.RandomState(seed)

        # Camera intrinsics (match the SLAM settings YAML)
        self.fx = 456.99
        self.fy = 456.99
        self.cx = width / 2.0
        self.cy = height / 2.0

        # Pre-generate ground texture (a reusable tile)
        self._ground_tile = self._make_ground_tile()
        # Pre-generate random feature dots on the ground
        self._dot_positions = self._make_dot_positions()
        # Pre-generate wall marker tile for distinctive landmarks
        self._wall_tile = self._make_wall_tile()

        # Camera-path ellipse semi-axes
        self.semi_major = self.straight / 2.0 + self.radius
        self.semi_minor = self.radius

        # Inner and outer wall ellipses (concentric, offset by wall_distance)
        self._inner_a = max(self.semi_major - self.wall_dist, 0.1)
        self._inner_b = max(self.semi_minor - self.wall_dist, 0.1)
        self._outer_a = self.semi_major + self.wall_dist
        self._outer_b = self.semi_minor + self.wall_dist

        # Angular velocity for one lap (rad/s)
        self.omega = 2.0 * math.pi * self.fps / self.n_frames

    def _ellipse_path(self, theta: float) -> Tuple[float, float, float]:
        """Return (x, z, heading) for angle *theta* on the ellipse.

        The path is a closed ellipse in the XZ-plane::

            x(θ) = a·cos(θ)    z(θ) = b·sin(θ)

        with counter-clockwise traversal (θ increasing).

        Parameters
        ----------
        theta : float
            Angle in radians, typically in [0, 2π).

        Returns
        -------
        x, z : float
            World-space position on the ellipse.
        heading : float
            Direction of travel (tangent angle, radians).
        """
        a = self.semi_major
        b = self.semi_minor
        x = a * math.cos(theta)
        z = b * math.sin(theta)
        # Tangent direction: dx/dθ = -a sin θ,  dz/dθ = b cos θ
        heading = math.atan2(b * math.cos(theta), -a * math.sin(theta))
        return x, z, heading

    def _ellipse_derivatives(
        self, theta: float, omega: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute velocity, acceleration and heading-rate on the ellipse.

        Parameters
        ----------
        theta : float
            Current angle on the ellipse (radians).
        omega : float
            Angular rate dθ/dt (radians/s, constant for uniform rotation).

        Returns
        -------
        vel_world : ndarray (3,) – velocity [vx, 0, vz] m/s
        acc_world : ndarray (3,) – acceleration [ax, 0, az] m/s²
        heading_rate : float     – dheading/dt (rad/s)
        """
        a = self.semi_major
        b = self.semi_minor
        st, ct = math.sin(theta), math.cos(theta)

        # Velocity
        vx = -a * st * omega
        vz = b * ct * omega

        # Acceleration
        ax = -a * ct * omega ** 2
        az = -b * st * omega ** 2

        # Heading rate via quotient rule on atan2(b·cosθ, -a·sinθ)
        hx = -a * st          # tangent x-component
        hy = b * ct           # tangent z-component
        dhx = -a * ct * omega
        dhy = -b * st * omega
        denom = hx ** 2 + hy ** 2
        heading_rate = (hx * dhy - hy * dhx) / denom if denom > 1e-12 else 0.0

        return (
            np.array([vx, 0.0, vz], dtype=np.float64),
            np.array([ax, 0.0, az], dtype=np.float64),
            heading_rate,
        )

    def _make_ground_tile(self) -> np.ndarray:
        """Create a 256×256 BGR texture tile with checkerboard + noise."""
        tile = np.zeros((256, 256, 3), dtype=np.uint8)
        # Checkerboard (32-pixel squares)
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 0:
                    colour = (90 + self.rng.randint(0, 30),
                              90 + self.rng.randint(0, 30),
                              90 + self.rng.randint(0, 30))
                else:
                    colour = (40 + self.rng.randint(0, 20),
                              40 + self.rng.randint(0, 20),
                              40 + self.rng.randint(0, 20))
                tile[row*32:(row+1)*32, col*32:(col+1)*32] = colour

        # Add random noise for more ORB features
        noise = self.rng.randint(0, 15, (256, 256, 3), dtype=np.uint8)
        tile = np.clip(tile.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        # Add random circles and crosses for distinctive features
        for _ in range(40):
            cx = self.rng.randint(10, 246)
            cy = self.rng.randint(10, 246)
            r = self.rng.randint(3, 12)
            colour = tuple(int(c) for c in self.rng.randint(60, 200, 3))
            if self.rng.random() > 0.5:
                cv2.circle(tile, (cx, cy), r, colour, -1)
            else:
                cv2.drawMarker(tile, (cx, cy), colour,
                               cv2.MARKER_CROSS, r * 2, 2)
        return tile

    def _make_dot_positions(self) -> np.ndarray:
        """Generate random 3D dot positions on the ground plane (y=0)."""
        L = self.straight
        R = self.radius
        n_dots = 500
        # Scatter dots in and around the track
        x = self.rng.uniform(-L / 2 - R - 1, L / 2 + R + 1, n_dots)
        z = self.rng.uniform(-R - 1, R + 1, n_dots)
        return np.column_stack([x, np.zeros(n_dots), z]).astype(np.float32)

    def _make_wall_tile(self) -> np.ndarray:
        """Create a 256×256 BGR tile for wall markers.

        Contains coloured rectangles, arrows, and numbers at regular
        intervals to provide distinctive visual landmarks that help
        ORB-SLAM3 track and relocalize, especially in turns.
        """
        tile = np.zeros((256, 256, 3), dtype=np.uint8)

        # Coloured rectangles (8 patches spread across the tile)
        colours = [
            (200, 60, 60), (60, 200, 60), (60, 60, 200),
            (200, 200, 60), (200, 60, 200), (60, 200, 200),
            (180, 120, 60), (120, 60, 180),
        ]
        for idx, col in enumerate(colours):
            row = (idx // 4) * 128
            col_off = (idx % 4) * 64
            cv2.rectangle(
                tile,
                (col_off + 8, row + 8),
                (col_off + 56, row + 56),
                col, -1,
            )
            # Add a border for extra edges
            cv2.rectangle(
                tile,
                (col_off + 8, row + 8),
                (col_off + 56, row + 56),
                (255, 255, 255), 1,
            )

        # Add digit text for unique identifiers
        for i in range(8):
            x = (i % 4) * 64 + 22
            y = (i // 4) * 128 + 80
            cv2.putText(
                tile, str(i), (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2,
            )

        # Add arrows / crosses
        for _ in range(20):
            cx = self.rng.randint(10, 246)
            cy = self.rng.randint(10, 246)
            col = tuple(int(c) for c in self.rng.randint(130, 255, 3))
            cv2.drawMarker(tile, (cx, cy), col, cv2.MARKER_DIAMOND, 8, 2)

        # Random noise for texture
        noise = self.rng.randint(0, 10, (256, 256, 3), dtype=np.uint8)
        tile = np.clip(tile.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

        return tile

    def _intersect_elliptic_cyl(
        self,
        cam_pos: np.ndarray,
        ray_world: np.ndarray,
        a: float,
        b: float,
    ) -> np.ndarray:
        """Intersect per-pixel rays with an elliptical vertical cylinder.

        Cylinder:  (x/a)^2 + (z/b)^2 = 1, y in [-wall_height, 0].

        Returns t-values (H, W); inf where no valid hit.
        """
        H, W = ray_world.shape[:2]
        ox, oz = cam_pos[0], cam_pos[2]
        dx = ray_world[..., 0]
        dz = ray_world[..., 2]

        # Normalise by semi-axes
        px, rx = ox / a, dx / a
        pz, rz = oz / b, dz / b

        A = rx ** 2 + rz ** 2
        B = 2.0 * (px * rx + pz * rz)
        C = px ** 2 + pz ** 2 - 1.0

        disc = B ** 2 - 4.0 * A * C

        t_hit = np.full((H, W), np.inf, dtype=np.float64)
        valid = disc >= 0
        sq = np.sqrt(np.maximum(disc, 0))
        denom = 2.0 * A + 1e-12

        # Try both roots; keep first positive hit within wall height
        for sign in (-1.0, 1.0):
            t_cand = (-B + sign * sq) / denom
            hit_y = cam_pos[1] + t_cand * ray_world[..., 1]
            ok = (
                valid
                & (t_cand > 0.05)
                & (t_hit == np.inf)
                & (hit_y >= -self.wall_height)
                & (hit_y <= 0.0)
            )
            t_hit = np.where(ok, t_cand, t_hit)

        return t_hit

    def _render_frame_fast(
        self, cam_x: float, cam_z: float, heading: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized ray-casting renderer for elliptical track walls.

        For each pixel we cast a ray from the camera, find its intersection
        with the ground plane (y=0) and the inner/outer elliptical cylinder
        walls, then pick the closest hit.
        """
        H, W = self.H, self.W

        c_h = math.cos(heading)
        s_h = math.sin(heading)

        # Camera-to-world rotation R_wc.
        # Composed as R_yaw @ R_pitch where:
        #   R_yaw   rotates so camera looks forward along the heading
        #   R_pitch tilts the camera downward by self.pitch radians
        cp = math.cos(self.pitch)
        sp = math.sin(self.pitch)
        R_pitch = np.array([
            [1,  0,   0 ],
            [0,  cp,  sp],
            [0, -sp,  cp],
        ], dtype=np.float64)
        R_yaw = np.array([
            [ s_h, 0, c_h],
            [ 0,   1, 0  ],
            [-c_h, 0, s_h],
        ], dtype=np.float64)
        R_wc = R_yaw @ R_pitch

        cam_pos = np.array([cam_x, -self.cam_h, cam_z], dtype=np.float64)

        # -- Per-pixel ray directions (camera frame) --------------------
        uu = np.arange(W, dtype=np.float64)
        vv = np.arange(H, dtype=np.float64)
        u_grid, v_grid = np.meshgrid(uu, vv)  # (H, W)

        dx = (u_grid - self.cx) / self.fx   # (H, W)
        dy = (v_grid - self.cy) / self.fy
        dz = np.ones_like(dx)

        # Stack into (H, W, 3) and rotate to world frame
        ray_cam = np.stack([dx, dy, dz], axis=-1)                 # (H, W, 3)
        ray_world = np.einsum('ij,hwj->hwi', R_wc, ray_cam)      # (H, W, 3)

        # -- Intersect with ground plane y = 0 -------------------------
        ry = ray_world[..., 1]  # (H, W)
        with np.errstate(divide='ignore', invalid='ignore'):
            t_ground = np.where(np.abs(ry) > 1e-8, -cam_pos[1] / ry, np.inf)
        t_ground[t_ground <= 0.05] = np.inf

        # -- Intersect with inner elliptical wall ----------------------
        t_inner = self._intersect_elliptic_cyl(
            cam_pos, ray_world, self._inner_a, self._inner_b,
        )

        # -- Intersect with outer elliptical wall ----------------------
        t_outer = self._intersect_elliptic_cyl(
            cam_pos, ray_world, self._outer_a, self._outer_b,
        )

        # -- Find closest hit ------------------------------------------
        t_all = np.stack([t_ground, t_inner, t_outer], axis=-1)  # (H, W, 3)
        which = np.argmin(t_all, axis=-1)  # 0=ground, 1=inner, 2=outer
        t_min = np.min(t_all, axis=-1)

        # Clamp max depth
        max_depth = 20.0
        no_hit = t_min > max_depth
        t_min[no_hit] = max_depth

        # World-space hit positions
        hit_world = cam_pos[np.newaxis, np.newaxis, :] + \
            t_min[..., np.newaxis] * ray_world

        # -- Depth (in camera Z) ---------------------------------------
        cam_fwd = R_wc[:, 2]
        delta = hit_world - cam_pos[np.newaxis, np.newaxis, :]
        depth = (delta * cam_fwd).sum(axis=-1).astype(np.float32)
        depth[depth < 0.05] = 0.0

        # -- RGB texturing ---------------------------------------------
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Ground pixels (only between inner & outer ellipse = track surface)
        gnd_mask = (which == 0) & (~no_hit)
        if gnd_mask.any():
            gx = hit_world[..., 0][gnd_mask]
            gz = hit_world[..., 2][gnd_mask]
            # Check if on the track (between inner and outer ellipse)
            r_inner = (gx / self._inner_a) ** 2 + (gz / self._inner_b) ** 2
            r_outer = (gx / self._outer_a) ** 2 + (gz / self._outer_b) ** 2
            on_track = (r_inner >= 1.0) & (r_outer <= 1.0)
            # Track surface: rich checkerboard texture
            tx = (gx * 50).astype(np.int32) % 256
            tz = (gz * 50).astype(np.int32) % 256
            tx[tx < 0] += 256
            tz[tz < 0] += 256
            colours = self._ground_tile[tz, tx]
            # Off-track: darker, less textured
            off_col = (colours * 0.3).astype(np.uint8)
            colours = np.where(on_track[:, None], colours, off_col)
            rgb[gnd_mask] = colours

        # Inner wall pixels (warm tones)
        iw_mask = (which == 1) & (~no_hit)
        if iw_mask.any():
            wx = hit_world[..., 0][iw_mask]
            wz = hit_world[..., 2][iw_mask]
            wy = hit_world[..., 1][iw_mask]
            # Parametric angle along the ellipse for texture coord
            ang = np.arctan2(wz / self._inner_b, wx / self._inner_a)
            mtx = ((ang * 40).astype(np.int32)) % 256
            mty = ((-wy) * 200).astype(np.int32) % 256
            mtx[mtx < 0] += 256
            mty[mty < 0] += 256
            # Base warm stripe pattern
            stripe = ((wx * 20 + wz * 20).astype(np.int32)) % 2
            c0 = np.array([140, 100, 80], dtype=np.uint8)
            c1 = np.array([100, 80, 60], dtype=np.uint8)
            base_col = np.where(stripe[:, None] == 1, c0, c1)
            # Overlay wall marker tile
            marker_col = self._wall_tile[mty, mtx]
            bright = marker_col.max(axis=1) > 120
            base_col[bright] = marker_col[bright]
            rgb[iw_mask] = base_col

        # Outer wall pixels (cool tones)
        ow_mask = (which == 2) & (~no_hit)
        if ow_mask.any():
            wx = hit_world[..., 0][ow_mask]
            wz = hit_world[..., 2][ow_mask]
            wy = hit_world[..., 1][ow_mask]
            ang = np.arctan2(wz / self._outer_b, wx / self._outer_a)
            mtx = ((ang * 40).astype(np.int32)) % 256
            mty = ((-wy) * 200).astype(np.int32) % 256
            mtx[mtx < 0] += 256
            mty[mty < 0] += 256
            stripe = ((wx * 20 + wz * 20).astype(np.int32)) % 2
            c0 = np.array([80, 100, 140], dtype=np.uint8)
            c1 = np.array([60, 80, 100], dtype=np.uint8)
            base_col = np.where(stripe[:, None] == 1, c0, c1)
            marker_col = self._wall_tile[mty, mtx]
            bright = marker_col.max(axis=1) > 120
            base_col[bright] = marker_col[bright]
            rgb[ow_mask] = base_col

        # Sky (miss)
        rgb[no_hit] = (30, 30, 30)
        depth[no_hit] = 0.0

        # Gaussian noise for realism
        noise = self.rng.normal(0, 2, rgb.shape).astype(np.int16)
        rgb = np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return rgb, depth

    def generate_all(
        self,
        n_loops: int = 1,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]]:
        """Generate a continuous multi-loop RGBD + IMU sequence.

        Parameters
        ----------
        n_loops : int
            Number of complete laps around the ellipse (default 1).

        Returns
        -------
        frames : list of (rgb, depth, pose_4x4, timestamp, imu)
            Each element is a tuple of:
            - rgb       : uint8 (H, W, 3) BGR frame
            - depth     : float32 (H, W) depth in metres
            - pose_4x4  : float32 (4, 4) ground-truth camera-to-world SE3
            - timestamp : float, seconds
            - imu       : dict with keys
                  'accel' : ndarray (3,) accelerometer (m/s²) in body frame
                  'gyro'  : ndarray (3,) gyroscope (rad/s)  in body frame
        """
        frames: list = []
        dt = 1.0 / self.fps
        total_frames = self.n_frames * n_loops
        omega = self.omega  # constant angular velocity dθ/dt

        # Gravity in world frame (+Y is downward in our convention)
        g_world = np.array([0.0, 9.81, 0.0], dtype=np.float64)

        for i in range(total_frames):
            theta = 2.0 * math.pi * (i % self.n_frames) / self.n_frames
            x, z, heading = self._ellipse_path(theta)
            timestamp = i * dt

            c_h = math.cos(heading)
            s_h = math.sin(heading)
            cp = math.cos(self.pitch)
            sp = math.sin(self.pitch)

            # Camera-to-world rotation: forward-looking along travel
            # direction with downward pitch.
            R_pitch = np.array([
                [1,  0,   0 ],
                [0,  cp,  sp],
                [0, -sp,  cp],
            ], dtype=np.float32)
            R_yaw = np.array([
                [ s_h, 0, c_h],
                [ 0,   1, 0  ],
                [-c_h, 0, s_h],
            ], dtype=np.float32)
            R_wc = R_yaw @ R_pitch

            pose = np.eye(4, dtype=np.float32)
            pose[:3, :3] = R_wc
            pose[0, 3] = x
            pose[1, 3] = -self.cam_h
            pose[2, 3] = z

            # ── Render ────────────────────────────────────────────────
            rgb, depth = self._render_frame_fast(x, z, heading)

            # ── IMU ─────────────────────────────────────────────────
            vel_w, acc_w, heading_rate = self._ellipse_derivatives(
                theta, omega,
            )

            # World-to-camera rotation (float64 for precision)
            R_cw = R_wc.astype(np.float64).T

            # Gyroscope: body-frame angular velocity
            # Rotation is purely about world Y axis at heading_rate
            omega_world = np.array([0.0, heading_rate, 0.0])
            gyro = R_cw @ omega_world

            # Accelerometer: specific force in body frame
            # a_measured = R_cw @ (a_kinematic - g_world)
            accel = R_cw @ (acc_w - g_world)

            # Add small sensor noise for realism
            gyro  += self.rng.normal(0, 0.005, 3)
            accel += self.rng.normal(0, 0.02, 3)

            imu = {
                'accel': accel.astype(np.float32),
                'gyro':  gyro.astype(np.float32),
            }

            frames.append((rgb, depth, pose, timestamp, imu))

        return frames
