"""
RGB-D for ORB-SLAM3 from a camera with no depth: the AutoDRIVE stock sim sends a
192x108 front camera and a 2D LiDAR, nothing else. lidar_to_depth() turns the current
scan into a depth image in the camera's frame: walls are the scan extruded vertically
(floor to wall_height), the floor is flat, anything else is 0 (no depth). Only the
current scan is used, no pose. CameraModel holds the intrinsics and the mount (fitted
to the sim's images by scripts/autodrive_camera_fit.py, docs/autodrive.md), and the
mask of the car's own body, which fills the bottom of the stock camera's view and must
not give ORB features.

Frames: vehicle x forward, y left, z up (REP-103), origin at the rear axle on the
ground plane; camera optical frame x right, y down, z forward; depth = optical z (m).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

ANGLE_MIN = -2.35619449        # rad, LiDAR beam 0 (right)
ANGLE_INCREMENT = 0.00436332   # rad / beam


@dataclass
class CameraModel:
    width: int = 192
    height: int = 108
    vfov_deg: float = 60.0         # vertical field of view (Unity camera)
    cam_x: float = -0.015          # m, ahead of the rear axle (devkit TF)
    cam_height: float = 0.21       # m above the floor (0.15 above the rear-axle frame + 0.06)
    pitch_deg: float = 10.0        # down (devkit TF: 10 deg about y)
    lidar_x: float = 0.2733        # m, ahead of the rear axle
    wall_height: float = 0.30      # m, top of the track walls
    max_range: float = 10.0        # m, LiDAR max / depth beyond this is 0
    scale: float = 1.0             # output resolution = scale x camera resolution
    mask_path: Optional[str] = None   # PNG, nonzero = car body (at camera resolution)
    mask: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def size(self) -> tuple[int, int]:
        return int(round(self.width * self.scale)), int(round(self.height * self.scale))

    def intrinsics(self) -> tuple[float, float, float, float]:
        """fx, fy, cx, cy at the output resolution (square pixels, centred)."""
        w, h = self.size
        fy = (h / 2.0) / np.tan(np.radians(self.vfov_deg) / 2.0)
        return fy, fy, (w - 1) / 2.0, (h - 1) / 2.0

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "CameraModel":
        """sim_camera/camera.yaml (camera_model: {...}) as written by autodrive_camera_fit.py."""
        import yaml

        from control.types import resolve_path

        with open(resolve_path(path)) as f:
            fields_ = (yaml.safe_load(f) or {}).get("camera_model", {})
        return cls(**{**fields_, **overrides})

    def orbslam_settings(self, fps: float = 15.0) -> str:
        """ORB-SLAM3 RGB-D settings for this camera at the output resolution."""
        w, h = self.size
        fx, fy, cx, cy = self.intrinsics()
        return SETTINGS_TEMPLATE.format(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h, fps=fps,
                                        vfov=self.vfov_deg, scale=self.scale)

    def load_mask(self) -> Optional[np.ndarray]:
        """Car-body mask at the output resolution (bool), from mask_path."""
        if self.mask is None and self.mask_path:
            import cv2

            from control.types import resolve_path

            m = cv2.imread(resolve_path(self.mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(self.mask_path)
            self.mask = m > 0
        if self.mask is None:
            return None
        w, h = self.size
        if self.mask.shape != (h, w):
            import cv2
            return cv2.resize(self.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return self.mask


SETTINGS_TEMPLATE = """%YAML:1.0
# AutoDRIVE stock sim front camera, RGB-D with depth from the LiDAR (workers/lidar_depth.py).
# Written by scripts/autodrive_camera_fit.py from sim_camera/camera.yaml (vertical FOV {vfov:.1f} deg,
# images upscaled x{scale:g} from 192x108); do not edit by hand.
File.version: "1.0"

Camera.type: "PinHole"
Camera1.fx: {fx:.6f}
Camera1.fy: {fy:.6f}
Camera1.cx: {cx:.6f}
Camera1.cy: {cy:.6f}
Camera1.k1: 0.0
Camera1.k2: 0.0
Camera1.p1: 0.0
Camera1.p2: 0.0
Camera.width: {width}
Camera.height: {height}
Camera.fps: {fps:g}
# 0: BGR, 1: RGB
Camera.RGB: 1

RGBD.DepthMapFactor: 1.0
# close / far point threshold = Stereo.b x Stereo.ThDepth = 5 m
Stereo.b: 0.1
Stereo.ThDepth: 50.0

# dark, low-texture scene: lower FAST thresholds than the OAK-D settings
ORBextractor.nFeatures: 2000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 6
ORBextractor.iniThFAST: 7
ORBextractor.minThFAST: 3

Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
"""


class LidarDepth:
    """Precomputes the per-pixel rays once; __call__(ranges) -> depth (H, W) float32 m."""

    def __init__(self, cam: CameraModel, bearing_step_deg: float = 0.1) -> None:
        self.cam = cam
        w, h = cam.size
        fx, fy, cx, cy = cam.intrinsics()
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        xn, yn = (u - cx) / fx, (v - cy) / fy
        th = np.radians(cam.pitch_deg)
        # optical (xn, yn, 1) -> vehicle (fwd, left, up), camera pitched down by th
        self.d_fwd = np.cos(th) - yn * np.sin(th)
        self.d_left = -xn
        self.d_up = -np.sin(th) - yn * np.cos(th)
        self.d_h = np.hypot(self.d_fwd, self.d_left)
        self.bearing = np.arctan2(self.d_left, self.d_fwd)
        half = np.abs(self.bearing).max() + np.radians(1.0)
        step = np.radians(bearing_step_deg)
        self.grid = np.arange(-half, half + step, step)
        self.mask = cam.load_mask()

    def wall_distance(self, ranges: np.ndarray) -> np.ndarray:
        """Horizontal distance from the camera to the scan along each grid bearing (inf = none)."""
        cam = self.cam
        r = np.asarray(ranges, dtype=np.float64)[:1080]
        a = ANGLE_MIN + np.arange(len(r)) * ANGLE_INCREMENT
        ok = np.isfinite(r) & (r > 0.06) & (r < cam.max_range)
        px = cam.lidar_x + r * np.cos(a) - cam.cam_x       # scan points relative to the camera
        py = r * np.sin(a)
        # consecutive valid beams form wall segments; a jump > 0.3 m is a gap, not a wall
        p0 = np.column_stack([px[:-1], py[:-1]])
        p1 = np.column_stack([px[1:], py[1:]])
        seg = ok[:-1] & ok[1:] & (np.hypot(*(p1 - p0).T) < 0.3)
        p0, p1 = p0[seg], p1[seg]
        if not len(p0):
            return np.full(len(self.grid), np.inf)
        d = np.column_stack([np.cos(self.grid), np.sin(self.grid)])     # (B, 2)
        e = p1 - p0                                                       # (S, 2)
        den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (p0[None, :, 0] * e[None, :, 1] - p0[None, :, 1] * e[None, :, 0]) / den
            s = (p0[None, :, 0] * d[:, None, 1] - p0[None, :, 1] * d[:, None, 0]) / den
        t = np.where((t > 0) & (s >= -1e-6) & (s <= 1 + 1e-6), t, np.inf)
        return t.min(axis=1)

    def __call__(self, ranges: np.ndarray) -> np.ndarray:
        cam = self.cam
        D = np.interp(self.bearing, self.grid, self.wall_distance(ranges))   # per pixel, horizontal m
        with np.errstate(divide="ignore", invalid="ignore"):
            t_wall = D / self.d_h                                 # optical depth where the ray meets the wall
            z_wall = cam.cam_height + t_wall * self.d_up          # height there
            t_floor = np.where(self.d_up < -1e-6, cam.cam_height / -self.d_up, np.inf)
        depth = np.zeros(self.d_up.shape, dtype=np.float32)
        floor = t_floor < t_wall
        wall = ~floor & np.isfinite(t_wall) & (z_wall >= 0.0) & (z_wall <= cam.wall_height)
        depth[floor] = t_floor[floor]
        depth[wall] = t_wall[wall]
        depth[depth > cam.max_range] = 0.0
        if self.mask is not None:
            depth[self.mask] = 0.0
        return depth


def mask_rgb(rgb: np.ndarray, mask: Optional[np.ndarray], clahe: bool = True) -> np.ndarray:
    """Hide the car body from ORB without creating an edge around it: each column under
    the mask continues the scene value just above it, smoothed across columns (a flat
    fill would put strong corners on the mask outline, which move with the camera). With
    clahe, local contrast is equalised so the dark asphalt and walls of the stock sim
    show texture to FAST."""
    import cv2

    out = rgb.copy()
    if clahe:     # before the fill, so the fill matches the equalised scene above it
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(lab[..., 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if mask is not None and mask.any():
        h, w = mask.shape
        top = np.where(mask.any(axis=0), mask.argmax(axis=0), h)        # first masked row per column
        rows = np.clip(top[None, :] - 1 - np.arange(3)[:, None], 0, h - 1)
        edge = out[rows, np.arange(w)[None, :]].astype(np.float32).mean(axis=0)   # (w, 3)
        edge = cv2.GaussianBlur(edge[None], (0, 0), sigmaX=6)[0]
        fill = np.broadcast_to(edge[None], out.shape).astype(out.dtype)
        out[mask] = fill[mask]
    return out
