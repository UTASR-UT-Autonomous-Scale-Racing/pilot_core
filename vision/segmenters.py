"""
Runtime drivable-track segmenters used by VisionWorker.

All segmenters implement ``get_mask(depth, rgb) -> uint8 (H, W)`` with
1 = drivable track and 0 = everything else, at the camera frame's resolution.

* :class:`OnnxTrackSegmenter` runs a network trained with
  ``scripts/train_vision.py`` (any of ``vision.models.MODELS``) through
  onnxruntime, so the car does not need torch.
* :class:`GroundPlaneSegmenter` needs no training: it fits the ground plane to
  the depth point cloud with RANSAC (M. A. Fischler, R. C. Bolles, "Random
  Sample Consensus", Comm. ACM 24(6), 1981) and marks as drivable the floor
  pixels between the car and the first obstacle in each image column, the
  free-space idea of the Stixel World (H. Badino, U. Franke, D. Pfeiffer,
  "The Stixel World", DAGM 2009).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np


class OnnxTrackSegmenter:
    """Exported network (``<model>.onnx`` + ``<model>.json`` sidecar)."""

    def __init__(self, onnx_path: str, threshold: float | None = None) -> None:
        import onnxruntime as ort

        from control.types import resolve_path

        onnx_path = resolve_path(onnx_path)
        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            self.meta = json.load(f)
        self.input_h, self.input_w = self.meta["input_size"]
        self.threshold = float(threshold if threshold is not None else self.meta.get("threshold", 0.5))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "1") or 1)
        self.session = ort.InferenceSession(onnx_path, opts, providers=ort.get_available_providers())
        self._input = self.session.get_inputs()[0].name

    def probability(self, rgb: np.ndarray) -> np.ndarray:
        """Track probability at the network's input size ``(input_h, input_w)``."""
        x = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        return self.session.run(None, {self._input: x})[0][0, 0]

    def get_mask(self, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        mask = (self.probability(rgb) > self.threshold).astype(np.uint8)
        h, w = rgb.shape[:2]
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


@dataclass
class GroundPlaneConfig:
    max_range: float = 5.0            # m, ignore depth beyond this
    pixel_step: int = 4               # subsample the depth image
    plane_tolerance: float = 0.03     # m, |distance to plane| counted as floor
    obstacle_height: float = 0.04     # m above the plane that counts as an obstacle
    ransac_iterations: int = 60
    min_floor_fraction: float = 0.05  # of sampled pixels; below this no plane is trusted
    # camera height above an accepted floor, m. Rejects planes above the camera
    # (ceiling, table top) and an upside-down camera
    camera_height_range: tuple = (0.03, 1.0)


class GroundPlaneSegmenter:
    """Training-free drivable area from depth: RANSAC floor + per-column free space."""

    def __init__(self, K, cfg: GroundPlaneConfig | None = None, seed: int = 0) -> None:
        self.K = K
        self.cfg = cfg or GroundPlaneConfig()
        self._rng = np.random.default_rng(seed)
        self.plane: np.ndarray | None = None     # (n, d): n·p + d = 0, n points up (towards -y)

    def _points(self, depth: np.ndarray):
        s = self.cfg.pixel_step
        v, u = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
        z = depth[::s, ::s].astype(np.float64)
        valid = (z > 0.05) & (z < self.cfg.max_range)
        x = (u - self.K.cx) / self.K.fx * z
        y = (v - self.K.cy) / self.K.fy * z
        return np.stack([x, y, z], axis=-1), valid

    def fit_plane(self, pts: np.ndarray) -> np.ndarray | None:
        """RANSAC plane on candidate floor points (lower image half, camera frame y down)."""
        c = self.cfg
        if len(pts) < 3:
            return None
        best, best_inliers = None, 0
        for _ in range(c.ransac_iterations):
            p = pts[self._rng.choice(len(pts), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm
            if n[1] > 0:               # make the normal point up (camera y is down)
                n = -n
            if abs(n[1]) < 0.7:        # reject planes far from horizontal
                continue
            d = -n @ p[0]
            inliers = int((np.abs(pts @ n + d) < c.plane_tolerance).sum())
            if inliers > best_inliers:
                best, best_inliers = np.append(n, d), inliers
        if best is None or best_inliers < c.min_floor_fraction * len(pts):
            return None
        # Least-squares refit on the inliers
        inl = pts[np.abs(pts @ best[:3] + best[3]) < c.plane_tolerance]
        centroid = inl.mean(axis=0)
        n = np.linalg.svd(inl - centroid, full_matrices=False)[2][-1]
        n = -n if n[1] > 0 else n
        d = -n @ centroid                                       # camera height above the plane
        if not c.camera_height_range[0] <= d <= c.camera_height_range[1]:
            return None
        return np.append(n, d)

    def get_mask(self, depth: np.ndarray, rgb: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        pts, valid = self._points(depth)
        rows = pts.shape[0]
        lower = valid.copy()
        lower[: rows // 2] = False
        plane = self.fit_plane(pts[lower])
        if plane is not None:
            self.plane = plane                                   # else keep the last good plane
        h, w = depth.shape
        if self.plane is None:
            return np.zeros((h, w), np.uint8)

        height = pts @ self.plane[:3] + self.plane[3]           # m above the floor
        floor = valid & (np.abs(height) < c.plane_tolerance)
        obstacle = valid & (height > c.obstacle_height)

        # Free space: scan each column from the bottom up to the first obstacle
        drivable = np.zeros_like(floor)
        blocked = np.zeros(floor.shape[1], dtype=bool)
        for r in range(rows - 1, -1, -1):
            blocked |= obstacle[r]
            drivable[r] = floor[r] & ~blocked
        small = drivable.astype(np.uint8)
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

