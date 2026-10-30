"""
SegScanWorker: segmentation mask (VisionWorker) + the depth of its frame -> a 2-D scan in the
LiDAR's format (control/seg_scan.py), for a pipeline without a LiDAR: LidarProjectionWorker
turns these scans into the wall point cloud the mapping lap's boundaries come from.

The scan's origin is seg_scan.origin_forward behind the camera; give LidarProjectionWorker that
origin's mount in the vehicle frame as its scan_offset (and as scan_origin here). With no depth
(the sim without LiDAR) each track edge is placed where its pixel's ray meets the floor
(camera_height, camera_pitch).

A camera sees ~60 deg, a LiDAR 270: with a pose_buffer and memory_s, the edges seen in the last
memory_s seconds are kept (world frame) and fill the beams outside the current view, so a wall
that has left the image (the inside of a hairpin) still bounds the scan the gap follower drives on.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .base_worker import BaseWorker


class SegScanWorker(BaseWorker):
    """Config:
      segment_buffer   SegmentationBuffer (VisionWorker)
      frame_buffer     FrameBuffer the masks were computed from (for their depth)
      output_buffer    LidarBuffer: 1080 ranges, -135..+135 deg, as the LiDAR's
      seg_scan         camera_settings + control.seg_scan.SegScanConfig fields
      pose_buffer      PoseBuffer (Twv), optional: with memory_s, remembered edges fill unseen beams
      memory_s         s of edges kept, default 0 (off)
      scan_origin      [x, y] of the scan origin in the vehicle frame (the scan_offset), default [0, 0]
    """

    def __init__(self, name: str = "seg_scan", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._last_seq = 0
        self._scans = 0
        self._missed = 0
        self._pose_seq = 0
        self._poses: list = []
        self._memory: list = []            # (t, (N, 2) world-frame edge points)

    def setup(self) -> None:
        from control.depth_ftg import Intrinsics
        from control.seg_scan import SegScanConfig

        self._seg_buf = self.config["segment_buffer"]
        self._frame_buf = self.config["frame_buffer"]
        self._out = self.config["output_buffer"]
        cfg = dict(self.config.get("seg_scan", {}))
        self._K = Intrinsics.from_orbslam_settings(cfg.pop("camera_settings", "orbslam_oakd_rgbd.yaml"))
        self._cfg = SegScanConfig(**cfg)
        self._pose_buf = self.config.get("pose_buffer")
        self._memory_s = float(self.config.get("memory_s", 0.0))
        self._origin = np.asarray(self.config.get("scan_origin", [0.0, 0.0]), dtype=np.float64)[:2]
        from control.lidar_gap import ANGLE_INCREMENT, ANGLE_MIN

        self._ang = ANGLE_MIN + np.arange(1080) * ANGLE_INCREMENT
        self._a0, self._da = ANGLE_MIN, ANGLE_INCREMENT
        self.logger.info(f"SegScanWorker ready (segmentation edge -> scan; pitch {self._cfg.camera_pitch:.3f} rad, "
                         f"height {self._cfg.camera_height:.3f} m, floor fallback {'on' if self._cfg.floor_fallback else 'off'})")

    def step(self) -> None:
        from control.seg_scan import seg_edge_scan

        seg = self._seg_buf.read_latest()
        if seg is None or seg[0].seq_no <= self._last_seq:
            time.sleep(0.005)
            return
        header, mask = seg
        self._last_seq = header.seq_no
        # VisionWorker stamps a mask with its frame's capture time: find that frame's depth
        frames = [self._frame_buf.read_latest()]
        if frames[0] is not None and frames[0][0].produced_ts != header.produced_ts:
            frames = self._frame_buf.read_since(max(0, frames[0][0].seq_no - 4))
        depth = next((f[2] for f in frames if f is not None and f[0].produced_ts == header.produced_ts), None)
        if depth is None:
            self._missed += 1                 # its frame has left the ring
            return
        ranges = seg_edge_scan(mask, depth, self._K, self._cfg)
        if self._memory_s > 0 and self._pose_buf is not None:
            ranges = self._with_memory(ranges, header.produced_ts)
        self._out.write(ranges, ts=header.produced_ts)
        self._scans += 1
        if self._scans % 200 == 0:
            self.logger.info(f"{self._scans} scans ({self._missed} masks without their frame)")

    def _pose_at(self, t: float):
        for header, pose, *_ in self._pose_buf.read_since(self._pose_seq):
            self._pose_seq = header.seq_no
            self._poses.append((header.produced_ts, np.asarray(pose, dtype=np.float64)))
        if not self._poses:
            return None
        while len(self._poses) > 1 and self._poses[0][0] < self._poses[-1][0] - 1.0:
            self._poses.pop(0)
        return min(self._poses, key=lambda p: abs(p[0] - t))[1]

    def _with_memory(self, ranges: np.ndarray, t: float) -> np.ndarray:
        """Remember this scan's edges (world frame); fill its unknown (NaN) beams from the
        edges remembered from the last memory_s seconds."""
        T = self._pose_at(t)
        if T is None:
            return ranges
        R, p = T[:2, :2], T[:2, 3]
        o = p + R @ self._origin                                   # scan origin, world
        ok = np.isfinite(ranges) & (ranges > 0)
        pts = np.column_stack([ranges[ok] * np.cos(self._ang[ok]), ranges[ok] * np.sin(self._ang[ok])])
        self._memory.append((t, pts @ R.T + o))
        self._memory = [(tm, q) for tm, q in self._memory if t - tm <= self._memory_s]
        unknown = np.isnan(ranges)
        if not unknown.any() or len(self._memory) < 2:
            return ranges
        q = np.vstack([q for _, q in self._memory[:-1]]) - o
        q = q @ R                                                   # world -> scan frame
        r = np.hypot(q[:, 0], q[:, 1])
        b = np.round((np.arctan2(q[:, 1], q[:, 0]) - self._a0) / self._da).astype(int)
        keep = (b >= 0) & (b < len(ranges)) & (r > 0.05)
        best = np.full(len(ranges), np.inf)
        np.minimum.at(best, b[keep], r[keep])
        fill = unknown & np.isfinite(best)
        out = ranges.copy()
        out[fill] = best[fill]
        return out.astype(np.float32)
