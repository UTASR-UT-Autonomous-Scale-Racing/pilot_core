"""SegScanWorker: a segmentation mask + its frame's (absent) depth -> a LiDAR-format scan, with
the track edge placed on the floor; no LiDAR anywhere."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np

from control.depth_ftg import Intrinsics
from control.lidar_gap import ANGLE_INCREMENT, ANGLE_MIN
from shared_buffer import FrameBuffer, LidarBuffer
from shared_buffer.segment_buffer import SegmentationBuffer
from workers.seg_scan_worker import SegScanWorker

SETTINGS = "sim_camera/orbslam_settings.yaml"          # 384 x 216, the sim camera upscaled


def test_edge_on_the_floor_without_depth():
    K = Intrinsics.from_orbslam_settings(SETTINGS)
    h, w = 216, 384
    pitch, height = 0.1833, 0.17
    # floor up to the row where the floor is 2.0 m ahead; nothing drivable above it
    yn = (np.tan(np.arctan2(height, 2.0) - pitch))     # ray below the optical axis
    edge_row = int(round(K.cy + yn * K.fy))
    mask = np.zeros((h, w), np.uint8)
    mask[edge_row:, :] = 1
    u = uuid.uuid4().hex[:8]
    sb = SegmentationBuffer(slots=4, name=f"ss_s_{u}", create=True, max_payload=h * w + 64)
    fb = FrameBuffer(slots=4, name=f"ss_f_{u}", max_payload=16 + h * w * 7)
    lb = LidarBuffer(slots=4, name=f"ss_l_{u}")
    w_ = SegScanWorker(name="seg_scan", segment_buffer=sb, frame_buffer=fb, output_buffer=lb,
                       seg_scan={"camera_settings": SETTINGS, "camera_pitch": pitch, "camera_height": height})
    w_._logger = MagicMock()
    w_.setup()
    try:
        fb.write(np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.float32), ts=1.0)   # no depth
        sb.write(mask, ts=1.0)
        w_.step()
        _, ranges = lb.read_latest()
        ahead = int(round((0.0 - ANGLE_MIN) / ANGLE_INCREMENT))
        assert abs(float(ranges[ahead]) - 2.0) < 0.1
        assert np.isfinite(ranges).sum() > 50              # the edge across the camera's view
    finally:
        for b in (sb, fb, lb):
            b.unlink()
