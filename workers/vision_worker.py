"""
FrameBuffer -> drivable-track segmentation -> SegmentationBuffer.

Output contract: a uint8 mask at frame resolution, 1 = drivable track and
0 = everything else (walls, off-track floor, background).

Segmenter (``vision.model`` in the config):

* ``pilot_vision`` (default): a model from the pilot_vision submodule, loaded
  from ``segmentation_config_path`` by utils.load_model_from_config
  (``get_mask(depth, rgb)``); its output is converted to the mask contract.
* ``onnx``: a network trained with scripts/train_vision.py (``vision.onnx_model``).
* ``ground_plane``: training-free RANSAC floor + free space from depth.
"""

from __future__ import annotations

import time
from typing import Any

from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, SegmentationBuffer

MODES = ("pilot_vision", "onnx", "ground_plane")


class VisionWorker(BaseWorker):
    """Segmentation inference worker.

    Config kwargs:
      input_buffer     FrameBuffer (RGB + depth in metres)
      output_buffer    SegmentationBuffer
      mode             one of MODES (default ``pilot_vision``)
      model_config     pilot_vision model YAML (mode ``pilot_vision``)
      onnx_model       exported network (mode ``onnx``)
      threshold        track-probability threshold override (mode ``onnx``)
      camera_settings  ORB-SLAM3 settings file with the intrinsics (mode ``ground_plane``)
      ground_plane     vision.segmenters.GroundPlaneConfig overrides
    """

    def __init__(self, name: str = "vision", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._seg_buf: SegmentationBuffer | None = None
        self._segmenter = None
        self._legacy = False
        self._last_seq = 0
        self._waiting_logged = False

    def setup(self) -> None:
        self._frame_buf = self.config["input_buffer"]
        self._seg_buf = self.config["output_buffer"]
        mode = self.config.get("mode", "pilot_vision")

        if mode == "pilot_vision":
            from utils import load_model_from_config

            self._segmenter, class_name = load_model_from_config(self.config["model_config"])
            self._legacy = True
            self.logger.info(f"VisionWorker ready — pilot_vision model {class_name}")
        elif mode == "onnx":
            from vision.segmenters import OnnxTrackSegmenter

            path = self.config.get("onnx_model", "track_seg.onnx")
            self._segmenter = OnnxTrackSegmenter(path, self.config.get("threshold"))
            self.logger.info(f"VisionWorker ready — {self._segmenter.meta.get('model')} ({path})")
        elif mode == "ground_plane":
            from control.depth_ftg import Intrinsics
            from vision.segmenters import GroundPlaneConfig, GroundPlaneSegmenter

            K = Intrinsics.from_orbslam_settings(self.config.get("camera_settings", "orbslam_oakd_rgbd.yaml"))
            self._segmenter = GroundPlaneSegmenter(K, GroundPlaneConfig(**self.config.get("ground_plane", {})))
            self.logger.info("VisionWorker ready — ground_plane (RANSAC floor + free space)")
        else:
            raise ValueError(f"Unknown vision mode {mode!r}; options: {', '.join(MODES)}")

    def step(self) -> None:
        result = self._frame_buf.read_latest()
        if result is None:
            if not self._waiting_logged:
                self.logger.debug("frame_buf empty — waiting for frames")
                self._waiting_logged = True
            time.sleep(0.001)
            return

        header, rgb, depth = result
        if header.seq_no <= self._last_seq:
            time.sleep(0.001)
            return
        self._last_seq = header.seq_no

        mask = self._segmenter.get_mask(depth, rgb)      # note the arg order: depth first
        if self._legacy:
            from vision.segmenters import to_track_mask

            mask = to_track_mask(mask, depth.shape)
        if mask is not None:
            self._seg_buf.write(mask)

    def teardown(self) -> None:
        self.logger.info("VisionWorker shutting down")
