"""
VisionWorker — reads frames from FrameBuffer, runs segmentation model
inference, and writes masks into SegmentationBuffer.

The segmentation model class (from ``pilot_vision``) is loaded dynamically
from a YAML config file — the same ``load_model_from_config`` mechanism
used by the existing codebase.

On Jetson Orin this will use CUDA / TensorRT for inference.
"""

from __future__ import annotations

import time
from typing import Any


from .base_worker import BaseWorker
from shared_buffer import FrameBuffer, SegmentationBuffer


class VisionWorker(BaseWorker):
    """Worker process for segmentation inference.

    Config kwargs
    -------------
    input_buffer : FrameBuffer
        Shared FrameBuffer to read RGB+Depth from.
    output_buffer : SegmentationBuffer
        Shared SegmentationBuffer to write masks into.
    model_config : str
        Path to the segmentation model YAML config file.
    device : str
        CUDA device string (e.g. ``"cuda:0"``).  Ignored on CPU-only machines.
    use_tensorrt : bool
        If True, attempt to load a TensorRT engine (default False).
    """

    def __init__(self, name: str = "vision", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._frame_buf: FrameBuffer | None = None
        self._seg_buf: SegmentationBuffer | None = None
        self._model = None
        self._last_seq = 0
        self._waiting_logged = False

    def setup(self) -> None:
        self._frame_buf = self.config["input_buffer"]
        self._seg_buf = self.config["output_buffer"]

        # Load the segmentation model from config
        model_config_path = self.config["model_config"]
        from utils import load_model_from_config

        self._model, class_name = load_model_from_config(model_config_path)
        self.logger.info(f"Loaded segmentation model: {class_name}")

        device = self.config.get("device", "cpu")
        self.logger.info(f"VisionWorker ready (device={device})")

    def step(self) -> None:
        result = self._frame_buf.read_latest()
        if result is None:
            if not self._waiting_logged:
                self.logger.debug("frame_buf empty — waiting for frames")
                self._waiting_logged = True
            time.sleep(0.001)
            return

        header, rgb, depth = result

        # Skip if we already processed this frame
        if header.seq_no <= self._last_seq:
            time.sleep(0.001)
            return
        self._last_seq = header.seq_no

        # self.logger.debug(f"Running get_mask on frame seq={header.seq_no} rgb={rgb.shape}")

        # Run segmentation — interface: model.get_mask(depth_frame, rgb_frame)
        mask = self._model.get_mask(depth, rgb)

        if mask is not None:
            self._seg_buf.write(mask)
        #     self.logger.debug(f"Wrote mask to seg_buf shape={mask.shape} max={mask.max():.3f}")
        # else:
        #     self.logger.warning("get_mask returned None")

    def teardown(self) -> None:
        self.logger.info("VisionWorker shutting down")
