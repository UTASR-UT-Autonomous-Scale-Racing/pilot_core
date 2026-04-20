"""
VisionWorker — reads frames from FrameBuffer, runs segmentation model
inference, and writes masks into SegmentationBuffer.

The model class (from ``pilot_vision``) is loaded from a YAML config via
:func:`utils.load_model_from_config` and must provide
``get_mask(depth, rgb) -> mask | None``.
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
        Device string (e.g. ``"cuda:0"``); currently only logged — the model
        config decides where inference runs.
    use_tensorrt : bool
        Currently unused (default False).
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

        if header.seq_no <= self._last_seq:
            time.sleep(0.001)
            return
        self._last_seq = header.seq_no

        # Model interface: get_mask(depth_frame, rgb_frame)
        mask = self._model.get_mask(depth, rgb)

        if mask is not None:
            self._seg_buf.write(mask)

    def teardown(self) -> None:
        self.logger.info("VisionWorker shutting down")
