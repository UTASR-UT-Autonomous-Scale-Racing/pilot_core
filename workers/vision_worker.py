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

import numpy as np

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
        self._debug = False
        self._debug_dir = "/tmp/pilot_debug/segments"
        self._frame_count = 0

    def setup(self) -> None:
        import os

        self._frame_buf = self.config["input_buffer"]
        self._seg_buf = self.config["output_buffer"]
        self._debug = self.config.get("debug", False)

        # Load the segmentation model from config
        model_config_path = self.config["model_config"]
        from utils import load_model_from_config

        self._model, class_name = load_model_from_config(model_config_path)
        self.logger.info(f"Loaded segmentation model: {class_name}")

        device = self.config.get("device", "cpu")

        if self._debug:
            os.makedirs(self._debug_dir, exist_ok=True)
            self.logger.info(
                f"Debug mode ON — exporting segment frames to {self._debug_dir}"
            )

        self.logger.info(f"VisionWorker ready (device={device})")

    def step(self) -> None:
        result = self._frame_buf.read_latest()
        if result is None:
            time.sleep(0.001)
            return

        header, rgb, depth = result

        # Skip if we already processed this frame
        if header.seq_no <= self._last_seq:
            time.sleep(0.001)
            return
        self._last_seq = header.seq_no

        # Run segmentation — interface: model.get_mask(depth_frame, rgb_frame)
        mask = self._model.get_mask(depth, rgb)

        if mask is not None:
            self._seg_buf.write(mask)

            self._frame_count += 1

            # Debug: save segmentation overlay every 30 frames (~1/sec @ 30 fps)
            if self._debug and self._frame_count % 30 == 1:
                self._save_debug_frame(rgb, mask)

    def _save_debug_frame(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        """Save an RGB + segmentation overlay to the debug directory."""
        try:
            import cv2

            # Colour the mask as a semi-transparent overlay
            overlay = rgb.copy()
            # Map mask classes to colours (green=drivable, red=boundary, etc.)
            colour_map = {
                1: (0, 200, 0),     # class 1 → green
                2: (0, 0, 200),     # class 2 → red
                3: (200, 200, 0),   # class 3 → cyan
            }
            for cls_id, colour in colour_map.items():
                overlay[mask == cls_id] = colour

            blended = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)

            path = f"{self._debug_dir}/seg_{self._frame_count:06d}.jpg"
            cv2.imwrite(path, blended)
            self.logger.debug(f"Debug segment frame saved → {path}")
        except Exception as exc:
            self.logger.warning(f"Failed to save debug frame: {exc}")

    def teardown(self) -> None:
        if self._debug:
            self.logger.info(
                f"Debug frames saved to {self._debug_dir} "
                f"({self._frame_count} total frames processed)"
            )
        self.logger.info("VisionWorker shutting down")
