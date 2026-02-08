"""
FrameBuffer — typed wrapper for RGB + Depth frame pairs.

Payload layout (contiguous bytes):
    [4 bytes]  uint32  rgb_height
    [4 bytes]  uint32  rgb_width
    [4 bytes]  uint32  depth_height
    [4 bytes]  uint32  depth_width
    [H*W*3]    uint8   RGB data   (row-major, BGR as from OpenCV)
    [H*W*2]    uint16  Depth data (millimetres or raw disparity)
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

# Default geometry — 640×480 RGB (3ch uint8) + 640×480 depth (float32)
_DEFAULT_RGB_SHAPE = (480, 640, 3)
_DEFAULT_DEPTH_SHAPE = (480, 640)

_META_FMT = "<IIII"  # 4 × uint32 = 16 bytes
_META_SIZE = struct.calcsize(_META_FMT)

# Max payload = meta + rgb (uint8) + depth (float32, 4 bytes per pixel)
_MAX_PAYLOAD = (
    _META_SIZE
    + _DEFAULT_RGB_SHAPE[0] * _DEFAULT_RGB_SHAPE[1] * _DEFAULT_RGB_SHAPE[2]
    + _DEFAULT_DEPTH_SHAPE[0] * _DEFAULT_DEPTH_SHAPE[1] * 4
)


class FrameBuffer:
    """High-level buffer for camera frame pairs (RGB + Depth).

    Parameters
    ----------
    slots : int
        Number of ring-buffer slots (default 8).
    name : str
        Shared-memory block name.
    create : bool
        Pass *True* from the orchestrator, *False* from workers.
    max_payload : int
        Override the maximum per-slot payload size if a different resolution
        is needed.
    """

    def __init__(
        self,
        slots: int = 8,
        name: str = "frame_buf",
        create: bool = True,
        max_payload: int = _MAX_PAYLOAD,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=max_payload,
            dtype_tag=DTypeTag.FRAME,
            create=create,
        )

    #  write ─

    def write(self, rgb: np.ndarray, depth: np.ndarray) -> SlotHeader:
        """Serialise and write an RGB + depth frame pair.

        Parameters
        ----------
        rgb : np.ndarray
            Shape ``(H, W, 3)``, dtype ``uint8``.
        depth : np.ndarray
            Shape ``(H, W)``, dtype ``uint16`` or ``float32``.
            If float32 the raw bytes are stored as-is; the reader must know.
        """
        rh, rw = rgb.shape[:2]
        dh, dw = depth.shape[:2]

        meta = struct.pack(_META_FMT, rh, rw, dh, dw)
        payload = meta + rgb.tobytes() + depth.tobytes()
        return self._rb.write(payload)

    #  read 

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray, np.ndarray]]:
        """Read the most recent frame pair.

        Returns
        -------
        (header, rgb, depth) or *None* if no frame has been written.
        ``rgb`` is ``(H, W, 3) uint8``; ``depth`` is ``(H, W) uint16``.
        """
        result = self._rb.read_latest()
        if result is None:
            return None

        header, payload = result
        rh, rw, dh, dw = struct.unpack_from(_META_FMT, payload, 0)

        rgb_size = rh * rw * 3
        rgb = np.frombuffer(payload, dtype=np.uint8, count=rgb_size, offset=_META_SIZE).reshape(
            (rh, rw, 3)
        )

        depth_offset = _META_SIZE + rgb_size
        # Infer depth dtype from remaining size
        remaining = len(payload) - depth_offset
        depth_pixels = dh * dw
        if remaining == depth_pixels * 4:
            depth = np.frombuffer(payload, dtype=np.float32, count=depth_pixels, offset=depth_offset).reshape(
                (dh, dw)
            )
        else:
            depth = np.frombuffer(payload, dtype=np.uint16, count=depth_pixels, offset=depth_offset).reshape(
                (dh, dw)
            )

        return header, rgb.copy(), depth.copy()

    #  lifecycle ─

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
