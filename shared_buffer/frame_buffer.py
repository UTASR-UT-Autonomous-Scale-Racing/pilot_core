"""
RGB + depth frame pairs.

Payload (contiguous):
    [4 bytes]  uint32  rgb_height
    [4 bytes]  uint32  rgb_width
    [4 bytes]  uint32  depth_height
    [4 bytes]  uint32  depth_width
    [H*W*3]    uint8   RGB, row-major, RGB channel order
    [H*W*4]    float32 depth in metres (what the camera workers write)
                or
    [H*W*2]    uint16  raw depth, e.g. mm

Depth dtype isn't stored; the reader works it out from what's left of the payload.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

# slots are sized for 640x480 RGB (uint8) + 640x480 float32 depth
_DEFAULT_RGB_SHAPE = (480, 640, 3)
_DEFAULT_DEPTH_SHAPE = (480, 640)

_META_FMT = "<IIII"  # 16 bytes
_META_SIZE = struct.calcsize(_META_FMT)

_MAX_PAYLOAD = (
    _META_SIZE
    + _DEFAULT_RGB_SHAPE[0] * _DEFAULT_RGB_SHAPE[1] * _DEFAULT_RGB_SHAPE[2]
    + _DEFAULT_DEPTH_SHAPE[0] * _DEFAULT_DEPTH_SHAPE[1] * 4
)


class FrameBuffer:
    """Camera frame pairs (RGB + depth).

    Pass a bigger max_payload if you run the camera above 640x480.
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

    def write(self, rgb: np.ndarray, depth: np.ndarray, ts: Optional[float] = None) -> SlotHeader:
        """rgb: (H, W, 3) uint8. depth: (H, W) float32 metres, or uint16."""
        rh, rw = rgb.shape[:2]
        dh, dw = depth.shape[:2]

        meta = struct.pack(_META_FMT, rh, rw, dh, dw)
        payload = meta + rgb.tobytes() + depth.tobytes()
        return self._rb.write(payload, ts)

    def latest_seq(self) -> int:
        """Highest published seq_no, without copying any data."""
        return self._rb.latest_seq()

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray, np.ndarray]]:
        """Newest (header, rgb, depth), or None if nothing's been written.

        depth comes back in whatever dtype it was written as.
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
        # 4 B/px left over -> float32, otherwise uint16
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

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
