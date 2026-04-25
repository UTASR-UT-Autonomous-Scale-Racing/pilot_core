"""
Segmentation masks.

Payload:
    [4 bytes]  uint32  height
    [4 bytes]  uint32  width
    [H*W*4]    float32 mask data (row-major)
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_DEFAULT_SHAPE = (480, 640)
_META_FMT = "<II"
_META_SIZE = struct.calcsize(_META_FMT)
_MAX_PAYLOAD = _META_SIZE + _DEFAULT_SHAPE[0] * _DEFAULT_SHAPE[1] * 4  # float32


class SegmentationBuffer:
    """Masks up to 640x480 by default; bump max_payload for bigger ones."""

    def __init__(
        self,
        slots: int = 4,
        name: str = "seg_buf",
        create: bool = True,
        max_payload: int = _MAX_PAYLOAD,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=max_payload,
            dtype_tag=DTypeTag.SEGMENTATION,
            create=create,
        )

    def write(self, mask: np.ndarray, ts: Optional[float] = None) -> SlotHeader:
        """Any numeric (H, W) mask; stored as float32."""
        mask_f32 = mask.astype(np.float32)
        h, w = mask_f32.shape[:2]
        meta = struct.pack(_META_FMT, h, w)
        payload = meta + mask_f32.tobytes()
        return self._rb.write(payload, ts)

    def latest_seq(self) -> int:
        """Highest published seq_no, without copying any data."""
        return self._rb.latest_seq()

    def read_latest(self) -> Optional[Tuple[SlotHeader, np.ndarray]]:
        result = self._rb.read_latest()
        if result is None:
            return None
        header, payload = result
        h, w = struct.unpack_from(_META_FMT, payload, 0)
        mask = np.frombuffer(payload, dtype=np.float32, count=h * w, offset=_META_SIZE).reshape(
            (h, w)
        ).copy()
        return header, mask

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
