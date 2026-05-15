"""
Segmentation masks. Payload: uint32 height, uint32 width, H*W uint8 (row-major,
values clipped to 0..255). decode() also reads schema-v1 float32 masks (by size).
read_latest() -> (header, (H, W) uint8); decode(payload) -> (H, W) uint8.
"""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_DEFAULT_SHAPE = (480, 640)
_META_FMT = "<II"
_META_SIZE = struct.calcsize(_META_FMT)
_MAX_PAYLOAD = _META_SIZE + _DEFAULT_SHAPE[0] * _DEFAULT_SHAPE[1]  # uint8


def max_payload_for(height: int, width: int) -> int:
    """Slot payload size needed for an (height, width) mask."""
    return _META_SIZE + height * width


class SegmentationBuffer(TypedRingBuffer):
    """Masks up to 640x480 by default; bump max_payload for bigger ones
    (see max_payload_for)."""

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
        """Any numeric (H, W) mask; stored as uint8."""
        mask = np.asarray(mask)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        h, w = mask.shape[:2]
        payload = struct.pack(_META_FMT, h, w) + np.ascontiguousarray(mask).tobytes()
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> np.ndarray:
        """payload -> (H, W) uint8 mask"""
        h, w = struct.unpack_from(_META_FMT, payload, 0)
        n = h * w
        if len(payload) - _META_SIZE >= n * 4 and n > 0:
            # schema v1 recording: float32 mask
            legacy = np.frombuffer(payload, dtype=np.float32, count=n, offset=_META_SIZE)
            return np.clip(legacy, 0, 255).astype(np.uint8).reshape((h, w))
        return np.frombuffer(payload, dtype=np.uint8, count=n, offset=_META_SIZE).reshape((h, w)).copy()
