"""
ControlBuffer: normalised steering + throttle.

Payload:
    [4 bytes]  float32  steering  (0..1, 0.5 = straight, below = left)
    [4 bytes]  float32  throttle  (0..1, 0.5 = neutral, below = reverse)

Both map linearly onto PWM pulse widths, see control/pwm.py.
read_latest() -> (header, steering, throttle); decode(payload) -> (steering, throttle).
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_CTRL_FMT = "<ff"
_PAYLOAD_SIZE = struct.calcsize(_CTRL_FMT)  # 8 bytes


class ControlBuffer(TypedRingBuffer):
    """Normalised control outputs. create=True from the orchestrator,
    False from workers."""

    _MULTI = True

    def __init__(
        self,
        slots: int = 8,
        name: str = "ctrl_buf",
        create: bool = True,
    ) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_PAYLOAD_SIZE,
            dtype_tag=DTypeTag.CONTROL,
            create=create,
        )

    def write(self, steering: float, throttle: float, ts: Optional[float] = None) -> SlotHeader:
        payload = struct.pack(_CTRL_FMT, steering, throttle)
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> Tuple[float, float]:
        """payload -> (steering, throttle)"""
        return struct.unpack_from(_CTRL_FMT, payload, 0)
