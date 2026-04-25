"""
ControlBuffer: normalised steering + throttle.

Payload:
    [4 bytes]  float32  steering  (0..1, 0.5 = straight, below = left)
    [4 bytes]  float32  throttle  (0..1, 0.5 = neutral, below = reverse)

Both map linearly onto PWM pulse widths, see control/pwm.py.
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_CTRL_FMT = "<ff"
_PAYLOAD_SIZE = struct.calcsize(_CTRL_FMT)  # 8 bytes


class ControlBuffer:
    """Normalised control outputs. create=True from the orchestrator,
    False from workers."""

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

    def write(self, steering: float, throttle: float) -> SlotHeader:
        payload = struct.pack(_CTRL_FMT, steering, throttle)
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, float, float]]:
        """-> (header, steering, throttle), or None."""
        result = self._rb.read_latest()
        if result is None:
            return None
        header, payload = result
        steering, throttle = struct.unpack(_CTRL_FMT, payload[:_PAYLOAD_SIZE])
        return header, steering, throttle

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
