"""
ControlBuffer — typed wrapper for normalised steering + throttle.

Payload layout:
    [4 bytes]  float32  steering  (normalised 0.0–1.0)
    [4 bytes]  float32  throttle  (normalised 0.0–1.0)
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

import numpy as np

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_CTRL_FMT = "<ff"
_PAYLOAD_SIZE = struct.calcsize(_CTRL_FMT)  # 8 bytes


class ControlBuffer:
    """High-level buffer for normalised control outputs.

    Parameters
    ----------
    slots : int
        Number of ring-buffer slots (default 8).
    name : str
        Shared-memory block name.
    create : bool
        Pass *True* from orchestrator, *False* from workers.
    """

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
        """Write normalised steering and throttle.

        Parameters
        ----------
        steering : float
            Normalised steering value (0.0–1.0).
        throttle : float
            Normalised throttle value (0.0–1.0).
        """
        payload = struct.pack(_CTRL_FMT, steering, throttle)
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, float, float]]:
        """Read the most recent control output.

        Returns
        -------
        (header, steering, throttle) or *None*.
        """
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
