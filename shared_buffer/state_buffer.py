"""
StateBuffer — estimated vehicle state in the BEV plane (from StateWorker).

Payload layout:
    [8 bytes]  float64  t         (state time, time.time() clock)
    [4 bytes]  float32  x, y      (m, BEV plane)
    [4 bytes]  float32  yaw       (rad, CCW from +x)
    [4 bytes]  float32  v         (m/s, signed)
    [4 bytes]  float32  yaw_rate  (rad/s)
"""

from __future__ import annotations

import struct
from typing import Optional, Tuple

from control.types import VehicleState

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader

_STATE_FMT = "<d5f"
_PAYLOAD_SIZE = struct.calcsize(_STATE_FMT)


class StateBuffer:
    def __init__(self, slots: int = 8, name: str = "state_buf", create: bool = True) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_PAYLOAD_SIZE,
            dtype_tag=DTypeTag.STATE,
            create=create,
        )

    def write(self, state: VehicleState) -> SlotHeader:
        payload = struct.pack(_STATE_FMT, state.t, state.x, state.y, state.yaw, state.v, state.yaw_rate)
        return self._rb.write(payload)

    def read_latest(self) -> Optional[Tuple[SlotHeader, VehicleState]]:
        result = self._rb.read_latest()
        if result is None:
            return None
        header, payload = result
        t, x, y, yaw, v, w = struct.unpack(_STATE_FMT, payload[:_PAYLOAD_SIZE])
        return header, VehicleState(t=t, x=x, y=y, yaw=yaw, v=v, yaw_rate=w)

    def close(self):
        self._rb.close()

    def unlink(self):
        self._rb.unlink()
