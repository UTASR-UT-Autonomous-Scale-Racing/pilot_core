"""
Estimated vehicle state in the BEV plane, written by StateWorker.

Payload layout:
    [8 bytes]  float64  t         (state time, time.monotonic() clock)
    [8 bytes]  float32  x, y      (m, BEV plane)
    [4 bytes]  float32  yaw       (rad, CCW from +x)
    [4 bytes]  float32  v         (m/s, signed)
    [4 bytes]  float32  yaw_rate  (rad/s)

read_latest() -> (header, VehicleState); decode(payload) -> VehicleState.
"""

from __future__ import annotations

import struct
from typing import Optional

from control.types import VehicleState

from .ring_buffer import SharedRingBuffer
from .schema import DTypeTag, SlotHeader
from .typed import TypedRingBuffer

_STATE_FMT = "<d5f"
_PAYLOAD_SIZE = struct.calcsize(_STATE_FMT)


class StateBuffer(TypedRingBuffer):
    """Ring buffer of control.types.VehicleState."""

    def __init__(self, slots: int = 8, name: str = "state_buf", create: bool = True) -> None:
        self._rb = SharedRingBuffer(
            name=name,
            slots=slots,
            slot_payload_size=_PAYLOAD_SIZE,
            dtype_tag=DTypeTag.STATE,
            create=create,
        )

    def write(self, state: VehicleState, ts: Optional[float] = None) -> SlotHeader:
        payload = struct.pack(_STATE_FMT, state.t, state.x, state.y, state.yaw, state.v, state.yaw_rate)
        return self._rb.write(payload, ts)

    @staticmethod
    def decode(payload: bytes) -> VehicleState:
        t, x, y, yaw, v, w = struct.unpack_from(_STATE_FMT, payload, 0)
        return VehicleState(t=t, x=x, y=y, yaw=yaw, v=v, yaw_rate=w)
