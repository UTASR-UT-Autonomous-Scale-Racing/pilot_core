"""
Slot header layout for the shared ring buffers.

Each slot is a fixed 64-byte header followed by a payload of up to the slot's
max size. The timestamps in the header are what consumers use to judge freshness.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum

class DTypeTag(IntEnum):
    UNKNOWN = 0
    FRAME = 1          # RGB + depth pair
    IMU = 2            # 9 floats
    SEGMENTATION = 3
    POSE = 4           # 4x4 pose + map points
    CONTROL = 5        # steering + throttle in [0, 1], 0.5 = neutral
    LIDAR = 6          # 1080 x float32 ranges
    STATE = 7          # estimated vehicle state (BEV)
    POINTCLOUD = 8     # N x 3 float32 SLAM map points
    BOUNDARY = 9       # outer + inner track polylines
    RAW_IMU = 10       # accel xyz + gyro xyz, 6 floats (ORB-SLAM3 inertial modes)


# seq_no, produced_ts, read_ts, payload_size, schema_version, dtype_tag, reserved.
# Timestamps are time.monotonic(); read_ts sits at offset 16 and is only stamped on request.
HEADER_FORMAT = "<QddIHH32s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 64 bytes
HEADER_STRUCT = struct.Struct(HEADER_FORMAT)

# 2: trailing seq_no, uint8 segmentation masks, POINTCLOUD / BOUNDARY tags; v1 files still read
SCHEMA_VERSION = 2
SUPPORTED_SCHEMA_VERSIONS = (1, 2)


@dataclass
class SlotHeader:
    """One decoded slot header."""
    seq_no: int
    produced_ts: float
    read_ts: float
    payload_size: int
    schema_version: int
    dtype_tag: int

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(
            self.seq_no,
            self.produced_ts,
            self.read_ts,
            self.payload_size,
            self.schema_version,
            self.dtype_tag,
            b"\x00" * 32,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "SlotHeader":
        seq_no, produced_ts, read_ts, payload_size, schema_version, dtype_tag, _ = (
            HEADER_STRUCT.unpack(data)
        )
        return cls(
            seq_no=seq_no,
            produced_ts=produced_ts,
            read_ts=read_ts,
            payload_size=payload_size,
            schema_version=schema_version,
            dtype_tag=dtype_tag,
        )
