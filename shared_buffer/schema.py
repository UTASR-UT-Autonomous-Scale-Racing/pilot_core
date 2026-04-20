"""
Schema definitions for the shared ring buffer slot header.

Every slot contains a fixed-size header followed by a variable (but max-bounded) payload.
The header carries timestamps and metadata so consumers can always determine freshness.
"""

import struct
from dataclasses import dataclass
from enum import IntEnum

# --- DType tags ---

class DTypeTag(IntEnum):
    """Identifies the payload type stored in a buffer slot."""
    UNKNOWN = 0
    FRAME = 1          # RGB + Depth frame pair
    IMU = 2            # 9-float IMU vector
    SEGMENTATION = 3   # Segmentation mask
    POSE = 4           # 4x4 pose + map points
    CONTROL = 5        # Steering + throttle in [0, 1] (0.5 = neutral)
    LIDAR = 6          # 1080-float32 range scan
    STATE = 7          # Estimated vehicle state (BEV)


# --- Header layout ---
#
#   Offset  Size  Type      Field
#   0       8     uint64    seq_no           (monotonically increasing write counter)
#   8       8     float64   produced_ts      (time.time() at write)
#   16      8     float64   read_ts          (time.time() at read — set by consumer)
#   24      4     uint32    payload_size     (actual bytes written in payload area)
#   28      2     uint16    schema_version
#   30      2     uint16    dtype_tag        (DTypeTag value)
#   32      32    bytes     _reserved        (padding to 64 bytes)
#
HEADER_FORMAT = "<QddIHH32s"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 64 bytes
HEADER_STRUCT = struct.Struct(HEADER_FORMAT)

SCHEMA_VERSION = 1


@dataclass
class SlotHeader:
    """Decoded header for one ring-buffer slot."""
    seq_no: int
    produced_ts: float
    read_ts: float
    payload_size: int
    schema_version: int
    dtype_tag: int

    def pack(self) -> bytes:
        """Serialise to bytes suitable for writing into shared memory."""
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
        """Deserialise from raw bytes read from shared memory."""
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
