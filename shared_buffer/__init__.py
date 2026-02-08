"""
shared_buffer — Zero-copy shared-memory ring buffers for inter-process communication.

Each buffer is a fixed-slot ring backed by multiprocessing.shared_memory.
Producers write; consumers read the most recent entry (pop-most-recent semantics).
"""

from .ring_buffer import SharedRingBuffer
from .schema import SlotHeader, DTypeTag, HEADER_SIZE, HEADER_STRUCT
from .frame_buffer import FrameBuffer
from .imu_buffer import IMUBuffer
from .segment_buffer import SegmentationBuffer
from .pose_buffer import PoseBuffer
from .control_buffer import ControlBuffer
from .pointcloud_buffer import PointCloudBuffer
from .track_boundary_buffer import TrackBoundaryBuffer

__all__ = [
    "SharedRingBuffer",
    "SlotHeader",
    "DTypeTag",
    "HEADER_SIZE",
    "HEADER_STRUCT",
    "FrameBuffer",
    "IMUBuffer",
    "SegmentationBuffer",
    "PoseBuffer",
    "ControlBuffer",
    "PointCloudBuffer",
    "TrackBoundaryBuffer",
]
