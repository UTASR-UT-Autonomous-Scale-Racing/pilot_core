"""
Shared-memory ring buffers used to pass data between the worker processes.

One writer per buffer, seqlock-protected slots (details in ring_buffer.py).
Readers always get the newest entry. Each typed wrapper describes its payload
byte layout at the top of its own module.
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
from .lidar_buffer import LidarBuffer
from .state_buffer import StateBuffer

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
    "LidarBuffer",
    "StateBuffer",
]
