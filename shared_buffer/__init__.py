"""
Shared-memory ring buffers used to pass data between the worker processes.

One writer per buffer, seqlock-protected slots (details in ring_buffer.py).
read_latest() returns the newest entry; read_since(seq) returns every entry
newer than seq that is still in the ring. Each typed wrapper describes its
payload byte layout at the top of its own module and exposes a
decode(payload) staticmethod, so raw payloads (e.g. from recordings) decode
exactly like read_latest().

Buffer names: the typed buffers take a ``name`` (the POSIX shm block name,
visible under /dev/shm). To run two pipelines side by side, or to keep a
crashed run's blocks from colliding with a new one, give every buffer of a
run the same prefix via buffer_name(prefix, base), e.g.
``FrameBuffer(name=buffer_name("run42", "frame_buf"))`` -> "run42_frame_buf".
"""

from typing import Dict, Optional, Type

from .ring_buffer import SharedRingBuffer
from .schema import SlotHeader, DTypeTag, HEADER_SIZE, HEADER_STRUCT
from .typed import TypedRingBuffer
from .frame_buffer import FrameBuffer
from .imu_buffer import IMUBuffer
from .segment_buffer import SegmentationBuffer
from .pose_buffer import PoseBuffer
from .control_buffer import ControlBuffer
from .pointcloud_buffer import PointCloudBuffer
from .track_boundary_buffer import TrackBoundaryBuffer
from .lidar_buffer import LidarBuffer
from .state_buffer import StateBuffer


def buffer_name(prefix: Optional[str], base: str) -> str:
    """Shared-memory block name for ``base`` (e.g. "frame_buf") in a run
    with the given prefix; no prefix -> base unchanged."""
    if not prefix:
        return base
    return f"{prefix.rstrip('_')}_{base}"


# DTypeTag -> typed buffer class, for decoding raw payloads
BUFFER_CLASSES: Dict[int, Type[TypedRingBuffer]] = {
    DTypeTag.FRAME: FrameBuffer,
    DTypeTag.IMU: IMUBuffer,
    DTypeTag.SEGMENTATION: SegmentationBuffer,
    DTypeTag.POSE: PoseBuffer,
    DTypeTag.CONTROL: ControlBuffer,
    DTypeTag.LIDAR: LidarBuffer,
    DTypeTag.STATE: StateBuffer,
    DTypeTag.POINTCLOUD: PointCloudBuffer,
    DTypeTag.BOUNDARY: TrackBoundaryBuffer,
}

# recording / config names -> typed buffer class (start.py's recording names)
BUFFER_CLASSES_BY_NAME: Dict[str, Type[TypedRingBuffer]] = {
    "frame": FrameBuffer,
    "imu": IMUBuffer,
    "segmentation": SegmentationBuffer,
    "pose": PoseBuffer,
    "control": ControlBuffer,
    "lidar": LidarBuffer,
    "state": StateBuffer,
    "pointcloud": PointCloudBuffer,
    "boundary": TrackBoundaryBuffer,
}


def buffer_class_for(dtype_tag: int = 0, name: str = "") -> Optional[Type[TypedRingBuffer]]:
    """Typed buffer class for a DTypeTag, falling back to a recording name
    ("frame", "imu", ...) when the tag is UNKNOWN. None if neither matches."""
    cls = BUFFER_CLASSES.get(int(dtype_tag))
    if cls is None and name:
        cls = BUFFER_CLASSES_BY_NAME.get(name)
    return cls


__all__ = [
    "SharedRingBuffer",
    "SlotHeader",
    "DTypeTag",
    "HEADER_SIZE",
    "HEADER_STRUCT",
    "TypedRingBuffer",
    "FrameBuffer",
    "IMUBuffer",
    "SegmentationBuffer",
    "PoseBuffer",
    "ControlBuffer",
    "PointCloudBuffer",
    "TrackBoundaryBuffer",
    "LidarBuffer",
    "StateBuffer",
    "BUFFER_CLASSES",
    "BUFFER_CLASSES_BY_NAME",
    "buffer_class_for",
    "buffer_name",
]
