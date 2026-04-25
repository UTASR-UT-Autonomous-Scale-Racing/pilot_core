"""Bird's-eye-view helpers used by both the mapping and control workers."""

from __future__ import annotations

import numpy as np

def pose_to_bev(
    pose: np.ndarray,
    bev_axes: tuple[int, int] = (0, 2),
    forward_axis: int = 2,
) -> tuple[np.ndarray, float]:
    """BEV position + heading from a body-to-world pose.

    bev_axes = (a0, a1) are the world axes spanning the ground plane;
    forward_axis is the body axis that points in the direction of travel.
      ORB-SLAM3 camera frame (Y-down, Z-forward): (0, 2), forward 2
      AutoDRIVE / ROS frame (Z-up, X-forward):    (0, 1), forward 0

    Heading goes from +a0 towards +a1, which matches how the track
    controllers compute bearings (atan2(d1, d0)).
    """
    a0, a1 = bev_axes
    position = np.array([pose[a0, 3], pose[a1, 3]], dtype=np.float32)
    heading = float(np.arctan2(pose[a1, forward_axis], pose[a0, forward_axis]))
    return position, heading
