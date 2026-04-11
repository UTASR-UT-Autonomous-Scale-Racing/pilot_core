"""
control — vehicle control library (pure Python, no worker/IPC dependencies).

    types        VehicleParams, VehicleState, Command
    track        TrackModel (centre-line, curvature, projection)
    controllers  PurePursuit, Stanley, TrackFollower, build_track_follower
    speed        curvature speed profile + no-brake longitudinal controller
    estimator    PoseEKF (pose [+ speed] → state)
    safety       SafetySupervisor (arming, watchdog, speed cap, slew limits)
    sim          closed-loop bicycle bench for evaluation
"""

from .controllers import PurePursuit, Stanley, TrackFollower, build_track_follower
from .estimator import EstimatorConfig, PoseEKF
from .safety import SafetyConfig, SafetySupervisor
from .track import TrackModel
from .types import Command, VehicleParams, VehicleState, wrap_angle

__all__ = [
    "Command", "EstimatorConfig", "PoseEKF", "PurePursuit", "SafetyConfig",
    "SafetySupervisor", "Stanley", "TrackFollower", "TrackModel", "VehicleParams",
    "VehicleState", "build_track_follower", "wrap_angle",
]
