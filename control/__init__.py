"""Vehicle control library. Pure Python, nothing in here should import the
workers or the IPC buffers.

Rough map: types (params/state/command), track (centre-line model),
controllers + speed (lateral laws and the speed planner), estimator (pose EKF),
safety (arming, watchdog, caps), sim (closed-loop bicycle bench).
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
