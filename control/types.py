"""
Core control types shared by controllers, the state estimator, the safety
supervisor and the closed-loop simulator.

Conventions (see README "Buffer contracts"):

* BEV plane: 2-D ``(x, y)`` = the two world ground-plane axes selected by
  ``bev_axes``.  Headings are measured counter-clockwise from +x.
* Signed commands: steering **+1 = full left** (counter-clockwise, ROS /
  REP-103), throttle +1 = full forward, -1 = full backward, 0 = neutral
  (the car coasts and slows down; there is no separate brake).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class VehicleParams:
    """Physical vehicle parameters (``vehicle_params.yaml``)."""

    wheelbase: float = 0.3302
    width: float = 0.2032
    max_steering_angle: float = 0.4189
    max_steering_vel: float = 3.2
    max_speed: float = 7.0
    max_accel: float = 7.51
    max_decel: float = 8.26
    friction_coeff: float = 0.523

    @classmethod
    def load(cls, path: str = "vehicle_params.yaml") -> "VehicleParams":
        """Load from YAML; unknown keys are ignored, missing keys use defaults."""
        if not os.path.isabs(path):
            path = os.path.join(_REPO_ROOT, path)
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in raw.items() if k in known})


@dataclass
class VehicleState:
    """Estimated vehicle state in the BEV plane."""

    t: float                 # seconds (time.time() clock)
    x: float
    y: float
    yaw: float               # rad, CCW from +x
    v: float = 0.0           # m/s, signed (negative = moving backward)
    yaw_rate: float = 0.0    # rad/s

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class Command:
    """Signed actuation command (see module docstring for conventions)."""

    steering: float = 0.0
    throttle: float = 0.0

    def clipped(self) -> "Command":
        return Command(
            float(np.clip(self.steering, -1.0, 1.0)),
            float(np.clip(self.throttle, -1.0, 1.0)),
        )


def wrap_angle(a: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi
