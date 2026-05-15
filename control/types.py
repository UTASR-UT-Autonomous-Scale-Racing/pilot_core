"""
Types shared by the controllers, estimator, safety supervisor and sim.

Headings are CCW from +x in the BEV plane (axes from bev_axes). Commands are signed:
steering +1 = full left (REP-103); throttle +1 forward, -1 backward, 0 neutral (coasts, no brake).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

import numpy as np
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path: str) -> str:
    """`path` as given if it exists, otherwise relative to the repo root."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    return os.path.join(_REPO_ROOT, path)


@dataclass(frozen=True)
class VehicleParams:
    """Physical params, loaded from vehicle_params.yaml."""

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
        # unknown keys are ignored, missing ones fall back to defaults
        with open(resolve_path(path)) as f:
            raw = yaml.safe_load(f) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in raw.items() if k in known})


@dataclass
class VehicleState:
    t: float                 # s, time.monotonic() clock
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
    """Signed command. Conventions are in the module docstring."""

    steering: float = 0.0
    throttle: float = 0.0

    def clipped(self) -> "Command":
        return Command(
            float(np.clip(self.steering, -1.0, 1.0)),
            float(np.clip(self.throttle, -1.0, 1.0)),
        )


def wrap_angle(a: float) -> float:
    """-> [-pi, pi)"""
    return (a + np.pi) % (2.0 * np.pi) - np.pi
