"""
LQR lateral controller on the kinematic path-error model.

Ref: J. M. Snider, "Automatic Steering Methods for Autonomous Automobile Path Tracking", CMU-RI-TR-09-08, 2009.
State [e (+ = left of path), yaw - path_heading]; input delta - atan(L * kappa). Gains cached per 0.1 m/s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from scipy.linalg import solve_discrete_are

from .track import Projection, TrackModel
from .types import VehicleParams, VehicleState, wrap_angle


def error_model(v: float, wheelbase: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete (A, B) of the path-error model, forward Euler."""
    A = np.array([[1.0, v * dt], [0.0, 1.0]])
    B = np.array([[0.0], [v / wheelbase * dt]])
    return A, B


@dataclass
class LQR:
    params: VehicleParams
    q_lateral: float = 4.0        # weight on e² (m⁻²)
    q_heading: float = 1.0        # on e_psi^2, rad^-2
    r_steer: float = 2.0          # on (delta - delta_ff)^2, rad^-2
    dt: float = 1.0 / 30.0        # control period the gains are designed for
    # design speed floor; v=0 makes the model singular
    v_min: float = 0.3
    _gains: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        pass

    def gain(self, v: float) -> np.ndarray:
        key = int(round(max(abs(v), self.v_min) * 10))
        if key not in self._gains:
            A, B = error_model(key / 10.0, self.params.wheelbase, self.dt)
            Q = np.diag([self.q_lateral, self.q_heading])
            R = np.array([[self.r_steer]])
            P = solve_discrete_are(A, B, Q, R)
            self._gains[key] = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        return self._gains[key]

    def steer(self, state: VehicleState, track: TrackModel, proj: Projection) -> float:
        x = np.array([proj.lateral_error, wrap_angle(state.yaw - proj.path_heading)])
        delta_ff = float(np.arctan(self.params.wheelbase * proj.curvature))
        return float(delta_ff - (self.gain(state.v) @ x)[0])
