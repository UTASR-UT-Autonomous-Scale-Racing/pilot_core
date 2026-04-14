"""
Linear time-varying MPC for lateral control (kinematic model, OSQP).

References: J. Kong, M. Pfeiffer, G. Schildbach, F. Borrelli, "Kinematic and
Dynamic Vehicle Models for Autonomous Driving Control Design", IEEE IV 2015;
P. Falcone et al., "Predictive Active Steering Control for Autonomous
Vehicle Systems", IEEE TCST 15(3), 2007.  Solver: B. Stellato et al.,
"OSQP: an operator splitting solver for quadratic programs", Math. Prog.
Comp., 2020.

Over a horizon of ``N`` steps the path-error model (see :mod:`control.lqr`)
is driven by the previewed track curvature:

    x_{k+1} = A x_k + B delta_k + c_k,   c_k = [0, -v kappa_k dt]

cost  sum_k  x_k' Q x_k + r (delta_k - delta_ff,k)^2 + r_d (delta_k - delta_{k-1})^2
s.t.  |delta_k| <= delta_max,   |delta_k - delta_{k-1}| <= steer_rate * dt

The initial state is propagated over the actuation ``delay`` with the last
command before solving.  The QP is condensed onto the ``N`` steering inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse

from .lqr import error_model
from .track import Projection, TrackModel
from .types import VehicleParams, VehicleState, wrap_angle


@dataclass
class MPC:
    params: VehicleParams
    horizon: int = 20                 # steps
    dt: float = 0.05                  # s per step (horizon = 1.0 s)
    q_lateral: float = 10.0
    q_heading: float = 2.0
    q_terminal: float = 5.0           # terminal cost multiplier
    r_steer: float = 1.0              # on (delta - delta_ff)²
    r_rate: float = 20.0              # on (delta_k - delta_{k-1})²
    delay: float = 0.05               # s, actuation delay compensated before solving
    v_min: float = 0.3

    def __post_init__(self) -> None:
        self._prev_delta = 0.0
        self.last_solve_ok = True

    def reset(self) -> None:
        self._prev_delta = 0.0

    def _predictions(self, A: np.ndarray, B: np.ndarray, c: np.ndarray, x0: np.ndarray):
        """Condensed dynamics: X = Phi x0 + Gamma u + Psi (stacked x_1..x_N)."""
        N = self.horizon
        Phi = np.zeros((2 * N, 2))
        Gamma = np.zeros((2 * N, N))
        Psi = np.zeros(2 * N)
        # A^i B for i = 0..N-1, so Gamma[k, j] = A^(k-j) B
        AiB = [B[:, 0]]
        for _ in range(N - 1):
            AiB.append(A @ AiB[-1])
        Ak = np.eye(2)
        drift = np.zeros(2)
        for k in range(N):
            Ak = A @ Ak
            drift = A @ drift + c[k]
            Phi[2 * k:2 * k + 2] = Ak
            Psi[2 * k:2 * k + 2] = drift
            for j in range(k + 1):
                Gamma[2 * k:2 * k + 2, j] = AiB[k - j]
        return Phi @ x0 + Psi, Gamma

    def steer(self, state: VehicleState, track: TrackModel, proj: Projection) -> float:
        p, N, dt = self.params, self.horizon, self.dt
        v = max(abs(state.v), self.v_min)
        A, B = error_model(v, p.wheelbase, dt)

        # Curvature preview along the horizon (+ the delay)
        s_k = proj.s + v * (self.delay + dt * np.arange(N))
        kappa = np.array([track.curvature[track.index_at(s)] for s in s_k])
        delta_ff = np.arctan(p.wheelbase * kappa)
        c = np.column_stack([np.zeros(N), -v * kappa * dt])

        # Initial error, propagated over the actuation delay with the last command
        x0 = np.array([proj.lateral_error, wrap_angle(state.yaw - proj.path_heading)])
        if self.delay > 0:
            Ad, Bd = error_model(v, p.wheelbase, self.delay)
            x0 = Ad @ x0 + Bd[:, 0] * self._prev_delta + np.array([0.0, -v * proj.curvature * self.delay])

        free, Gamma = self._predictions(A, B, c, x0)

        Qk = np.diag([self.q_lateral, self.q_heading])
        Qbar = np.kron(np.eye(N), Qk)
        Qbar[-2:, -2:] *= self.q_terminal
        # Rate operator D: (D u)_k = u_k - u_{k-1}, with u_{-1} = previous command
        D = np.eye(N) - np.eye(N, k=-1)
        d0 = np.zeros(N)
        d0[0] = self._prev_delta

        H = Gamma.T @ Qbar @ Gamma + self.r_steer * np.eye(N) + self.r_rate * D.T @ D
        g = Gamma.T @ Qbar @ free - self.r_steer * delta_ff - self.r_rate * D.T @ d0

        rate = p.max_steering_vel * dt
        Acon = sparse.vstack([sparse.eye(N), sparse.csc_matrix(D)], format="csc")
        lo = np.concatenate([np.full(N, -p.max_steering_angle), d0 - rate])
        hi = np.concatenate([np.full(N, p.max_steering_angle), d0 + rate])

        solver = osqp.OSQP()
        solver.setup(sparse.triu(sparse.csc_matrix(H + 1e-9 * np.eye(N)), format="csc"), g, Acon, lo, hi,
                     verbose=False, eps_abs=1e-5, eps_rel=1e-5, max_iter=4000, polishing=False)
        res = solver.solve(raise_error=False)
        self.last_solve_ok = res.info.status in ("solved", "solved inaccurate")
        if self.last_solve_ok and res.x is not None:
            delta = float(res.x[0])
        else:   # keep the previous command rather than jumping
            delta = self._prev_delta
        self._prev_delta = float(np.clip(delta, -p.max_steering_angle, p.max_steering_angle))
        return self._prev_delta
