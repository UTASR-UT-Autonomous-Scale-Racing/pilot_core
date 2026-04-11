"""
Closed-loop evaluation bench: kinematic bicycle plant + simulated SLAM pose
sensor, driving the *same* estimator → controller → safety chain used on
the car.  Used by the pytest benchmarks and ``scripts/sim_bench.py``.

Plant model
-----------
* Kinematic bicycle referenced at the rear axle (``vehicle_params.yaml``).
* Steering: rate-limited servo (``max_steering_vel``) behind a transport delay.
* Drivetrain with no brake: throttle ``u != 0`` drives the speed towards
  ``u * speed_per_throttle`` with time constant ``tau`` (negative = backward);
  neutral (``u == 0``) coasts, decelerating at ``coast_decel``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .controllers import TrackFollower
from .estimator import EstimatorConfig, PoseEKF
from .safety import SafetyConfig, SafetySupervisor
from .types import Command, VehicleParams, VehicleState


# ---------------------------------------------------------------- tracks

def _offset_closed(centre: np.ndarray, offset: float) -> np.ndarray:
    d = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return centre + offset * n


def oval_track(a: float = 6.5, b: float = 4.0, width: float = 1.6, n: int = 400):
    """Ellipse centre-line ``a × b`` → (outer, inner, centre)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centre = np.column_stack([a * np.cos(th), b * np.sin(th)])
    return _offset_closed(centre, -width / 2), _offset_closed(centre, width / 2), centre


def twisty_track(radius: float = 6.0, width: float = 1.4, n: int = 800):
    """Closed track with varying curvature (tight and fast sections)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius + 1.6 * np.sin(3 * th) + 0.9 * np.cos(2 * th)
    centre = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return _offset_closed(centre, -width / 2), _offset_closed(centre, width / 2), centre


# ----------------------------------------------------------------- plant

@dataclass
class PlantConfig:
    speed_per_throttle: float = 3.0
    tau: float = 0.4
    coast_decel: float = 0.8
    steer_delay: float = 0.05          # s transport delay on steering
    pose_rate: float = 15.0            # Hz
    pose_latency: float = 0.06         # s
    pos_noise: float = 0.01            # m
    yaw_noise: float = 0.01            # rad


class BicyclePlant:
    def __init__(self, params: VehicleParams, cfg: PlantConfig, x: float, y: float, yaw: float) -> None:
        self.p, self.c = params, cfg
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0
        self.delta = 0.0
        self._steer_queue: deque = deque()

    def step(self, cmd: Command, t: float, dt: float) -> None:
        # Steering: transport delay, then rate-limited servo
        self._steer_queue.append((t + self.c.steer_delay, cmd.steering * self.p.max_steering_angle))
        target = self.delta
        while self._steer_queue and self._steer_queue[0][0] <= t:
            target = self._steer_queue.popleft()[1]
        max_d = self.p.max_steering_vel * dt
        self.delta += float(np.clip(target - self.delta, -max_d, max_d))

        # Drivetrain: no brake
        if cmd.throttle != 0.0:
            self.v += (cmd.throttle * self.c.speed_per_throttle - self.v) / self.c.tau * dt
        else:
            self.v -= np.sign(self.v) * min(abs(self.v), self.c.coast_decel * dt)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.p.wheelbase * np.tan(self.delta) * dt


# ---------------------------------------------------------------- metrics

@dataclass
class RunResult:
    laps: float
    lap_times: List[float]
    xtrack_rms: float
    xtrack_max: float
    off_track_fraction: float
    mean_speed: float
    max_speed: float
    steer_rate_rms: float           # signed units / s
    log: np.ndarray = field(repr=False, default=None)   # t, x, y, v, steer, throttle, e

    def summary(self) -> str:
        best = f"{min(self.lap_times):.2f}s" if self.lap_times else "-"
        return (
            f"laps={self.laps:.2f}  best_lap={best}  xtrack rms/max={self.xtrack_rms:.3f}/{self.xtrack_max:.3f} m  "
            f"off_track={100 * self.off_track_fraction:.1f}%  speed mean/max={self.mean_speed:.2f}/{self.max_speed:.2f} m/s  "
            f"steer_rate_rms={self.steer_rate_rms:.2f}/s"
        )


def run_closed_loop(
    make_follower: Callable[[np.ndarray, np.ndarray], TrackFollower],
    outer: np.ndarray,
    inner: np.ndarray,
    params: VehicleParams,
    duration: float = 60.0,
    plant: Optional[PlantConfig] = None,
    safety: Optional[SafetyConfig] = None,
    estimator: Optional[EstimatorConfig] = None,
    control_rate: float = 30.0,
    dt: float = 0.005,
    seed: int = 0,
) -> RunResult:
    """Drive a lap-loop scenario and score it against the true track."""
    rng = np.random.default_rng(seed)
    plant_cfg = plant or PlantConfig()
    follower = make_follower(outer, inner)
    track = follower.track

    # Start on the centre-line, aligned with the driving direction
    start = track.points[0]
    car = BicyclePlant(params, plant_cfg, float(start[0]), float(start[1]), float(track.heading[0]))
    ekf = PoseEKF(estimator)
    sup = SafetySupervisor(safety)

    pose_queue: deque = deque()
    next_pose = next_ctrl = 0.0
    cmd: Optional[Command] = None
    cmd_t: Optional[float] = None
    out = Command()
    rows = []
    hint = None
    s_prev = None
    s_total = 0.0
    lap_start = 0.0
    lap_times: List[float] = []

    t = 0.0
    while t < duration:
        # Pose sensor: sample the truth, deliver after latency
        if t >= next_pose:
            pose_queue.append((t + plant_cfg.pose_latency, t,
                               car.x + rng.normal(0, plant_cfg.pos_noise),
                               car.y + rng.normal(0, plant_cfg.pos_noise),
                               car.yaw + rng.normal(0, plant_cfg.yaw_noise)))
            next_pose += 1.0 / plant_cfg.pose_rate
        while pose_queue and pose_queue[0][0] <= t:
            _, ts, px, py, pyaw = pose_queue.popleft()
            ekf.update_pose(ts, px, py, pyaw)

        # Controller
        if t >= next_ctrl:
            state = ekf.predict_to(t)
            if state is not None:
                cmd, cmd_t = follower.step(state), t
            next_ctrl += 1.0 / control_rate

        # Safety + plant
        est = ekf.predict_to(t)
        out = sup.step(t, cmd, cmd_t, armed=True, speed=est.v if est else None)
        car.step(out, t, dt)

        # Score against the truth
        pr = track.project(car.x, car.y, hint=hint)
        hint = pr.idx
        if s_prev is not None:
            ds = (pr.s - s_prev + track.length / 2) % track.length - track.length / 2
            s_total += ds
            if s_total >= track.length * (len(lap_times) + 1):
                lap_times.append(t - lap_start)
                lap_start = t
        s_prev = pr.s
        rows.append((t, car.x, car.y, car.v, out.steering, out.throttle, pr.lateral_error,
                     float(track.half_width[pr.idx])))
        t += dt

    log = np.array(rows)
    e = log[:, 6]
    steer_rate = np.diff(log[:, 4]) / dt
    return RunResult(
        laps=s_total / track.length,
        lap_times=lap_times,
        xtrack_rms=float(np.sqrt(np.mean(e ** 2))),
        xtrack_max=float(np.max(np.abs(e))),
        off_track_fraction=float(np.mean(np.abs(e) > log[:, 7] - params.width / 2)),
        mean_speed=float(np.mean(log[:, 3])),
        max_speed=float(np.max(log[:, 3])),
        steer_rate_rms=float(np.sqrt(np.mean(steer_rate ** 2))),
        log=log[:, :7],
    )


def make_follower_factory(mode: str, params: VehicleParams, lateral_cfg=None, speed_cfg=None):
    """Convenience factory: boundaries → TrackFollower for ``mode``."""
    from .controllers import build_track_follower
    from .track import TrackModel

    def factory(outer, inner):
        return build_track_follower(mode, TrackModel(outer, inner), params, lateral_cfg, speed_cfg)

    return factory


__all__ = [
    "BicyclePlant", "PlantConfig", "RunResult", "oval_track", "twisty_track",
    "run_closed_loop", "make_follower_factory", "VehicleState",
]
