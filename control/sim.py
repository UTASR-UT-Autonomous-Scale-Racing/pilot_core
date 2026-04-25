"""
Closed-loop bench. Kinematic bicycle plant and a fake SLAM pose sensor,
driving the same estimator -> controller -> safety chain we run on the car.
The pytest benchmarks and scripts/sim_bench.py both use it.

Plant is from R. Rajamani, "Vehicle Dynamics and Control", Springer, 2012,
and J. Kong et al., IEEE IV 2015. Kinematic bicycle referenced at the rear
axle, params from vehicle_params.yaml. Steering is a rate-limited servo
(max_steering_vel) behind a transport delay. No brake: throttle u != 0 pulls
speed toward u * speed_per_throttle with time constant tau (negative u =
backward), and u == 0 coasts down at coast_decel.
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


# test tracks

def _offset_closed(centre: np.ndarray, offset: float) -> np.ndarray:
    d = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    return centre + offset * n


def oval_track(a: float = 6.5, b: float = 4.0, width: float = 1.6, n: int = 400):
    """a x b ellipse. Returns (outer, inner, centre)."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centre = np.column_stack([a * np.cos(th), b * np.sin(th)])
    return _offset_closed(centre, -width / 2), _offset_closed(centre, width / 2), centre


def twisty_track(radius: float = 6.0, width: float = 1.4, n: int = 800):
    """Loop with a mix of tight and fast sections."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = radius + 1.6 * np.sin(3 * th) + 0.9 * np.cos(2 * th)
    centre = np.column_stack([r * np.cos(th), r * np.sin(th)])
    return _offset_closed(centre, -width / 2), _offset_closed(centre, width / 2), centre


def random_track(rng: np.random.Generator, n: int = 800):
    # random size / width / curvature, used for demo collection
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = rng.uniform(5.0, 7.5)
    r = radius * np.ones(n)
    for k in (2, 3, 4):
        r += rng.uniform(0, 0.25 * radius / k) * np.cos(k * th + rng.uniform(0, 2 * np.pi))
    centre = np.column_stack([r * np.cos(th), r * np.sin(th)])
    width = rng.uniform(1.3, 1.9)
    return _offset_closed(centre, -width / 2), _offset_closed(centre, width / 2), centre


@dataclass
class PlantConfig:
    speed_per_throttle: float = 3.0    # m/s steady-state speed per unit throttle
    tau: float = 0.4                   # s, drivetrain time constant
    coast_decel: float = 0.8           # m/s^2 at neutral throttle
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
        # steering: transport delay, then rate-limited servo
        self._steer_queue.append((t + self.c.steer_delay, cmd.steering * self.p.max_steering_angle))
        target = self.delta
        while self._steer_queue and self._steer_queue[0][0] <= t:
            target = self._steer_queue.popleft()[1]
        max_d = self.p.max_steering_vel * dt
        self.delta += float(np.clip(target - self.delta, -max_d, max_d))

        # no brake. neutral just coasts
        if cmd.throttle != 0.0:
            self.v += (cmd.throttle * self.c.speed_per_throttle - self.v) / self.c.tau * dt
        else:
            self.v -= np.sign(self.v) * min(abs(self.v), self.c.coast_decel * dt)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / self.p.wheelbase * np.tan(self.delta) * dt


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
    """Drive laps and score against the true track."""
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
        # pose sensor: sample truth now, deliver after pose_latency
        if t >= next_pose:
            pose_queue.append((t + plant_cfg.pose_latency, t,
                               car.x + rng.normal(0, plant_cfg.pos_noise),
                               car.y + rng.normal(0, plant_cfg.pos_noise),
                               car.yaw + rng.normal(0, plant_cfg.yaw_noise)))
            next_pose += 1.0 / plant_cfg.pose_rate
        while pose_queue and pose_queue[0][0] <= t:
            _, ts, px, py, pyaw = pose_queue.popleft()
            ekf.update_pose(ts, px, py, pyaw)

        if t >= next_ctrl:
            state = ekf.predict_to(t)
            if state is not None:
                cmd, cmd_t = follower.step(state), t
            next_ctrl += 1.0 / control_rate

        est = ekf.predict_to(t)
        out = sup.step(t, cmd, cmd_t, armed=True, speed=est.v if est else None)
        car.step(out, t, dt)

        # scoring uses the true position, not the estimate
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


def render_depth(x, y, yaw, walls, K, width, height, cam_height=0.12, wall_height=0.3, max_range=6.0):
    """Ray-cast a z-depth image (metres) of vertical track walls.

    walls: list of closed (N, 2) polylines. Camera is at (x, y) looking along
    yaw, level, cam_height above a flat floor.
    """
    u = np.arange(width)
    bearing = np.arctan2(-(u - K.cx), K.fx)                 # + = left, per column
    ang = yaw + bearing
    d = np.column_stack([np.cos(ang), np.sin(ang)])          # (W, 2) ray dirs
    p0 = np.vstack([np.vstack([w, w[:1]])[:-1] for w in walls])
    p1 = np.vstack([np.vstack([w, w[:1]])[1:] for w in walls])
    e = p1 - p0                                              # (S, 2)
    o = np.array([x, y])
    # solve o + t d = p0 + s e for every (ray, segment) pair
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    w_ = p0[None] - o
    t = (w_[..., 0] * e[None, :, 1] - w_[..., 1] * e[None, :, 0]) / np.where(denom == 0, np.nan, denom)
    sg = (w_[..., 0] * d[:, None, 1] - w_[..., 1] * d[:, None, 0]) / np.where(denom == 0, np.nan, denom)
    t = np.where((t > 0) & (sg >= 0) & (sg <= 1), t, np.inf)
    dist = np.minimum(np.nanmin(t, axis=1), max_range)       # (W,) horizontal distance to wall
    z_wall = dist * np.cos(bearing)

    v = np.arange(height)[:, None]
    ray_y = (v - K.cy) / K.fy                                # y per unit z (down)
    depth = np.zeros((height, width))
    # wall pixels = rays that hit between the floor and the top of the wall
    y_at_wall = ray_y * z_wall[None, :]
    on_wall = (y_at_wall <= cam_height) & (y_at_wall >= cam_height - wall_height) & (dist[None, :] < max_range)
    depth[on_wall] = np.broadcast_to(z_wall[None, :], depth.shape)[on_wall]
    # floor in front of the wall
    with np.errstate(divide="ignore"):
        z_floor = np.where(ray_y > 1e-6, cam_height / ray_y, np.inf)
    floor = ~on_wall & (np.broadcast_to(z_floor, depth.shape) < np.minimum(z_wall[None, :], max_range))
    depth[floor] = np.broadcast_to(z_floor, depth.shape)[floor]
    return depth


def run_depth_closed_loop(follower, outer, inner, params, duration=40.0, plant=None,
                          safety=None, width=160, height=100, camera_rate=15.0, dt=0.005,
                          K=None, camera_height=None):
    """Like run_closed_loop but for depth-image controllers
    (follower.step(depth) -> Command, e.g. DepthGapFollower or
    VAEPolicyFollower). Only rendered depth goes in, no pose and no map; the
    true track is used just for scoring.

    K and camera_height default to follower.K / follower.cfg.camera_height if
    it has them (DepthGapFollower does). Learned followers don't, so pass
    them in.
    """
    K = K if K is not None else getattr(follower, "K", None)
    if camera_height is None:
        camera_height = getattr(getattr(follower, "cfg", None), "camera_height", None)
    if K is None or camera_height is None:
        raise ValueError("run_depth_closed_loop needs K and camera_height for this follower")
    from .track import TrackModel

    plant_cfg = plant or PlantConfig()
    track = TrackModel(outer, inner)
    start = track.points[0]
    car = BicyclePlant(params, plant_cfg, float(start[0]), float(start[1]), float(track.heading[0]))
    sup = SafetySupervisor(safety)
    cmd = cmd_t = None
    next_frame = 0.0
    rows, hint, s_prev, s_total = [], None, None, 0.0
    t = 0.0
    while t < duration:
        if t >= next_frame:
            depth = render_depth(car.x, car.y, car.yaw, [outer, inner], K, width, height,
                                 cam_height=camera_height)
            cmd, cmd_t = follower.step(depth), t
            next_frame += 1.0 / camera_rate
        out = sup.step(t, cmd, cmd_t, armed=True, speed=car.v)
        car.step(out, t, dt)
        pr = track.project(car.x, car.y, hint=hint)
        hint = pr.idx
        if s_prev is not None:
            s_total += (pr.s - s_prev + track.length / 2) % track.length - track.length / 2
        s_prev = pr.s
        rows.append((t, car.x, car.y, car.v, out.steering, out.throttle, pr.lateral_error,
                     float(track.half_width[pr.idx])))
        t += dt
    log = np.array(rows)
    e = log[:, 6]
    return RunResult(
        laps=s_total / track.length, lap_times=[],
        xtrack_rms=float(np.sqrt(np.mean(e ** 2))), xtrack_max=float(np.max(np.abs(e))),
        off_track_fraction=float(np.mean(np.abs(e) > log[:, 7] - params.width / 2)),
        mean_speed=float(np.mean(log[:, 3])), max_speed=float(np.max(log[:, 3])),
        steer_rate_rms=float(np.sqrt(np.mean((np.diff(log[:, 4]) / dt) ** 2))), log=log[:, :7],
    )


def make_follower_factory(mode: str, params: VehicleParams, lateral_cfg=None, speed_cfg=None):
    """Returns f(outer, inner) -> TrackFollower for mode."""
    from .controllers import build_track_follower
    from .track import TrackModel

    def factory(outer, inner):
        return build_track_follower(mode, TrackModel(outer, inner), params, lateral_cfg, speed_cfg)

    return factory


__all__ = [
    "BicyclePlant", "PlantConfig", "RunResult", "oval_track", "twisty_track", "random_track",
    "run_closed_loop", "run_depth_closed_loop", "render_depth", "make_follower_factory", "VehicleState",
]
