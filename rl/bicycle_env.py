"""Kinematic-bicycle stand-in for rl/track_env.TrackEnv, for fast PPO pretraining.

Same observation (control/boundary.track_observation on prepare_boundary'd outer/inner
lines), same action space and action -> command path (signed_to_unit, forward-only
throttle, SafetySupervisor with the car's SafetyConfig, as ActuatorWorker runs it) and the
same reward terms and termination as TrackEnv, so a model pretrained here can be resumed by
``rl/train.py --resume`` on the live sim and exported for the on-car ``policy`` mode.

Plant (SimPlant): rear-axle kinematic bicycle fitted to the AutoDRIVE RoboRacer
(docs/autodrive.md "Calibration", proof/runs/2e_calibration): throttle is a speed
set-point of 24.8 m/s per unit, reached with an acceleration-limited first-order response
(~5.5 m/s^2 up, ~6 m/s^2 down to a neutral set-point of 0), wheelbase 0.342 m, +-30 deg
steering, plus a friction cap on yaw rate (|v * r| <= a_lat_max).

Deliberate differences from TrackEnv (the sim's counters do not exist here):
  * lap / checkpoints come from progress along the centre-line (TrackModel), not the sim:
    a lap is one centre-line length of net forward progress since reset (episodes can start
    anywhere on the loop), a checkpoint every ``checkpoint_spacing`` m of new progress
    (the stock sim build reports no checkpoints; set checkpoint_spacing=0 to match it).
  * a collision is any body circle (3 along the car, r = 0.14 m) touching a true wall
    polyline, instead of the sim's collision counter; no post-reset collision grace needed.
  * tracks are randomised per episode (random generated loops + the saved AutoDRIVE maps),
    starts are randomised, and pose/speed noise, actuation latency and small dynamics
    randomisation are added (all switchable) for robustness.
  * time is simulated (no wall-clock waits), so no stale-pose truncation.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Optional, Sequence

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("gymnasium is required: pip install gymnasium")

from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

from control.boundary import obs_dim, prepare_boundary, track_observation
from control.safety import SafetyConfig, SafetySupervisor
from control.sim import BicyclePlant, PlantConfig, random_track
from control.track import TrackModel
from control.types import Command, VehicleParams
from workers.control_worker import signed_to_unit, unit_to_signed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAPS = (os.path.join(REPO_ROOT, "maps", "autodrive_practice"),
                os.path.join(REPO_ROOT, "maps", "autodrive_compete"))
VEHICLE_PARAMS = os.path.join(REPO_ROOT, "vehicle_params_roboracer.yaml")
# where the sim respawns the car on both AutoDRIVE tracks (sim_messages.csv after a reset)
AUTODRIVE_SPAWN = (0.80, 3.158, -np.pi / 2)

# TrackEnv kwargs that only make sense against the live sim (accepted, ignored here)
_SIM_ONLY_KWARGS = ("sim_host", "sim_port", "shm_prefix", "stale_timeout", "reset_timeout")


@dataclass
class SimPlantConfig(PlantConfig):
    """PlantConfig for the AutoDRIVE RoboRacer (fit to the 2e calibration run)."""
    speed_per_throttle: float = 24.8   # m/s set-point per unit throttle
    tau: float = 0.05                  # s, set-point tracking time constant
    coast_decel: float = 6.0           # m/s^2 (neutral = set-point 0: the sim brakes)
    steer_delay: float = 0.0           # latency is modelled on the command instead
    max_accel: float = 5.5             # m/s^2, observed 0 -> 2 m/s ramp
    a_lat_max: float = 5.1             # m/s^2, friction_coeff 0.523 * g
    steer_gain: float = 1.0            # applied angle per commanded angle


class SimPlant(BicyclePlant):
    """BicyclePlant with the sim's speed set-point drivetrain and a lateral-grip cap."""

    def step(self, cmd: Command, t: float, dt: float) -> None:
        c = self.c
        self._steer_queue.append((t + c.steer_delay, cmd.steering * c.steer_gain * self.p.max_steering_angle))
        target = self.delta
        while self._steer_queue and self._steer_queue[0][0] <= t:
            target = self._steer_queue.popleft()[1]
        max_d = self.p.max_steering_vel * dt
        self.delta += float(np.clip(target - self.delta, -max_d, max_d))

        v_set = cmd.throttle * c.speed_per_throttle   # 0 at neutral: the sim stops the car
        accel = float(np.clip((v_set - self.v) / c.tau, -c.coast_decel, c.max_accel))
        self.v += accel * dt

        yaw_rate = self.v / self.p.wheelbase * np.tan(self.delta)
        if abs(self.v) > 1e-3:
            cap = c.a_lat_max / abs(self.v)   # understeer once grip runs out
            yaw_rate = float(np.clip(yaw_rate, -cap, cap))
        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += yaw_rate * dt


# tracks

def _resample(pts: np.ndarray, spacing: float) -> np.ndarray:
    loop = np.vstack([pts, pts[:1]])
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(loop, axis=0), axis=1))])
    n = max(int(round(arc[-1] / spacing)), 8)
    s = np.arange(n) * (arc[-1] / n)
    return np.column_stack([np.interp(s, arc, loop[:, 0]), np.interp(s, arc, loop[:, 1])])


def _curvature(c: np.ndarray, k: int) -> np.ndarray:
    fwd, back = np.roll(c, -k, axis=0), np.roll(c, k, axis=0)
    d1, d2 = (fwd - back) / 2.0, fwd - 2.0 * c + back
    return (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-12)


def generate_track(rng: np.random.Generator, width_range=(1.2, 1.6), length_range=(25.0, 60.0),
                   point_spacing=(0.08, 0.14), max_tries: int = 60):
    """Random closed track -> (outer, inner) raw boundaries, ~mapped-boundary point density.

    Periodic spline through jittered control points on a stretched loop (straights, hairpins
    down to ~0.8 m centre-line radius), varying width; rejects loops whose walls would cross.
    Which side is "outer" (first) is random, as on the saved maps.
    """
    ds = 0.05
    for _ in range(max_tries):
        n = int(rng.integers(5, 13))
        ang = (np.arange(n) + rng.uniform(-0.35, 0.35, n)) * 2 * np.pi / n
        rad = rng.uniform(0.35, 1.0, n)
        pts = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
        pts *= np.array([rng.uniform(1.0, 3.5), 1.0])
        rot = rng.uniform(0, 2 * np.pi)
        pts = pts @ np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]).T
        closed = np.vstack([pts, pts[:1]])
        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
        dense = CubicSpline(u, closed, bc_type="periodic")(np.linspace(0, u[-1], 3000, endpoint=False))
        length = float(np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1).sum())
        dense *= rng.uniform(*length_range) / length
        centre = _resample(dense, ds)
        m = len(centre)
        s = np.arange(m) * ds
        w0 = rng.uniform(*width_range)
        width = w0 * (1 + rng.uniform(0, 0.12) * np.sin(2 * np.pi * rng.integers(1, 4) * s / (m * ds)
                                                         + rng.uniform(0, 2 * np.pi)))
        kappa = np.abs(_curvature(centre, 5))
        if 1.0 / kappa.max() < max(width.max() / 2 + 0.12, 0.75):
            continue
        # walls of non-adjacent stretches must not touch
        gap = width.max() + 0.1
        pairs = cKDTree(centre).query_pairs(gap, output_type="ndarray")
        if len(pairs):
            sep = np.abs(pairs[:, 0] - pairs[:, 1])
            sep = np.minimum(sep, m - sep) * ds
            if np.any(sep > np.pi / 2 * gap + 0.5):
                continue
        d = np.roll(centre, -1, axis=0) - np.roll(centre, 1, axis=0)
        nrm = np.column_stack([-d[:, 1], d[:, 0]])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        left = centre + (width / 2)[:, None] * nrm
        right = centre - (width / 2)[:, None] * nrm
        lines = []
        for line in (left, right):
            line = _resample(line, rng.uniform(*point_spacing))
            line = line + rng.normal(0, 0.01, line.shape)          # mapping noise
            lines.append(np.roll(line, int(rng.integers(len(line))), axis=0))
        if rng.random() < 0.5:
            lines.reverse()
        return lines[0].astype(np.float32), lines[1].astype(np.float32)
    outer, inner, _ = random_track(rng)
    return outer.astype(np.float32), inner.astype(np.float32)


@dataclass
class _Track:
    name: str
    raw_outer: np.ndarray
    raw_inner: np.ndarray
    outer: np.ndarray            # prepared (policy observation)
    inner: np.ndarray
    model: TrackModel            # centre-line, progress / lap counting
    walls: cKDTree               # dense true walls, collisions
    spawn: Optional[tuple] = None


def load_map(path: str):
    """(outer, inner) raw (N, 2) boundaries from a saved map dir."""
    return (np.load(os.path.join(path, "outer_line.npy"))[:, :2].astype(np.float32),
            np.load(os.path.join(path, "inner_line.npy"))[:, :2].astype(np.float32))


# body: circles along the car (rear-axle frame), radius ~ half the 0.27 m body width
_BODY_X = np.array([-0.05, 0.15, 0.35])
_BODY_R = 0.14


class BicycleTrackEnv(gym.Env):
    """TrackEnv's observation/action/reward on a simulated kinematic bicycle.

    Shared with TrackEnv (same meaning): boundary_spacing, boundary_smooth_window, n_ahead,
    max_steps (10 Hz steps, scaled by policy_hz / 10), policy_hz, safety, stuck_grace_s.
    TrackEnv's sim-only kwargs (sim_host, sim_port, shm_prefix, ...) are accepted and ignored,
    so ``BicycleTrackEnv(**ENV_KWARGS)`` works with rl/train.py's ENV_KWARGS.

    Bicycle-only:
      map_dirs: saved maps (outer_line.npy/inner_line.npy) used as training tracks.
      p_map: chance an episode uses one of map_dirs instead of a generated track.
      p_reverse: chance a track is driven the other way round.
      p_spawn: chance a map episode starts at the sim's respawn pose instead of anywhere.
      start_lateral / start_yaw: uniform +- start offset (m) / heading error (rad).
      obs_noise: (position m, heading rad, speed m/s) Gaussian sigmas on the observed pose.
      latency: (min, max) s from action to actuator, drawn per episode.
      randomize_dynamics: small per-episode spread on speed gain, accel, wheelbase, grip.
      checkpoint_spacing: m of new progress per emulated checkpoint (0 = none).
      plant: SimPlantConfig overrides. vehicle_params: yaml path. dt: plant step (s).
    load_boundaries()/load_boundaries_from_path() pin one track, like TrackEnv.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        boundary_spacing: float = 0.4,
        boundary_smooth_window: int = 30,
        n_ahead: int = 10,
        max_steps: int = 2000,
        policy_hz: float = 10.0,
        safety: Optional[dict] = None,
        stuck_grace_s: float = 1.0,
        map_dirs: Sequence[str] = DEFAULT_MAPS,
        p_map: float = 0.3,
        p_reverse: float = 0.3,
        p_spawn: float = 0.3,
        start_lateral: float = 0.25,
        start_yaw: float = 0.2,
        obs_noise: Sequence[float] = (0.02, 0.01, 0.03),
        latency: Sequence[float] = (0.03, 0.08),
        randomize_dynamics: bool = True,
        checkpoint_spacing: float = 2.0,
        plant: Optional[dict] = None,
        vehicle_params: str = VEHICLE_PARAMS,
        dt: float = 0.01,
        **sim_only,
    ) -> None:
        super().__init__()
        unknown = set(sim_only) - set(_SIM_ONLY_KWARGS)
        if unknown:
            raise TypeError(f"unexpected kwargs {sorted(unknown)}")

        self._n_ahead = int(n_ahead)
        self._spacing = float(boundary_spacing)
        self._smooth_window = int(boundary_smooth_window)
        self._max_steps = max_steps * policy_hz / 10
        self._policy_hz = float(policy_hz)
        self._policy_dt = 1.0 / policy_hz
        self._dt = float(dt)
        self._substeps = max(int(round(self._policy_dt / self._dt)), 1)
        self._stuck_grace = int(round(stuck_grace_s * policy_hz))
        self.safety = asdict(SafetyConfig(**(safety or {})))

        self._map_dirs = list(map_dirs or [])
        self._p_map = float(p_map) if self._map_dirs else 0.0
        self._p_reverse = float(p_reverse)
        self._p_spawn = float(p_spawn)
        self._start_lateral = float(start_lateral)
        self._start_yaw = float(start_yaw)
        self._obs_noise = tuple(float(x) for x in obs_noise)
        self._latency = (float(latency[0]), float(latency[1]))
        self._randomize = bool(randomize_dynamics)
        self._checkpoint_spacing = float(checkpoint_spacing)
        self._plant_cfg = SimPlantConfig(**(plant or {}))
        self._params = VehicleParams.load(vehicle_params)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim(n_ahead),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),   # [steering, throttle]
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._fixed: Optional[_Track] = None          # load_boundaries()
        self._map_cache: dict = {}                    # (path, reversed) -> _Track
        self._track: Optional[_Track] = None
        # TrackEnv's names, for code that reads them (parity tests, logging)
        self._outer: Optional[np.ndarray] = None
        self._inner: Optional[np.ndarray] = None

        self._car: Optional[SimPlant] = None
        self._sup: Optional[SafetySupervisor] = None
        self._cmd_queue: deque = deque()
        self._cmd: Optional[Command] = None
        self._cmd_t: Optional[float] = None
        self._t = 0.0
        self._step_count = 0
        self._stuck_steps = 0
        self._stuck_grace_steps = 0
        self._hint: Optional[int] = None
        self._s_prev = 0.0
        self._progress = 0.0
        self._best_progress = 0.0
        self._checkpoints = 0
        self._lat = 0.05
        self._noise = True

    @property
    def policy_hz(self) -> float:
        return self._policy_hz

    @property
    def track(self) -> Optional[_Track]:
        return self._track

    # tracks

    def _make_track(self, name: str, outer: np.ndarray, inner: np.ndarray, reverse: bool = False,
                    spawn: Optional[tuple] = None) -> _Track:
        outer = np.asarray(outer, dtype=np.float32)[:, :2]
        inner = np.asarray(inner, dtype=np.float32)[:, :2]
        if reverse:
            outer, inner = outer[::-1].copy(), inner[::-1].copy()
            spawn = None if spawn is None else (spawn[0], spawn[1], spawn[2] + np.pi)
        walls = np.vstack([_resample(outer.astype(np.float64), 0.02), _resample(inner.astype(np.float64), 0.02)])
        return _Track(
            name=name + ("_rev" if reverse else ""),
            raw_outer=outer, raw_inner=inner,
            outer=prepare_boundary(outer, self._smooth_window, self._spacing),
            inner=prepare_boundary(inner, self._smooth_window, self._spacing),
            model=TrackModel(outer, inner),
            walls=cKDTree(walls),
            spawn=spawn,
        )

    def load_boundaries(self, outer: np.ndarray, inner: np.ndarray, spawn: Optional[tuple] = None) -> None:
        """Pin raw (N, 2) boundaries (in the driving direction) for every episode, like TrackEnv."""
        self._fixed = self._make_track("fixed", outer, inner, spawn=spawn)
        self._set_track(self._fixed)

    def load_boundaries_from_path(self, path: str) -> None:
        """path is a dir with outer_line.npy + inner_line.npy (spawn = the sim's for AutoDRIVE maps)."""
        spawn = AUTODRIVE_SPAWN if "autodrive" in os.path.basename(os.path.normpath(path)) else None
        self.load_boundaries(*load_map(path), spawn=spawn)

    def _set_track(self, track: _Track) -> None:
        self._track = track
        self._outer, self._inner = track.outer, track.inner

    def _map_track(self, path: str, reverse: bool) -> _Track:
        key = (path, reverse)
        if key not in self._map_cache:
            spawn = AUTODRIVE_SPAWN if "autodrive" in os.path.basename(os.path.normpath(path)) else None
            self._map_cache[key] = self._make_track(os.path.basename(os.path.normpath(path)),
                                                    *load_map(path), reverse=reverse, spawn=spawn)
        return self._map_cache[key]

    def _pick_track(self, options: dict) -> _Track:
        rng = self.np_random
        if "track" in options:
            t = options["track"]
            if isinstance(t, _Track):
                return t
            if isinstance(t, str):
                return self._map_track(t, bool(options.get("reverse", False)))
            return self._make_track("custom", t[0], t[1], reverse=bool(options.get("reverse", False)))
        if self._fixed is not None:
            return self._fixed
        reverse = bool(rng.random() < self._p_reverse)
        if rng.random() < self._p_map:
            return self._map_track(self._map_dirs[int(rng.integers(len(self._map_dirs)))], reverse)
        return self._make_track("random", *generate_track(rng), reverse=reverse)

    # gym API

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """options: track (_Track | map dir | (outer, inner)), reverse (bool),
        start ("spawn" | "random" | (x, y, yaw)), noise (bool, default True)."""
        super().reset(seed=seed)
        options = dict(options or {})
        rng = self.np_random
        self._set_track(self._pick_track(options))
        track = self._track
        self._noise = bool(options.get("noise", True))

        start = options.get("start")
        if start is None:
            start = "spawn" if (track.spawn is not None and rng.random() < self._p_spawn) else "random"
        if isinstance(start, str) and start == "spawn" and track.spawn is not None:
            x, y, yaw = track.spawn
        elif isinstance(start, str):
            m = track.model
            i = int(rng.integers(len(m.points))) if start == "random" else 0
            h = m.heading[i]
            off = rng.uniform(-1, 1) * min(self._start_lateral, max(m.half_width[i] - 0.3, 0.0))
            x, y = m.points[i] + off * np.array([-np.sin(h), np.cos(h)])
            yaw = h
        else:
            x, y, yaw = start
        if isinstance(start, str):
            yaw += rng.uniform(-1, 1) * self._start_yaw

        cfg = self._plant_cfg
        params = self._params
        if self._randomize:
            cfg = replace(cfg, speed_per_throttle=cfg.speed_per_throttle * rng.uniform(0.95, 1.05),
                          max_accel=cfg.max_accel * rng.uniform(0.8, 1.2),
                          coast_decel=cfg.coast_decel * rng.uniform(0.8, 1.2),
                          a_lat_max=cfg.a_lat_max * rng.uniform(0.85, 1.15),
                          steer_gain=cfg.steer_gain * rng.uniform(0.95, 1.05))
            params = replace(params, wheelbase=params.wheelbase * rng.uniform(0.97, 1.03))
        self._car = SimPlant(params, cfg, float(x), float(y), float(yaw))
        self._sup = SafetySupervisor(SafetyConfig(**self.safety))
        self._lat = float(rng.uniform(*self._latency))
        self._cmd_queue.clear()
        self._cmd, self._cmd_t = None, None
        self._t = 0.0

        self._step_count = 0
        self._stuck_grace_steps = self._stuck_grace
        self._stuck_steps = 0
        pr = track.model.project(self._car.x, self._car.y)
        self._hint, self._s_prev = pr.idx, pr.s
        self._progress = self._best_progress = 0.0
        self._checkpoints = 0
        return self._observe(), {"track": track.name}

    def step(self, action: np.ndarray):
        assert self._car is not None, "Call reset() before step()"
        steering, throttle = float(action[0]), float(action[1])
        # TrackEnv's path: signed -> ControlBuffer [0, 1] (forward only) -> ActuatorWorker signed
        s_cmd, t_cmd = unit_to_signed(*signed_to_unit(steering, max(throttle, 0.0)))
        self._cmd_queue.append((self._t + self._lat, Command(s_cmd, t_cmd)))

        car, sup = self._car, self._sup
        for _ in range(self._substeps):
            while self._cmd_queue and self._cmd_queue[0][0] <= self._t + 1e-9:
                _, self._cmd = self._cmd_queue.popleft()
                self._cmd_t = self._t
            out = sup.step(self._t, self._cmd, self._cmd_t, armed=True, speed=car.v)
            car.step(out, self._t, self._dt)
            self._t += self._dt
        self._step_count += 1

        speed = float(car.v)
        pos = np.array([car.x, car.y], dtype=np.float32)
        track = self._track

        # progress along the centre-line (the sim's lap/checkpoint counters)
        pr = track.model.project(car.x, car.y, hint=self._hint)
        self._hint = pr.idx
        L = track.model.length
        self._progress += (pr.s - self._s_prev + L / 2) % L - L / 2
        self._s_prev = pr.s
        new_checkpoints = 0
        if self._progress > self._best_progress:
            self._best_progress = self._progress
            if self._checkpoint_spacing > 0:
                n = int(self._best_progress // self._checkpoint_spacing)
                new_checkpoints, self._checkpoints = n - self._checkpoints, n
        lap_completed = bool(self._progress >= L)

        # wall contact of the body circles (the sim's collision counter)
        c, s = np.cos(car.yaw), np.sin(car.yaw)
        body = np.column_stack([car.x + _BODY_X * c, car.y + _BODY_X * s])
        d_wall = float(track.walls.query(body)[0].min())
        collided = bool(d_wall < _BODY_R or abs(pr.lateral_error) > track.model.half_width[pr.idx] + 0.3)

        if self._stuck_grace_steps > 0:
            self._stuck_grace_steps -= 1
            self._stuck_steps = 0
        elif speed < 0.1:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        stuck = self._stuck_steps >= 10

        obs = self._observe()

        # TrackEnv's reward
        reward = speed * 0.5
        reward += new_checkpoints * 2.0
        if lap_completed:
            reward += 2.0
        if collided or stuck:
            reward -= 20.0
        d_outer = float(np.linalg.norm(self._outer - pos, axis=1).min())
        d_inner = float(np.linalg.norm(self._inner - pos, axis=1).min())
        reward -= 1 / max(min(d_outer, d_inner), 0.05) * 0.1

        terminated = collided or lap_completed or stuck
        truncated = self._step_count >= self._max_steps
        return obs, float(reward), bool(terminated), bool(truncated), {
            "speed": speed,
            "collided": collided,
            "stuck": stuck,
            "lap_completed": lap_completed,
            "checkpoints": self._checkpoints,
            "d_outer": d_outer,
            "d_inner": d_inner,
            "progress": float(self._progress),
            "lap_time": self._t if lap_completed else None,
            "xtrack": pr.lateral_error,
            "steering": float(sup.output.steering),
            "throttle": float(sup.output.throttle),
            "track": track.name,
        }

    # internals

    def true_pose(self) -> tuple[float, float, float, float]:
        """(x, y, yaw, v) of the plant."""
        c = self._car
        return float(c.x), float(c.y), float(c.yaw), float(c.v)

    def _observe(self) -> np.ndarray:
        car = self._car
        x, y, yaw, v = car.x, car.y, car.yaw, car.v
        if self._noise and any(self._obs_noise):
            rng = self.np_random
            sp, sy, sv = self._obs_noise
            x, y = x + rng.normal(0, sp), y + rng.normal(0, sp)
            yaw, v = yaw + rng.normal(0, sy), v + rng.normal(0, sv)
        return self._build_obs(np.array([x, y], dtype=np.float32), float(yaw), float(v))

    def _build_obs(self, pos: np.ndarray, heading: float, speed: float) -> np.ndarray:
        """Same as TrackEnv._build_obs."""
        assert self._outer is not None and self._inner is not None
        return track_observation(pos, heading, speed, self._outer, self._inner, self._n_ahead)
