"""Gymnasium env for RL training against the AutoDRIVE simulator.

TrackEnv() -> env.start() -> load_boundaries_from_path(...) -> reset()/step(); close() frees shm.
Obs: (n_ahead+1)*4 outer/inner boundary points (vehicle frame) + speed. Action: [steer -1..1, throttle 0..1].
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import uuid
from dataclasses import asdict
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("gymnasium is required: pip install gymnasium")

from control.boundary import obs_dim, prepare_boundary, track_observation
from control.safety import SafetyConfig
from shared_buffer import ControlBuffer, PoseBuffer, StateBuffer
from workers import ActuatorWorker, ZMQSensorWorker
from workers.control_worker import signed_to_unit

# AutoDRIVE world is Z-up, X-forward, Y-left, so the ground plane is pose
# axes (0, 1)
_BEV_A0, _BEV_A1 = 0, 1


class TrackEnv(gym.Env):
    """AutoDRIVE sim over ZMQ, as a Gymnasium env.

    zmq_host/zmq_sensor_port/zmq_control_port: autodrive_bridge (5555 in, 5556 out).
    boundary_spacing (m), boundary_smooth_window (points): boundary resampling.
    n_ahead: boundary points ahead of the closest one per side in the obs.
    max_steps: episode length in 10 Hz steps (scaled by policy_hz / 10).
    policy_hz: action rate; step() sleeps 1/policy_hz then waits for a new pose.
    safety: SafetyConfig overrides (car's ``safety`` section), None = defaults.
    shm_prefix: shared-memory name prefix, random by default.
    stale_timeout / reset_timeout / stuck_grace_s: seconds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        zmq_host: str = "localhost",
        zmq_sensor_port: int = 5555,
        zmq_control_port: int = 5556,
        boundary_spacing: float = 0.4,
        boundary_smooth_window: int = 30,
        n_ahead: int = 10,
        max_steps: int = 2000,
        policy_hz: float = 10.0,
        safety: Optional[dict] = None,
        shm_prefix: Optional[str] = None,
        stale_timeout: float = 0.5,
        reset_timeout: float = 5.0,
        stuck_grace_s: float = 1.0,
    ) -> None:
        super().__init__()

        self._n_ahead = n_ahead
        self._spacing = boundary_spacing
        self._smooth_window = boundary_smooth_window
        self._max_steps = max_steps * policy_hz / 10
        self._policy_hz = float(policy_hz)
        self._policy_dt = 1.0 / policy_hz
        self._stale_timeout = float(stale_timeout)
        self._reset_timeout = float(reset_timeout)
        self._stuck_grace = int(round(stuck_grace_s * policy_hz))
        # full SafetyConfig, so it can be recorded next to the model
        self.safety = asdict(SafetyConfig(**(safety or {})))

        # buffers are created here, the workers attach to them
        prefix = shm_prefix or f"rl_{uuid.uuid4().hex[:8]}"
        self._pose_buf = PoseBuffer(slots=32, name=f"{prefix}_pose", create=True)
        self._ctrl_buf = ControlBuffer(slots=32, name=f"{prefix}_ctrl", create=True)
        # speed source for the actuator's speed cap, written by ZMQSensorWorker
        self._state_buf = StateBuffer(slots=32, name=f"{prefix}_state", create=True)
        self._buffers = (self._pose_buf, self._ctrl_buf, self._state_buf)

        # reset() sets this, ActuatorWorker clears it once the reset is sent
        self._reset_event = mp.Event()
        # ZMQSensorWorker mirrors the sim's counters into these
        self._collision_count = mp.Value('i', 0)
        self._lap_count = mp.Value('i', 0)
        self._checkpoint_count = mp.Value('i', 0)

        # not started until env.start(). No frame/IMU buffers: the env only
        # needs pose, speed and the counters, so image decoding is skipped
        self._sensor_worker = ZMQSensorWorker(
            name="rl_zmq_sensor",
            host=zmq_host,
            port=zmq_sensor_port,
            frame_buffer=None,
            imu_buffer=None,
            pose_buffer=self._pose_buf,
            state_buffer=self._state_buf,
            collision_counter=self._collision_count,
            lap_counter=self._lap_count,
            checkpoint_counter=self._checkpoint_count,
        )
        self._actuator_worker = ActuatorWorker(
            name="rl_actuator",
            input_buffer=self._ctrl_buf,
            zmq_enabled=True,
            zmq_port=zmq_control_port,
            reset_event=self._reset_event,
            state_buffer=self._state_buf,
            safety=self.safety,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim(n_ahead),), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),   # [steering, throttle]
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # set by load_boundaries()
        self._outer: Optional[np.ndarray] = None
        self._inner: Optional[np.ndarray] = None

        self._step_count: int = 0
        self._last_pose_seq: int = 0     # seq of the pose the last obs was built from
        self._last_pose_ts: float = 0.0  # its produced_ts (time.monotonic())
        self._last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self._prev_collisions: int = 0
        self._prev_laps: int = 0
        self._prev_checkpoints: int = 0
        self._collision_grace_steps: int = 0  # collisions ignored for this many steps after reset
        self._stuck_grace_steps: int = 0      # low speed not counted as stuck for this many steps
        self._stuck_steps: int = 0  # consecutive steps under 0.1 m/s

        self._processes: list[mp.Process] = []
        self._closed = False

    @property
    def policy_hz(self) -> float:
        return self._policy_hz

    def start(self) -> None:
        """Spawn the sensor + actuator processes. Call once, before training."""
        for worker in (self._sensor_worker, self._actuator_worker):
            p = mp.Process(target=worker.run, name=worker.name, daemon=True)
            p.start()
            self._processes.append(p)

    def load_boundaries(self, outer: np.ndarray, inner: np.ndarray) -> None:
        """Takes raw (N, 2) boundaries from MappingWorker, smooths and resamples them."""
        self._outer = prepare_boundary(outer, self._smooth_window, self._spacing)
        self._inner = prepare_boundary(inner, self._smooth_window, self._spacing)

    def load_boundaries_from_path(self, path: str) -> None:
        """path is a dir with outer_line.npy + inner_line.npy, e.g. rl/boundaries"""
        outer = np.load(os.path.join(path, "outer_line.npy"))[:, :2]
        inner = np.load(os.path.join(path, "inner_line.npy"))[:, :2]
        self.load_boundaries(outer, inner)

    # gym API

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        assert self._outer is not None, "Call load_boundaries() before reset()"

        # stop driving, then tell the sim to put the car back at the start
        self._ctrl_buf.write(*signed_to_unit(0.0, 0.0))
        self._reset_event.set()
        deadline = time.monotonic() + self._reset_timeout
        while self._reset_event.is_set():   # ActuatorWorker clears it when sent
            if time.monotonic() > deadline:
                raise TimeoutError("ActuatorWorker did not send the reset command (is it running?)")
            time.sleep(0.01)
        seq_at_reset = self._pose_buf.latest_seq()

        # give the sim time to reposition and the collision counter to stop
        # moving after the respawn
        time.sleep(0.4)

        # only a pose produced after the reset went out counts
        result = self._wait_for_pose(after_seq=seq_at_reset, timeout=self._reset_timeout)
        if result is None:
            raise TimeoutError("No fresh pose from the simulator after reset")
        pos, heading, speed = result

        self._step_count = 0
        # read counters only after settling, otherwise stale collisions leak in
        self._prev_collisions = self._collision_count.value
        self._prev_laps = self._lap_count.value
        self._prev_checkpoints = self._checkpoint_count.value
        self._collision_grace_steps = 3
        self._stuck_grace_steps = self._stuck_grace
        self._stuck_steps = 0

        obs = self._build_obs(pos, heading, speed)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._outer is not None, "Call load_boundaries() before step()"

        steering, throttle = float(action[0]), float(action[1])

        # ControlBuffer wants [0, 1] with 0.5 neutral.
        # steering: [-1, 1] (+1 = left) -> [0, 1] (1 = right)
        # throttle: [0, 1] -> [0.5, 1.0], forward only
        steer_norm, throttle_norm = signed_to_unit(steering, max(throttle, 0.0))
        self._ctrl_buf.write(steer_norm, throttle_norm)

        # ZMQSensorWorker keeps receiving while we sleep
        time.sleep(self._policy_dt)

        result = self._wait_for_pose(after_seq=self._last_pose_seq,
                                     deadline=self._last_pose_ts + self._stale_timeout)
        if result is None:
            # pose stream stalled: end the episode instead of feeding garbage
            self._ctrl_buf.write(*signed_to_unit(0.0, 0.0))
            return self._last_obs.copy(), 0.0, False, True, {"stale_pose": True}

        pos, heading, speed = result
        self._step_count += 1

        # stuck = not moving, e.g. pinned against a wall. Not counted right
        # after a reset while the car is still getting going
        if self._stuck_grace_steps > 0:
            self._stuck_grace_steps -= 1
            self._stuck_steps = 0
        elif speed < 0.1:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        stuck = self._stuck_steps >= 10

        obs = self._build_obs(pos, heading, speed)

        reward = speed * 0.5

        # checkpoints give a denser progress signal than laps alone
        current_checkpoints = self._checkpoint_count.value
        new_checkpoints = max(0, current_checkpoints - self._prev_checkpoints)
        reward += new_checkpoints * 2.0
        self._prev_checkpoints = current_checkpoints

        current_laps = self._lap_count.value
        lap_completed = current_laps > self._prev_laps
        self._prev_laps = current_laps
        if lap_completed:
            reward += 2.0

        # Collisions come from Unity's WallCollisionCounter over ZMQ. For the
        # first few steps after reset we ignore them, since the previous
        # episode's collision can still show up
        current_collisions = self._collision_count.value
        if self._collision_grace_steps > 0:
            self._collision_grace_steps -= 1
            collided = False
        else:
            collided = current_collisions > self._prev_collisions
        self._prev_collisions = current_collisions
        if collided or stuck:
            reward -= 20.0

        # centring penalty, blows up near either wall
        d_outer = float(np.linalg.norm(self._outer - pos, axis=1).min())
        d_inner = float(np.linalg.norm(self._inner - pos, axis=1).min())
        reward -= 1 / max(min(d_outer, d_inner), 0.05) * 0.1

        terminated = collided or lap_completed or stuck
        truncated = self._step_count >= self._max_steps

        return obs, reward, terminated, truncated, {
            "speed": speed,
            "collided": collided,
            "stuck": stuck,
            "lap_completed": lap_completed,
            "checkpoints": current_checkpoints,
            "d_outer": d_outer,
            "d_inner": d_inner,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # ask the workers to finish (the actuator sends neutral on the way out)
        for worker in (self._sensor_worker, self._actuator_worker):
            worker.stop()
        for p in self._processes:
            p.join(timeout=3.0)
        for p in self._processes:
            if p.is_alive():   # fallback only
                p.terminate()
                p.join(timeout=1.0)
        self._processes.clear()
        for buf in self._buffers:
            try:
                buf.unlink()
            except Exception:
                pass

    # internals

    def _wait_for_pose(
        self,
        after_seq: int,
        timeout: Optional[float] = None,
        deadline: Optional[float] = None,
    ) -> Optional[tuple[np.ndarray, float, float]]:
        """(pos, heading, speed) from the first pose with seq > after_seq, or
        None if none arrives before the deadline (or within timeout)."""
        if deadline is None:
            deadline = time.monotonic() + (timeout if timeout is not None else self._stale_timeout)
        while True:
            if self._pose_buf.latest_seq() > after_seq:
                result = self._pose_buf.read_latest()
                if result is not None and result[0].seq_no > after_seq:
                    header, pose, _, speed = result
                    self._last_pose_seq = header.seq_no
                    self._last_pose_ts = header.produced_ts
                    pos = np.array([pose[_BEV_A0, 3], pose[_BEV_A1, 3]], dtype=np.float32)
                    heading = float(np.arctan2(pose[_BEV_A1, 0], pose[_BEV_A0, 0]))
                    return pos, heading, float(speed)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)

    def _build_obs(self, pos: np.ndarray, heading: float, speed: float) -> np.ndarray:
        assert self._outer is not None and self._inner is not None

        obs = track_observation(pos, heading, speed, self._outer, self._inner, self._n_ahead)
        self._last_obs = obs
        return obs
