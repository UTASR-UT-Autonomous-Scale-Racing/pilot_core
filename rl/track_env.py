"""
rl/track_env.py — Gymnasium environment for RL training on the AutoDRIVE simulator.

Lifecycle
---------
1. Instantiate TrackEnv — creates shared buffers, does NOT start workers yet.
2. Call env.start() — spawns ZMQSensorWorker + ActuatorWorker as daemon processes.
3. Call env.load_boundaries_from_path("rl/boundaries") — loads, smooths, and resamples
   pre-saved .npy boundary files.  Or call load_boundaries(outer, inner) directly.
4. env.reset() / env.step(action) as a standard Gymnasium loop.
5. env.close() to stop workers and release shared memory.

Observation ((n_ahead+1)*4 + 1 floats, vehicle-local frame):
    outer[0..n_ahead] × (x, y)  — (n_ahead+1)*2 floats  (closest + n_ahead forward)
    inner[0..n_ahead] × (x, y)  — (n_ahead+1)*2 floats
    speed_mps                    — 1 float

Action (2 floats):
    [steering, throttle]  — steering in [-1, 1], throttle in [0, 1]

Reward:
    +forward velocity each step
    +speed each step
    +5 per checkpoint passed
    +10 on lap completion
    -5 on collision
    -small centering penalty (1/d_wall)
    -smoothness penalty on action deltas

Termination:
    terminated=True  on wall collision (ZMQ counter from simulator)
    terminated=True  on lap completion
    terminated=True  if stuck (throttle > 0.2 but speed < 0.1 for 3+ consecutive steps)
    truncated=True   after max_steps
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    raise ImportError("gymnasium is required: pip install gymnasium")

from shared_buffer import (
    ControlBuffer,
    FrameBuffer,
    IMUBuffer,
    PoseBuffer,
)
from workers import ActuatorWorker, ZMQSensorWorker
from workers.control_worker import signed_to_unit
from workers.boundary_resampler import boundary_obs, resample_uniform, smooth_boundary

# BEV axes for the AutoDRIVE simulator: world frame is Z-up, X-forward, Y-left.
# pose matrix columns: 0=X, 1=Y, 2=Z.  Horizontal plane = axes (0, 1).
_BEV_A0, _BEV_A1 = 0, 1


class TrackEnv(gym.Env):
    """Gymnasium environment wrapping the AutoDRIVE simulator via ZMQ.

    Parameters
    ----------
    zmq_host : str
        Hostname of the autodrive_bridge (default: localhost).
    zmq_sensor_port : int
        Port the bridge publishes sensor data on (default: 5555).
    zmq_control_port : int
        Port the bridge subscribes to control commands on (default: 5556).
    boundary_spacing : float
        Arc-length spacing (metres) used when resampling boundary polylines.
    n_ahead : int
        Number of boundary points ahead of the closest to include in obs.
        Total points per boundary = n_ahead + 1.
    max_steps : int
        Episode length before truncation.
    policy_hz : float
        Target frequency of the RL control loop.  The env sleeps for
        1/policy_hz seconds after writing each action, then reads the
        latest state accumulated by ZMQSensorWorker during that interval.
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
        vis_zmq_port: Optional[int] = 5558,
    ) -> None:
        super().__init__()

        self._n_ahead = n_ahead
        self._spacing = boundary_spacing
        self._smooth_window = boundary_smooth_window
        self._max_steps = max_steps * policy_hz/10
        self._policy_dt = 1.0 / policy_hz

        # Optional ZMQ publisher for RViz observation visualization
        self._vis_sock = None
        if vis_zmq_port is not None:
            import zmq as _zmq
            _ctx = _zmq.Context.instance()
            self._vis_sock = _ctx.socket(_zmq.PUSH)
            self._vis_sock.setsockopt(_zmq.SNDHWM, 1)
            self._vis_sock.bind(f"tcp://*:{vis_zmq_port}")

        # Shared buffers — created here (parent process), opened by workers
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass  # already set

        self._pose_buf = PoseBuffer(slots=32, name="rl_pose_buf", create=True)
        self._ctrl_buf = ControlBuffer(slots=32, name="rl_ctrl_buf", create=True)
        # ZMQSensorWorker requires frame + imu buffers even if we don't use them
        self._frame_buf = FrameBuffer(
            slots=4, name="rl_frame_buf", create=True,
            max_payload=16 + 360 * 640 * 7,  # enough for 360×640 RGB+depth
        )
        self._imu_buf = IMUBuffer(slots=32, name="rl_imu_buf", create=True)

        # Reset signal — set by env.reset(), cleared by ActuatorWorker after sending
        self._reset_event = mp.Event()
        # Telemetry counters — ZMQSensorWorker mirrors simulator values here
        self._collision_count = mp.Value('i', 0)
        self._lap_count = mp.Value('i', 0)
        self._checkpoint_count = mp.Value('i', 0)

        # Workers (not started yet — call env.start())
        self._sensor_worker = ZMQSensorWorker(
            name="rl_zmq_sensor",
            host=zmq_host,
            port=zmq_sensor_port,
            frame_buffer=self._frame_buf,
            imu_buffer=self._imu_buf,
            pose_buffer=self._pose_buf,
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
        )

        # Gymnasium spaces
        obs_size = (n_ahead + 1) * 4 + 1  # (11 outer + 11 inner) × 2 coords + speed
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),   # [steering, throttle]
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Resampled boundaries — set by load_boundaries()
        self._outer: Optional[np.ndarray] = None
        self._inner: Optional[np.ndarray] = None

        # Episode state
        self._step_count: int = 0
        self._last_pose_seq: int = 0
        self._prev_collisions: int = 0
        self._prev_laps: int = 0
        self._prev_checkpoints: int = 0
        self._prev_action: np.ndarray = np.zeros(2, dtype=np.float32)  # [steering, throttle]
        self._collision_grace_steps: int = 0  # ignore collisions for N steps after reset
        self._stuck_steps: int = 0            # consecutive steps with throttle>0.2 but speed~0

        self._processes: list[mp.Process] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn sensor and actuator worker processes.  Call once before training."""
        for worker in (self._sensor_worker, self._actuator_worker):
            p = mp.Process(target=worker.run, name=worker.name, daemon=True)
            p.start()
            self._processes.append(p)

    def load_boundaries(self, outer: np.ndarray, inner: np.ndarray) -> None:
        """Smooth, resample, and store track boundaries.

        Parameters
        ----------
        outer : np.ndarray
            (N, 2) raw outer boundary from MappingWorker.
        inner : np.ndarray
            (M, 2) raw inner boundary from MappingWorker.
        """
        self._outer = resample_uniform(smooth_boundary(outer, self._smooth_window), self._spacing)
        self._inner = resample_uniform(smooth_boundary(inner, self._smooth_window), self._spacing)

    def load_boundaries_from_path(self, path: str) -> None:
        """Load pre-saved boundaries from a directory containing outer_line.npy / inner_line.npy.

        Parameters
        ----------
        path : str
            Directory path, e.g. "rl/boundaries".
        """
        import os
        outer = np.load(os.path.join(path, "outer_line.npy"))[:, :2]
        inner = np.load(os.path.join(path, "inner_line.npy"))[:, :2]
        self.load_boundaries(outer, inner)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        assert self._outer is not None, "Call load_boundaries() before reset()"

        # Signal the simulator to reset the vehicle position
        self._reset_event.set()
        time.sleep(0.3)  # wait for simulator to reposition and flush stale poses

        # Block until a fresh pose arrives
        pos, heading, _ = self._wait_for_pose()

        # Additional settle time — lets collision counter stabilize after respawn
        time.sleep(0.1)

        self._step_count = 0
        # Read counters AFTER settle time to capture any stale collision events
        self._prev_collisions = self._collision_count.value
        self._prev_laps = self._lap_count.value
        self._prev_checkpoints = self._checkpoint_count.value
        self._prev_action = np.zeros(2, dtype=np.float32)
        # Grace period: ignore collisions for first N steps after reset
        self._collision_grace_steps = 3
        self._stuck_steps = 0

        # temporary hard code
        # self._policy_dt = 1/20.0

        obs = self._build_obs(pos, heading, speed=0.0)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._outer is not None, "Call load_boundaries() before step()"

        steering, throttle = float(action[0]), float(action[1])

        # ControlBuffer expects [0, 1] where 0.5 = neutral, 1.0 = full forward, 0.0 = full reverse
        # steering: [-1, 1] (+1 = left) → ControlBuffer [0, 1] (1 = right);
        # throttle: [0, 1] → [0.5, 1.0] (forward only)
        steer_norm, throttle_norm = signed_to_unit(steering, max(throttle, 0.0))
        self._ctrl_buf.write(steer_norm, throttle_norm)

        # Sleep for policy cadence — ZMQSensorWorker accumulates frames meanwhile
        time.sleep(self._policy_dt)

        # Read latest state
        result = self._read_pose()
        if result is None:
            # Pose lost — return zero obs, no reward, keep episode alive
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zero_obs, 0.0, False, False, {}

        pos, heading, speed = result

        # Stuck detection: throttle applied but car isn't moving → pinned against wall
        if speed < 0.1:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        stuck = self._stuck_steps >= 10

        obs = self._build_obs(pos, heading, speed)

        reward = speed * 0.5

        self._step_count += 1

        # Checkpoint reward — make these more significant to provide denser signal
        current_checkpoints = self._checkpoint_count.value
        new_checkpoints = max(0, current_checkpoints - self._prev_checkpoints)
        reward += new_checkpoints * 2.0
        self._prev_checkpoints = current_checkpoints

        # Lap completed = full loop bonus (reduced to lower variance)
        current_laps = self._lap_count.value
        lap_completed = current_laps > self._prev_laps
        self._prev_laps = current_laps
        if lap_completed:
            reward += 2.0  # was 100.0 — checkpoints now provide more of the signal

        # Collision — detected from Unity's WallCollisionCounter via ZMQ
        # Grace period: ignore collisions for first few steps after reset to avoid
        # detecting stale collision events from the previous episode
        current_collisions = self._collision_count.value
        if self._collision_grace_steps > 0:
            self._collision_grace_steps -= 1
            collided = False
            self._prev_collisions = current_collisions  # sync counter during grace
        else:
            collided = current_collisions > self._prev_collisions
            self._prev_collisions = current_collisions
        if collided or stuck:
            reward -= 20.0

        # Centering reward — min distance to either wall, larger = more centered
        assert self._outer is not None and self._inner is not None
        d_outer = float(np.linalg.norm(self._outer - pos, axis=1).min())
        d_inner = float(np.linalg.norm(self._inner - pos, axis=1).min())
        reward -= 1/max(min(d_outer, d_inner), 0.05) * 0.1

        # Action smoothness — penalize sudden steering/throttle changes
        steer_delta = abs(steering - float(self._prev_action[0]))
        throttle_delta = abs(throttle - float(self._prev_action[1]))
        # reward -= steer_delta * 0.1
        # reward -= throttle_delta * 0.05
        self._prev_action = np.array([steering, throttle], dtype=np.float32)

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
        for p in self._processes:
            p.terminate()
            p.join(timeout=2)
        for buf in (self._pose_buf, self._ctrl_buf, self._frame_buf, self._imu_buf):
            try:
                buf.unlink()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_pose(self) -> Optional[tuple[np.ndarray, float, float]]:
        """Read latest pose from PoseBuffer.  Returns (pos, heading, speed) or None."""
        result = self._pose_buf.read_latest()
        if result is None:
            return None
        _, pose, _, speed = result
        pos = np.array([pose[_BEV_A0, 3], pose[_BEV_A1, 3]], dtype=np.float32)
        heading = float(np.arctan2(pose[_BEV_A1, 0], pose[_BEV_A0, 0]))
        return pos, heading, float(speed)

    def _wait_for_pose(self, timeout: float = 5.0) -> tuple[np.ndarray, float, float]:
        """Block until a pose is available.  Raises TimeoutError if none arrives."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self._read_pose()
            if result is not None:
                return result
            time.sleep(0.02)
        raise TimeoutError("No pose received from simulator within timeout")

    def _build_obs(
        self, pos: np.ndarray, heading: float, speed: float
    ) -> np.ndarray:
        assert self._outer is not None and self._inner is not None

        # Extract world-frame obs points for optional visualization
        n_ahead = self._n_ahead
        idx_o = int(np.linalg.norm(self._outer - pos, axis=1).argmin())
        idx_i = int(np.linalg.norm(self._inner - pos, axis=1).argmin())
        n, m = len(self._outer), len(self._inner)
        o_world = self._outer[np.arange(idx_o, idx_o + n_ahead + 1) % n]
        i_world = self._inner[np.arange(idx_i, idx_i + n_ahead + 1) % m]

        if self._vis_sock is not None:
            import json
            import zmq as _zmq
            msg = json.dumps({
                "pos": pos.tolist(),
                "outer": o_world.tolist(),
                "inner": i_world.tolist(),
            })
            try:
                self._vis_sock.send_string(msg, flags=_zmq.NOBLOCK)
            except _zmq.Again:
                pass  # no receiver connected yet — drop frame

        bnd = boundary_obs(pos, heading, self._outer, self._inner, self._n_ahead)
        return np.append(bnd, speed).astype(np.float32)
