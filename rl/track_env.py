"""
Gymnasium env for RL training against the AutoDRIVE simulator.

Usage: construct TrackEnv (this creates the shared buffers but doesn't start
any workers), call env.start() to spawn ZMQSensorWorker + ActuatorWorker,
then load_boundaries_from_path("rl/boundaries") or load_boundaries(outer,
inner). After that it's a normal reset()/step() loop. close() kills the
workers and frees the shared memory.

Observation, (n_ahead+1)*4 + 1 floats in the vehicle-local frame:
    outer[0..n_ahead] (x, y)   closest point + n_ahead ahead
    inner[0..n_ahead] (x, y)
    speed (m/s)

Action is [steering in -1..1, throttle in 0..1].

Reward per step:
    +0.5 * speed
    +2 per checkpoint, +2 on finishing a lap
    -20 on collision or getting stuck
    -0.1 / max(d_wall, 0.05), d_wall = distance to the nearest boundary (m)

Episode ends (terminated) on a wall hit, a completed lap, or being stuck
(< 0.1 m/s for 10 steps in a row). The collision counter comes from the sim
over ZMQ. Truncation happens at max_steps, scaled by policy_hz / 10.
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

# AutoDRIVE world is Z-up, X-forward, Y-left, so the ground plane is pose
# axes (0, 1)
_BEV_A0, _BEV_A1 = 0, 1


class TrackEnv(gym.Env):
    """AutoDRIVE sim over ZMQ, as a Gymnasium env.

    Parameters
    ----------
    zmq_host, zmq_sensor_port, zmq_control_port
        Where the autodrive_bridge is. Sensor data comes in on 5555,
        controls go out on 5556 by default.
    boundary_spacing : float
        Resample spacing along the boundary polylines, metres.
    boundary_smooth_window : int
        Moving-average window (in points) applied before resampling.
    n_ahead : int
        Boundary points ahead of the closest one to put in the obs, so each
        side contributes n_ahead + 1 points.
    max_steps : int
        Episode length in 10 Hz steps. The real limit is
        max_steps * policy_hz / 10, which keeps episode duration in seconds
        the same whatever policy_hz is.
    policy_hz : float
        After each action the env sleeps 1/policy_hz and then reads whatever
        state ZMQSensorWorker has in the meantime.
    vis_zmq_port : int or None
        PUSH socket streaming the observed boundary points to
        rl/vis_obs_ros2.py. None turns it off.
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

        # optional: stream obs to RViz
        self._vis_sock = None
        if vis_zmq_port is not None:
            import zmq as _zmq
            _ctx = _zmq.Context.instance()
            self._vis_sock = _ctx.socket(_zmq.PUSH)
            self._vis_sock.setsockopt(_zmq.SNDHWM, 1)
            self._vis_sock.bind(f"tcp://*:{vis_zmq_port}")

        # buffers are created in the parent, workers attach to them
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass  # already set

        self._pose_buf = PoseBuffer(slots=32, name="rl_pose_buf", create=True)
        self._ctrl_buf = ControlBuffer(slots=32, name="rl_ctrl_buf", create=True)
        # ZMQSensorWorker wants frame + imu buffers even though we ignore them
        self._frame_buf = FrameBuffer(
            slots=4, name="rl_frame_buf", create=True,
            max_payload=16 + 360 * 640 * 7,  # 360x640 RGB + float32 depth
        )
        self._imu_buf = IMUBuffer(slots=32, name="rl_imu_buf", create=True)

        # reset() sets this, ActuatorWorker clears it once the reset is sent
        self._reset_event = mp.Event()
        # ZMQSensorWorker mirrors the sim's counters into these
        self._collision_count = mp.Value('i', 0)
        self._lap_count = mp.Value('i', 0)
        self._checkpoint_count = mp.Value('i', 0)

        # not started until env.start()
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

        obs_size = (n_ahead + 1) * 4 + 1  # (outer + inner) * xy + speed
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
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
        self._last_pose_seq: int = 0
        self._prev_collisions: int = 0
        self._prev_laps: int = 0
        self._prev_checkpoints: int = 0
        self._collision_grace_steps: int = 0  # collisions ignored for this many steps after reset
        self._stuck_steps: int = 0  # consecutive steps under 0.1 m/s

        self._processes: list[mp.Process] = []

    def start(self) -> None:
        """Spawn the sensor + actuator processes. Call once, before training."""
        for worker in (self._sensor_worker, self._actuator_worker):
            p = mp.Process(target=worker.run, name=worker.name, daemon=True)
            p.start()
            self._processes.append(p)

    def load_boundaries(self, outer: np.ndarray, inner: np.ndarray) -> None:
        """Takes raw (N, 2) boundaries from MappingWorker, smooths and resamples them."""
        self._outer = resample_uniform(smooth_boundary(outer, self._smooth_window), self._spacing)
        self._inner = resample_uniform(smooth_boundary(inner, self._smooth_window), self._spacing)

    def load_boundaries_from_path(self, path: str) -> None:
        """path is a dir with outer_line.npy + inner_line.npy, e.g. rl/boundaries"""
        import os
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

        # tell the sim to put the car back at the start
        self._reset_event.set()
        time.sleep(0.3)  # give it time to reposition and flush stale poses

        pos, heading, _ = self._wait_for_pose()

        # extra settle so the collision counter stops moving after the respawn
        time.sleep(0.1)

        self._step_count = 0
        # read counters only after settling, otherwise stale collisions leak in
        self._prev_collisions = self._collision_count.value
        self._prev_laps = self._lap_count.value
        self._prev_checkpoints = self._checkpoint_count.value
        self._collision_grace_steps = 3
        self._stuck_steps = 0

        obs = self._build_obs(pos, heading, speed=0.0)
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

        result = self._read_pose()
        if result is None:
            # lost the pose: zero obs, no reward, but don't end the episode
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return zero_obs, 0.0, False, False, {}

        pos, heading, speed = result

        # stuck = not moving, e.g. pinned against a wall
        if speed < 0.1:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        stuck = self._stuck_steps >= 10

        obs = self._build_obs(pos, heading, speed)

        reward = speed * 0.5

        self._step_count += 1

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
            self._prev_collisions = current_collisions
        else:
            collided = current_collisions > self._prev_collisions
            self._prev_collisions = current_collisions
        if collided or stuck:
            reward -= 20.0

        # centring penalty, blows up near either wall
        assert self._outer is not None and self._inner is not None
        d_outer = float(np.linalg.norm(self._outer - pos, axis=1).min())
        d_inner = float(np.linalg.norm(self._inner - pos, axis=1).min())
        reward -= 1/max(min(d_outer, d_inner), 0.05) * 0.1

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

    def _read_pose(self) -> Optional[tuple[np.ndarray, float, float]]:
        """(pos, heading, speed) from the latest pose, or None."""
        result = self._pose_buf.read_latest()
        if result is None:
            return None
        _, pose, _, speed = result
        pos = np.array([pose[_BEV_A0, 3], pose[_BEV_A1, 3]], dtype=np.float32)
        heading = float(np.arctan2(pose[_BEV_A1, 0], pose[_BEV_A0, 0]))
        return pos, heading, float(speed)

    def _wait_for_pose(self, timeout: float = 5.0) -> tuple[np.ndarray, float, float]:
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

        # world-frame points, only used for the vis stream
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
                pass  # nobody listening yet, drop it

        bnd = boundary_obs(pos, heading, self._outer, self._inner, self._n_ahead)
        return np.append(bnd, speed).astype(np.float32)
