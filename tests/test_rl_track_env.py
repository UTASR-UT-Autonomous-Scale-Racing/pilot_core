"""TrackEnv against a fake simulator (local ZMQ sockets): reset handshake,
fresh-pose wait, stale-pose truncation, speed-capped actuator, clean close."""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
pytest.importorskip("gymnasium")

from control.sim import oval_track  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeSim:
    """Publishes poses at ~50 Hz on the sensor port, records the commands."""

    def __init__(self, sensor_port: int, control_port: int, x: float, y: float, yaw: float):
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://127.0.0.1:{sensor_port}")
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sub.connect(f"tcp://127.0.0.1:{control_port}")
        self.pose = (x, y, yaw)
        self.speed = 0.0
        self.publishing = True
        self.commands: list[dict] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            if self.publishing:
                x, y, yaw = self.pose
                self.pub.send_string(json.dumps({
                    "position": [x, y, 0.0],
                    "orientation_quat": [0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
                    "speed": self.speed, "collision_count": 0, "lap_count": 0, "checkpoint_count": 0,
                }))
            while self.sub.poll(0):
                self.commands.append(json.loads(self.sub.recv_string()))
            time.sleep(0.02)

    def close(self):
        self._stop.set()
        self._t.join(2)
        self.pub.close(linger=0)
        self.sub.close(linger=0)
        self.ctx.term()


def test_track_env_against_fake_sim():
    from multiprocessing import shared_memory

    from rl.track_env import TrackEnv

    sp, cp = _free_port(), _free_port()
    outer, inner, centre = oval_track()
    prefix = f"te_{uuid.uuid4().hex[:6]}"
    env = TrackEnv(zmq_host="127.0.0.1", zmq_sensor_port=sp, zmq_control_port=cp, n_ahead=5,
                   policy_hz=20.0, shm_prefix=prefix, stale_timeout=0.3, stuck_grace_s=1.0,
                   safety={"speed_cap": 1.5, "cap_margin": 0.2, "speed_per_throttle": None,
                           "command_timeout": 0.5})
    sim = FakeSim(sp, cp, float(centre[0, 0]), float(centre[0, 1]), np.pi / 2)
    try:
        env.load_boundaries(outer, inner)
        env.start()
        time.sleep(1.5)                          # workers up, sockets connected
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
        assert env._last_pose_seq > 0
        time.sleep(0.2)
        assert any(c.get("reset") for c in sim.commands), "reset never reached the sim"

        # stationary car inside the grace period: not stuck
        for _ in range(10):
            obs, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
            assert not trunc and not info["stuck"]

        # speed above the cap: the actuator's supervisor zeroes forward throttle
        # (training runs the same speed taper as the car)
        sim.speed = 2.0
        for _ in range(6):
            env.step(np.array([0.0, 1.0], np.float32))
        time.sleep(0.1)
        assert sim.commands[-1]["throttle"] == pytest.approx(0.0)
        # below the taper band the full command goes through
        sim.speed = 0.5
        for _ in range(10):
            env.step(np.array([0.0, 1.0], np.float32))
        time.sleep(0.1)
        assert sim.commands[-1]["throttle"] > 0.9

        # pose stream stalls -> truncated, not zero observations
        sim.publishing = False
        time.sleep(0.1)
        env.step(np.array([0.0, 1.0], np.float32))    # may still consume the last published pose
        t0 = time.monotonic()
        obs2, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
        assert trunc and not term and info.get("stale_pose")
        assert r == 0.0 and np.array_equal(obs2, env._last_obs)
        assert time.monotonic() - t0 < 1.0
    finally:
        procs = list(env._processes)
        env.close()
        sim.close()
    # graceful stop (exit code 0), not terminate()
    assert procs and all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
    for suffix in ("pose", "ctrl", "state"):
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=f"{prefix}_{suffix}")
