"""TrackEnv against the fake AutoDRIVE sim (tests/fake_autodrive.py): reset handshake,
fresh-pose wait, stale-pose truncation, speed-capped actuator, clean close."""

from __future__ import annotations

import socket
import time
import uuid

import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("gymnasium")

from control.sim import oval_track  # noqa: E402
from tests.fake_autodrive import FakeAutoDRIVE  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_track_env_against_fake_sim():
    from multiprocessing import shared_memory

    from rl.track_env import TrackEnv

    port = _free_port()
    outer, inner, _ = oval_track()
    prefix = f"te_{uuid.uuid4().hex[:6]}"
    env = TrackEnv(sim_host="127.0.0.1", sim_port=port, n_ahead=5,
                   policy_hz=20.0, shm_prefix=prefix, stale_timeout=0.3, stuck_grace_s=1.0,
                   safety={"speed_cap": 1.5, "cap_margin": 0.2, "speed_per_throttle": None,
                           "command_timeout": 0.5})
    sim = FakeAutoDRIVE(port, rate_hz=25.0, lidar=False, camera=False)
    try:
        env.load_boundaries(outer, inner)
        env.start()
        sim.start()          # after the env's workers are forked
        deadline = time.monotonic() + 10
        while not (sim.stats.connected and sim.stats.sent > 5):
            assert time.monotonic() < deadline, "sim never connected"
            time.sleep(0.05)
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
        assert env._last_pose_seq > 0
        time.sleep(0.2)
        assert sim.stats.resets == 1, "reset never reached the sim"
        assert any(c["V1 Reset"] == "True" for c in sim.stats.commands)

        # stationary car inside the grace period: not stuck
        sim.speed_override = 0.0
        for _ in range(10):
            obs, r, term, trunc, info = env.step(np.array([0.0, 1.0], np.float32))
            assert not trunc and not info["stuck"]

        # speed above the cap: the actuator's supervisor zeroes forward throttle
        # (training runs the same speed taper as the car)
        sim.speed_override = 2.0
        for _ in range(6):
            env.step(np.array([0.0, 1.0], np.float32))
        time.sleep(0.1)
        assert float(sim.stats.commands[-1]["V1 Throttle"]) == pytest.approx(0.0)
        # below the taper band the full command goes through
        sim.speed_override = 0.5
        for _ in range(10):
            env.step(np.array([0.0, 1.0], np.float32))
        time.sleep(0.1)
        assert float(sim.stats.commands[-1]["V1 Throttle"]) > 0.9

        # pose stream stalls -> truncated, not zero observations
        sim.paused = True
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
    for suffix in ("pose", "ctrl", "act", "state"):
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=f"{prefix}_{suffix}")


def test_reset_waits_out_the_respawn_pause():
    # the stock sim streams a few poses after a reset, then pauses ~2 s while it
    # respawns the car: reset() must not return into that pause, or every episode
    # ends on its first step as a stale pose
    from rl.track_env import TrackEnv

    port = _free_port()
    outer, inner, _ = oval_track()
    env = TrackEnv(sim_host="127.0.0.1", sim_port=port, n_ahead=5, policy_hz=20.0,
                   shm_prefix=f"te_{uuid.uuid4().hex[:6]}", stale_timeout=0.3, reset_timeout=10.0,
                   safety={"speed_cap": 1.5, "cap_margin": 0.2, "speed_per_throttle": None,
                           "command_timeout": 0.5})
    sim = FakeAutoDRIVE(port, rate_hz=25.0, lidar=False, camera=False)
    sim.reset_pause = (0.6, 1.0)        # begins after the respawn's first few poses
    try:
        env.load_boundaries(outer, inner)
        env.start()
        sim.start()
        deadline = time.monotonic() + 10
        while not (sim.stats.connected and sim.stats.sent > 5):
            assert time.monotonic() < deadline, "sim never connected"
            time.sleep(0.05)
        for _ in range(2):
            env.reset()
            for _ in range(5):
                _, _, term, trunc, info = env.step(np.array([0.0, 0.5], np.float32))
                assert not trunc and not info.get("stale_pose")
    finally:
        env.close()
        sim.close()
