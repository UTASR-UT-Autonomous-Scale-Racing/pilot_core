"""Fail-safe behaviour outside the supervisor: depth controllers on bad depth,
worker exit codes, SIGTERM and orchestrator death."""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time

import numpy as np
import pytest

from control import VehicleParams
from control.depth_ftg import DepthFTGConfig, DepthGapFollower, Intrinsics, depth_to_scan
from control.sim import render_depth
from workers.base_worker import BaseWorker

P = VehicleParams.load()
K = Intrinsics(99.0, 98.95, 81.75, 47.25)       # OAK-D RGB intrinsics scaled to 160x100


def _corridor():
    wall_l = np.array([[x, 0.8] for x in np.linspace(0, 10, 50)])
    wall_r = np.array([[x, -0.8] for x in np.linspace(0, 10, 50)])
    return render_depth(0.0, 0.0, 0.0, [wall_l, wall_r], K, 160, 100)


class TestDepthGuards:

    def test_drives_in_a_corridor(self):
        assert DepthGapFollower(P, K).step(_corridor()).throttle > 0

    @pytest.mark.parametrize("fill", [0.0, np.nan])
    def test_invalid_depth_is_neutral(self, fill):
        cmd = DepthGapFollower(P, K).step(np.full((100, 160), fill, np.float32))
        assert (cmd.steering, cmd.throttle) == (0.0, 0.0)

    def test_mostly_invalid_depth_is_neutral(self):
        depth = _corridor()
        depth[:95] = 0.0                  # only the bottom rows left
        cmd = DepthGapFollower(P, K, DepthFTGConfig(min_valid_fraction=0.2)).step(depth)
        assert cmd.throttle == 0.0

    def test_camera_pitch_levels_the_floor(self):
        # a camera pitched down sees the floor rise toward the image centre; with
        # the pitch configured it must not show up as an obstacle
        pitch = np.radians(10)
        h, w = 100, 160
        v, u = np.mgrid[0:h, 0:w].astype(np.float64)
        ray = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u)], -1)
        cp, sp = np.cos(pitch), np.sin(pitch)
        ray_y_level = ray[..., 1] * cp + ray[..., 2] * sp
        depth = np.where(ray_y_level > 1e-3, 0.12 / np.maximum(ray_y_level, 1e-3), 0.0)
        cfg = DepthFTGConfig(camera_pitch=pitch)
        _, ranges = depth_to_scan(depth.astype(np.float32), K, cfg)
        assert ranges.min() == pytest.approx(cfg.max_range)
        _, ranges_unlevelled = depth_to_scan(depth.astype(np.float32), K, DepthFTGConfig())
        assert ranges_unlevelled.min() < cfg.max_range


class _Failing(BaseWorker):
    max_step_failures = 3

    def setup(self):
        pass

    def step(self):
        raise RuntimeError("step failed")


class _BadSetup(BaseWorker):
    def setup(self):
        raise RuntimeError("no device")

    def step(self):
        pass


class _Idle(BaseWorker):
    def setup(self):
        pass

    def step(self):
        time.sleep(0.01)

    def teardown(self):
        open(self.config["marker"], "w").close()


_alive = []     # the worker (and its stop Event) must outlive the child's unpickling


def _run(worker) -> mp.Process:
    _alive.append(worker)
    p = mp.Process(target=worker.run)
    p.start()
    return p


def test_repeated_step_failures_exit_non_zero():
    p = _run(_Failing("failing"))
    p.join(10)
    assert p.exitcode == 1


def test_setup_failure_exits_non_zero():
    p = _run(_BadSetup("bad_setup"))
    p.join(10)
    assert p.exitcode == 1


def test_sigterm_runs_teardown(tmp_path):
    marker = tmp_path / "teardown"
    p = _run(_Idle("idle", marker=str(marker)))
    time.sleep(1.5)
    os.kill(p.pid, signal.SIGTERM)
    p.join(10)
    assert p.exitcode == 0 and marker.exists()


def _spawn_orphan(marker: str, pid_file: str) -> None:
    p = mp.Process(target=_Idle("orphan", marker=marker).run)
    p.start()
    with open(pid_file, "w") as f:
        f.write(str(p.pid))
    time.sleep(2.0)                        # let the child start up
    os._exit(0)                            # die without stopping the child


def test_worker_stops_when_orchestrator_dies(tmp_path):
    marker, pid_file = tmp_path / "teardown", tmp_path / "pid"
    parent = mp.Process(target=_spawn_orphan, args=(str(marker), str(pid_file)))
    parent.start()
    parent.join(10)
    deadline = time.time() + 15
    while not marker.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert marker.exists()
