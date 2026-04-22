"""
test_control_workers.py — Integration of the control library into the
pipeline workers over real shared-memory buffers:

    - ActuatorWorker: arming, watchdog, speed cap, fixed-rate output
    - StateWorker: poses → state estimate, stale detection
    - ControlWorker: pure_pursuit / stanley modes, registry
    - start.KeyboardControl: piped-stdin fallback
"""

from __future__ import annotations

import multiprocessing as mp
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from control import VehicleState
from control.sim import oval_track
from shared_buffer import ControlBuffer, PoseBuffer, StateBuffer, TrackBoundaryBuffer
from workers.control_worker import signed_to_unit, unit_to_signed


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _pump(worker, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        worker.step()


def test_unit_signed_round_trip():
    for s, t in [(0.3, -0.6), (-1.0, 1.0), (0.0, 0.0)]:
        assert unit_to_signed(*signed_to_unit(s, t)) == pytest.approx((s, t))


# --- Actuator ---

class TestActuatorSafety:

    @pytest.fixture()
    def rig(self):
        from workers import ActuatorWorker

        ctrl = ControlBuffer(slots=4, name=f"as_c_{_uid()}", create=True)
        state = StateBuffer(slots=4, name=f"as_s_{_uid()}", create=True)
        armed = mp.get_context("spawn").Event()
        w = ActuatorWorker(
            name="actuator", input_buffer=ctrl, armed_event=armed, state_buffer=state,
            safety={"steering_rate": 1000, "throttle_rate": 1000, "speed_per_throttle": None},
            rate_hz=200,
        )
        w._logger = MagicMock()
        w.setup()
        w._pwm = MagicMock(wraps=w._pwm)
        yield w, ctrl, state, armed
        ctrl.unlink()
        state.unlink()

    @staticmethod
    def _last_output(w):
        # PWMMapper is the only place ControlBuffer values reach the hardware
        return w._pwm.to_pwm.call_args[0]

    def test_neutral_until_armed(self, rig):
        w, ctrl, _, armed = rig
        ctrl.write(*signed_to_unit(0.5, 0.3))
        _pump(w, 0.05)
        assert self._last_output(w) == pytest.approx((0.5, 0.5))
        armed.set()
        ctrl.write(*signed_to_unit(0.5, 0.3))
        _pump(w, 0.05)
        assert self._last_output(w) == pytest.approx(signed_to_unit(0.5, 0.3))

    def test_watchdog_after_command_timeout(self, rig):
        w, ctrl, _, armed = rig
        armed.set()
        ctrl.write(*signed_to_unit(0.4, 0.3))
        _pump(w, 0.05)
        assert self._last_output(w)[1] > 0.5
        _pump(w, 0.2)   # > 150 ms without a new command
        steer_u, thr_u = self._last_output(w)
        assert thr_u == pytest.approx(0.5)                          # neutral throttle
        assert steer_u == pytest.approx(signed_to_unit(0.4, 0)[0])  # steering held

    def test_speed_cap_from_state(self, rig):
        w, ctrl, state, armed = rig
        armed.set()
        state.write(VehicleState(t=time.time(), x=0, y=0, yaw=0, v=1.6))
        ctrl.write(*signed_to_unit(0.0, 0.6))
        _pump(w, 0.03)
        assert self._last_output(w)[1] == pytest.approx(0.5)   # over the cap → coast


# --- State ---

class TestStateWorker:

    def test_estimates_motion_and_goes_stale(self):
        from workers import StateWorker

        pb = PoseBuffer(slots=8, name=f"sw_p_{_uid()}", create=True)
        sb = StateBuffer(slots=8, name=f"sw_s_{_uid()}", create=True)
        try:
            w = StateWorker(name="state", pose_buffer=pb, output_buffer=sb)
            w._logger = MagicMock()
            w.setup()
            # Camera frame: ground plane X/Z, forward = Z.  Drive along +Z at 1 m/s.
            t0 = time.time()
            while time.time() - t0 < 1.0:
                T = np.eye(4, dtype=np.float32)
                T[2, 3] = time.time() - t0
                pb.write(T)
                _pump(w, 1 / 15)
            _, st = sb.read_latest()
            assert st.v == pytest.approx(1.0, abs=0.15)
            assert st.yaw == pytest.approx(np.pi / 2, abs=0.05)   # +Z is 90° from +X
            _pump(w, 1.0)                     # no poses: publishes for stale_after (0.5 s), then stops
            _, st2 = sb.read_latest()
            assert time.time() - st2.t > 0.4  # nothing published for the last ~0.5 s
        finally:
            pb.unlink()
            sb.unlink()


# --- Control ---

@pytest.mark.parametrize("mode", ["pure_pursuit", "stanley"])
def test_control_worker_track_mode(mode):
    from workers import ControlWorker

    outer, inner, centre = oval_track()
    ctrl = ControlBuffer(slots=4, name=f"cw_c_{_uid()}", create=True)
    sb = StateBuffer(slots=4, name=f"cw_s_{_uid()}", create=True)
    bb = TrackBoundaryBuffer(slots=2, name=f"cw_b_{_uid()}", create=True)
    try:
        bb.write(outer.astype(np.float32), inner.astype(np.float32))
        w = ControlWorker(name="control", mode=mode, output_buffer=ctrl, state_buffer=sb, boundary_buffer=bb)
        w._logger = MagicMock()
        w.setup()
        # At (6.5, 0) heading clockwise (-y), the reverse of the boundary order,
        # so the worker must flip the track.  +x is the car's LEFT here: offset
        # 0.3 m left on a right-hand bend → must steer right, and drive forward.
        x, y = centre[0]
        sb.write(VehicleState(t=time.time(), x=x + 0.3, y=y, yaw=-np.pi / 2, v=0.5))
        w.step()
        _, steer_u, thr_u = ctrl.read_latest()
        assert w._follower is not None
        assert steer_u > 0.5          # buffer: 1 = full right
        assert thr_u > 0.5            # forward
    finally:
        ctrl.unlink()
        sb.unlink()
        bb.unlink()


def test_control_worker_rejects_unknown_mode():
    from workers import ControlWorker

    w = ControlWorker(name="control", mode="bogus", output_buffer=None)
    w._logger = MagicMock()
    with pytest.raises(ValueError):
        w.setup()


# --- Keyboard ---

def test_keyboard_control_piped_stdin(monkeypatch):
    import io
    import start

    fake = io.StringIO("a\n \nl\n")
    fake.isatty = lambda: False
    monkeypatch.setattr(start.sys, "stdin", fake)
    monkeypatch.setattr("select.select", lambda r, w, x, t: (r if fake.tell() < len(fake.getvalue()) else [], [], []))
    kb = start.KeyboardControl()
    assert kb.poll() == ["a", " ", "l"]
    kb.close()
