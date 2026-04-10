"""
test_control_contract.py — Regression tests for the ControlBuffer [0, 1]
contract between controllers, ControlWorker and ActuatorWorker.

Covers:
    - signed_to_unit(): PID/FTG [-1, 1] output → [0, 1]
    - pose_to_bev(): heading convention matches controller bearings
    - ControlWorker PID mode never writes out-of-range values
    - ControlWorker neural mode passes ``timestamp`` to models that take it
    - ActuatorWorker clamps out-of-range / NaN commands instead of crashing
"""

from __future__ import annotations

import math
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import ControlBuffer, PoseBuffer, SegmentationBuffer, IMUBuffer, TrackBoundaryBuffer
from workers.control_worker import ControlWorker, pose_to_bev, signed_to_unit


def _uid() -> str:
    return uuid.uuid4().hex[:8]


class TestSignedToUnit:

    @pytest.mark.parametrize(
        "signed, expected",
        [(-1.0, 0.0), (0.0, 0.5), (1.0, 1.0), (-0.5, 0.25), (-3.0, 0.0), (3.0, 1.0)],
    )
    def test_steering_mapping(self, signed, expected):
        steer, _ = signed_to_unit(signed, 0.3)
        assert steer == pytest.approx(expected)

    @pytest.mark.parametrize("throttle, expected", [(-0.4, 0.0), (0.3, 0.3), (1.7, 1.0)])
    def test_throttle_clipped(self, throttle, expected):
        _, thr = signed_to_unit(0.0, throttle)
        assert thr == pytest.approx(expected)


class TestPoseToBev:

    @pytest.mark.parametrize(
        "forward, expected",
        [((1, 0, 0), 0.0), ((0, 0, 1), np.pi / 2), ((-1, 0, 0), np.pi), ((0, 0, -1), -np.pi / 2)],
    )
    def test_heading_matches_bearing_convention(self, forward, expected):
        T = np.eye(4, dtype=np.float32)
        T[:3, 2] = forward
        T[0, 3], T[2, 3] = 1.5, -2.0
        pos, heading = pose_to_bev(T)
        np.testing.assert_allclose(pos, [1.5, -2.0])
        assert np.isclose(np.cos(heading), np.cos(expected), atol=1e-6)
        assert np.isclose(np.sin(heading), np.sin(expected), atol=1e-6)

    def test_straight_at_target_gives_zero_steer(self):
        """Driving along +X at a target dead ahead must not command a turn."""
        from workers.track_pid_controller import TrackPIDController

        pid = TrackPIDController(kp_steer=1.0, ki_steer=0.0, kd_steer=0.0, noise_std=0.0)
        T = np.eye(4, dtype=np.float32)
        T[:3, 2] = (1, 0, 0)
        pos, heading = pose_to_bev(T)
        outer = np.array([[5.0, 1.0]], dtype=np.float32)
        inner = np.array([[5.0, -1.0]], dtype=np.float32)
        steer, _ = pid.step(pos, heading, outer, inner)
        assert steer == pytest.approx(0.0, abs=1e-6)


class TestControlWorkerPIDContract:

    def test_pid_output_in_unit_range(self, track_boundaries):
        """Drive PID with poses around the track; every write must be in [0, 1]."""
        outer, inner = track_boundaries
        ctrl = ControlBuffer(slots=4, name=f"cc_ctrl_{_uid()}", create=True)
        pose = PoseBuffer(slots=4, name=f"cc_pose_{_uid()}", create=True)
        bnd = TrackBoundaryBuffer(slots=2, name=f"cc_bnd_{_uid()}", create=True)
        try:
            bnd.write(outer, inner)
            w = ControlWorker(
                name="control", mode="pid", output_buffer=ctrl,
                pose_buffer=pose, boundary_buffer=bnd,
                pid_config={"kp_steer": 5.0, "noise_std": 0.0},
            )
            w._logger = MagicMock()
            w.setup()

            for theta in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                T = np.eye(4, dtype=np.float32)
                T[0, 3], T[2, 3] = 6.5 * np.cos(theta), 4.0 * np.sin(theta)
                # Heading pointing *away* from the track forces large errors
                T[0, 2], T[2, 2] = np.cos(theta), np.sin(theta)
                pose.write(T)
                w.step()
                _, steer, thr = ctrl.read_latest()
                assert 0.0 <= steer <= 1.0
                assert 0.0 <= thr <= 1.0
        finally:
            for b in (ctrl, pose, bnd):
                b.unlink()


class _TimestampModel:
    def __init__(self):
        self.calls = []

    def get_action(self, state, mask, timestamp):
        self.calls.append(timestamp)
        return 0.5, 0.2


class _LegacyModel:
    def get_action(self, state, mask):
        return 1.5, -0.2  # deliberately out of range


class TestControlWorkerNeural:

    def _run(self, model, monkeypatch):
        seg = SegmentationBuffer(slots=4, name=f"cc_seg_{_uid()}", create=True)
        imu = IMUBuffer(slots=4, name=f"cc_imu_{_uid()}", create=True)
        ctrl = ControlBuffer(slots=4, name=f"cc_ctrl_{_uid()}", create=True)
        try:
            import utils
            monkeypatch.setattr(utils, "load_model_from_config", lambda _p: (model, type(model).__name__))
            w = ControlWorker(
                name="control", mode="neural", output_buffer=ctrl,
                input_buffers={"segments": seg, "imu": imu}, model_config="unused.yaml",
            )
            w._logger = MagicMock()
            w.setup()
            hdr = seg.write(np.zeros((400, 640), dtype=np.float32))
            w.step()
            return hdr, ctrl.read_latest()
        finally:
            for b in (seg, imu, ctrl):
                b.unlink()

    def test_timestamp_passed_when_accepted(self, monkeypatch):
        model = _TimestampModel()
        hdr, result = self._run(model, monkeypatch)
        assert model.calls == [pytest.approx(hdr.produced_ts)]
        assert result[1:] == pytest.approx((0.5, 0.2))

    def test_legacy_signature_and_clamping(self, monkeypatch):
        _, result = self._run(_LegacyModel(), monkeypatch)
        assert result[1:] == pytest.approx((1.0, 0.0))


class TestActuatorClamp:

    @pytest.mark.parametrize(
        "steer, thr, expected",
        [(-0.3, 0.2, (0.0, 0.2)), (1.4, 1.2, (1.0, 1.0)), (math.nan, math.nan, (0.5, 0.0))],
    )
    def test_out_of_range_is_clamped(self, steer, thr, expected):
        from workers import ActuatorWorker

        ctrl = ControlBuffer(slots=4, name=f"cc_act_{_uid()}", create=True)
        try:
            w = ActuatorWorker(name="actuator", input_buffer=ctrl, i2c_address=None, i2c_bus=None)
            w._logger = MagicMock()
            w.setup()
            w._controller = MagicMock(wraps=w._controller)
            ctrl.write(steer, thr)
            w.step()  # must not raise
            args = w._controller.get_normalized_action_output.call_args[0]
            assert args == pytest.approx(expected)
        finally:
            ctrl.unlink()
