"""SimDecoder, the sensor decoders behind AutoDRIVEWorker, offline:
ground-truth pose and state, raw depth scaling, defaults."""

from __future__ import annotations

import base64
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared_buffer import FrameBuffer, PoseBuffer, StateBuffer
from workers import SimDecoder


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _pose_msg(yaw: float, speed: float = 1.2) -> dict:
    return {"position": [1.0, 2.0, 0.0],
            "orientation_quat": [0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
            "speed": speed, "collision_count": 2}


def test_decoder_without_frames_writes_pose_and_state():
    import time

    pb = PoseBuffer(slots=4, name=f"zp_{_uid()}", create=True)
    sb = StateBuffer(slots=4, name=f"zs_{_uid()}", create=True)
    w = SimDecoder(frame_buffer=None, imu_buffer=None, pose_buffer=pb, state_buffer=sb)
    w._logger = MagicMock()
    w._pose_buf, w._state_buf = pb, sb
    try:
        w._process_pose_data(_pose_msg(0.5))
        _, pose, _, speed = pb.read_latest()
        assert speed == pytest.approx(1.2)
        _, st = sb.read_latest()
        assert (st.x, st.y, st.v) == pytest.approx((1.0, 2.0, 1.2))
        assert st.yaw == pytest.approx(0.5, abs=1e-5)
        assert abs(time.monotonic() - st.t) < 1.0
        w._logger.warning.assert_not_called()

        # silence: warns after no_message_warn_s
        w._no_msg_warn_s = 0.05
        w._last_msg_time = time.monotonic() - 0.2
        w._check_silence(time.monotonic())
        assert any("No sensor message" in str(c) for c in w._logger.warning.call_args_list)
    finally:
        pb.unlink()
        sb.unlink()


def _depth_msg(depth_u16: np.ndarray, shape) -> dict:
    rgb = np.zeros((*depth_u16.shape, 3), np.uint8)
    return {"front_camera": {"data": base64.b64encode(rgb.tobytes()).decode(), "encoding": "raw",
                             "shape": list(rgb.shape)},
            "depth_camera": {"data": base64.b64encode(depth_u16.tobytes()).decode(), "encoding": "raw",
                             "shape": shape}}


@pytest.mark.parametrize("scale,shape,expect", [
    (0.001, [4, 6], "scaled"),     # metres per unit
    (None, [4, 6], "zeros"),       # no scale configured -> refuse, zero depth
    (0.001, None, "zeros"),        # no shape -> no square-reshape guessing
])
def test_raw_depth_requires_scale(scale, shape, expect):
    pytest.importorskip("cv2")
    fb = FrameBuffer(slots=2, name=f"zf_{_uid()}", create=True)
    w = SimDecoder(frame_buffer=fb)
    w._logger = MagicMock()
    w._frame_buf = fb
    w._depth_raw_scale = scale
    try:
        d = (np.arange(24, dtype=np.uint16) * 100).reshape(4, 6)
        w._process_frame_data(_depth_msg(d, shape))
        _, _, depth = fb.read_latest()
        if expect == "scaled":
            np.testing.assert_allclose(depth, d * 0.001, rtol=1e-6)
        else:
            assert not depth.any()
        if scale is None:
            w._logger.error.assert_called_once()
    finally:
        fb.unlink()


def test_decoder_default_depth_max():
    assert SimDecoder()._depth_max_meters == 15.0
