"""ZMQSensorWorker / ZMQPublishWorker / scripts/collect_frames.py, offline
(local sockets, no simulator)."""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import socket
import time
import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")

from shared_buffer import FrameBuffer, PoseBuffer, SegmentationBuffer, StateBuffer  # noqa: E402
from workers import ZMQPublishWorker, ZMQSensorWorker  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _pose_msg(yaw: float, speed: float = 1.2) -> dict:
    return {"position": [1.0, 2.0, 0.0],
            "orientation_quat": [0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2))],
            "speed": speed, "collision_count": 2}


def test_sensor_worker_without_frames_writes_pose_and_state():
    port = _free_port()
    pb = PoseBuffer(slots=4, name=f"zp_{_uid()}", create=True)
    sb = StateBuffer(slots=4, name=f"zs_{_uid()}", create=True)
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    w = ZMQSensorWorker(host="127.0.0.1", port=port, frame_buffer=None, imu_buffer=None,
                        pose_buffer=pb, state_buffer=sb)
    w._logger = MagicMock()
    try:
        w.setup()
        msg = _pose_msg(0.5)
        # a camera frame that would fail to decode: must be ignored without frame_buffer
        msg["front_camera"] = {"data": "not base64!", "encoding": "jpeg"}
        deadline = time.monotonic() + 5.0
        while pb.latest_seq() == 0 and time.monotonic() < deadline:
            pub.send_string(json.dumps(msg))
            w.step()
        assert pb.latest_seq() > 0
        _, pose, _, speed = pb.read_latest()
        assert speed == pytest.approx(1.2)
        _, st = sb.read_latest()
        assert (st.x, st.y, st.v) == pytest.approx((1.0, 2.0, 1.2))
        assert st.yaw == pytest.approx(0.5, abs=1e-5)
        assert abs(time.monotonic() - st.t) < 1.0
        w._logger.warning.assert_not_called()

        # silence: poll-based step returns and warns after no_message_warn_s
        w._no_msg_warn_s = 0.05
        w._last_msg_time = time.monotonic() - 0.2
        w.step()
        assert any("No sensor message" in str(c) for c in w._logger.warning.call_args_list)
    finally:
        w.teardown()
        pub.close(linger=0)
        ctx.term()
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
    w = ZMQSensorWorker(frame_buffer=fb)
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


def test_sensor_worker_default_depth_max():
    assert ZMQSensorWorker()._depth_max_meters == 15.0


def test_publish_worker_no_control_topic_and_binary_mask():
    from shared_buffer import ControlBuffer

    port = _free_port()
    seg = SegmentationBuffer(slots=2, name=f"zg_{_uid()}", create=True)
    ctrl = ControlBuffer(slots=2, name=f"zc_{_uid()}", create=True)
    w = ZMQPublishWorker(port=port, seg_buffer=seg, control_buffer=ctrl,
                         publish={"control": True, "segmentation": True})
    w._logger = MagicMock()
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    try:
        w.setup()
        sub.connect(f"tcp://127.0.0.1:{port}")
        time.sleep(0.3)
        mask = np.zeros((8, 10), np.uint8)
        mask[2:6, 3:7] = 1
        ctrl.write(0.2, 0.8)
        topics, seg_msg = [], None
        deadline = time.monotonic() + 3.0
        while seg_msg is None and time.monotonic() < deadline:
            seg.write(mask)
            w.step()
            while sub.poll(10):
                topic, body = sub.recv_string().split(" ", 1)
                topics.append(topic)
                if topic == "segmentation":
                    seg_msg = json.loads(body)
        assert seg_msg is not None
        assert "control" not in topics
        got = np.frombuffer(base64.b64decode(seg_msg["mask"]), np.uint8).reshape(seg_msg["shape"])
        np.testing.assert_array_equal(got, mask)
        assert seg_msg["dtype"] == "uint8" and set(np.unique(got)) <= {0, 1}
    finally:
        w.teardown()
        sub.close(linger=0)
        ctx.term()
        seg.unlink()
        ctrl.unlink()


def _collect_frames():
    spec = importlib.util.spec_from_file_location("collect_frames", os.path.join(REPO, "scripts", "collect_frames.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_frames_encoding_and_sessions(tmp_path):
    cv2 = pytest.importorskip("cv2")
    cf = _collect_frames()

    rgb = np.zeros((6, 8, 3), np.uint8)
    rgb[..., 0] = 200                                   # red in RGB
    ok, jpg = cv2.imencode(".jpg", rgb)
    name, data = cf.frame_to_file({"data": base64.b64encode(jpg.tobytes()).decode(), "encoding": "jpeg"}, "f0")
    assert name == "f0.jpg" and data == jpg.tobytes()

    name, data = cf.frame_to_file({"data": base64.b64encode(rgb.tobytes()).decode(), "encoding": "raw",
                                   "shape": [6, 8, 3]}, "f1")
    assert name == "f1.png"
    back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert back[0, 0, 2] == 200 and back[0, 0, 0] == 0   # BGR on disk, red preserved

    with pytest.raises(ValueError):
        cf.frame_to_file({"data": "", "encoding": "h264"}, "f2")

    a, b = cf.session_dir(str(tmp_path)), cf.session_dir(str(tmp_path))
    assert a != b and os.path.isdir(a) and os.path.isdir(b)
    assert os.path.dirname(a) == str(tmp_path)
