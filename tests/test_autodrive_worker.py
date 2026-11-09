"""AutoDRIVEWorker against a real captured sim message and the fake sim
(tests/fake_autodrive.py): decoding, the wire contract (keys, string encoding,
steering sign, neutral when disarmed or stale, reset pulse, neutral on teardown),
and a closed loop through the whole pipeline (LiDAR mapping lap, then pure_pursuit)."""

from __future__ import annotations

import json
import logging
import math
import multiprocessing as mp
import os
import socket
import time
import uuid

import numpy as np
import pytest

pytest.importorskip("websockets")

from shared_buffer import ControlBuffer, FrameBuffer, IMUBuffer, LidarBuffer, PoseBuffer  # noqa: E402
from tests.conftest import REPO_ROOT  # noqa: E402
from tests.fake_autodrive import FakeAutoDRIVE  # noqa: E402
from workers import ActuatorWorker, AutoDRIVEWorker  # noqa: E402
from workers.autodrive_worker import (  # noqa: E402
    autodrive_to_internal, command_message, decode_lidar, encode_lidar, yaw_from_quat)
from workers.control_worker import signed_to_unit  # noqa: E402

SAMPLE = os.path.join(REPO_ROOT, "tests", "data", "autodrive_bridge_sample.json")


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait(cond, timeout: float, msg: str = "") -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out: {msg}")
        time.sleep(0.01)


# decoding


def test_real_sim_message_decodes():
    """A 'Bridge' message captured from the stock 2026-iros sim."""
    with open(SAMPLE) as f:
        msg = json.load(f)
    d = autodrive_to_internal(msg)
    assert d["position"] == pytest.approx([float(v) for v in msg["V1 Position"].split()])
    assert len(d["position"]) == 3
    # quaternion is x y z w: its yaw matches the sim's own Euler angles (Z = 4.7124 rad)
    euler_z = float(msg["V1 Orientation Euler Angles"].split()[2])
    assert yaw_from_quat(d["orientation_quat"]) % (2 * math.pi) == pytest.approx(euler_z, abs=1e-3)
    assert len(d["lidar"]) == 1081
    assert np.isinf(d["lidar"]).any() and (np.array(d["lidar"])[np.isfinite(d["lidar"])] > 0).all()
    assert d["lap_count"] == 0 and d["collision_count"] == 0
    assert d["last_lap_time"] == math.inf
    assert d["front_camera"]["encoding"] == "jpeg"


def test_lidar_round_trip_keeps_inf():
    r = np.array([0.5, 1.25, np.inf, 9.9999], dtype=np.float32)
    out = decode_lidar(encode_lidar(r))
    assert out[2] == np.inf and out[[0, 1, 3]] == pytest.approx(r[[0, 1, 3]], abs=1e-4)


def test_speed_is_signed_and_steering_echo_normalised():
    """Real sim: body-frame velocity (x < 0 reversing), steering echo in rad (30 deg = 1)."""
    d = autodrive_to_internal({"V1 Linear Velocity": "-1.9946 0.0000 -0.0806", "V1 Steering": "-0.2618"})
    assert d["speed"] == pytest.approx(-1.9946)
    assert d["steering"] == pytest.approx(-0.5, abs=1e-4) and d["steering_angle"] == pytest.approx(-0.2618)


def test_command_message_is_string_encoded():
    frame = command_message(0.25, -1.0, reset=True)
    assert frame.startswith("42")
    event, data = json.loads(frame[2:])
    assert event == "Bridge"
    assert data == {"V1 Throttle": "0.2500", "V1 Steering": "-1.0000", "V1 Reset": "True"}


def test_sim_message_fills_buffers_with_quaternion_order_and_jpeg():
    fake = FakeAutoDRIVE(port=1)          # not started: just its telemetry
    fake.car.yaw, fake.car.x, fake.car.y, fake.car.v = 1.0, 2.0, -1.5, 0.7
    names = [f"ad_{_uid()}" for _ in range(4)]
    pb, fb = PoseBuffer(slots=4, name=names[0]), FrameBuffer(slots=2, name=names[1], max_payload=200_000)
    lb, ib = LidarBuffer(slots=4, name=names[2]), IMUBuffer(slots=4, name=names[3])
    w = AutoDRIVEWorker(pose_buffer=pb, frame_buffer=fb, lidar_buffer=lb, imu_buffer=ib)
    w._pose_buf, w._frame_buf, w._lidar_buf, w._imu_buf = pb, fb, lb, ib
    try:
        d = autodrive_to_internal(fake.telemetry())
        w._process_pose_data(d)
        w._process_frame_data(d)
        w._process_lidar_data(d)
        _, pose, _, speed = pb.read_latest()
        assert pose[:2, 3] == pytest.approx([2.0, -1.5], abs=1e-4)
        assert math.atan2(pose[1, 0], pose[0, 0]) == pytest.approx(1.0, abs=1e-3)
        assert speed == pytest.approx(0.7, abs=1e-3)
        _, rgb, depth = fb.read_latest()
        assert rgb.shape == (108, 192, 3) and depth.shape == (108, 192)
        _, ranges = lb.read_latest()
        assert ranges.shape == (1080,)
        # beam 540 looks straight ahead: the oval's wall is in front, not behind
        assert np.isfinite(ranges[540])
    finally:
        for b in (pb, fb, lb, ib):
            b.unlink()


_SPAWN_CHECK = """
import multiprocessing as mp, sys
mp.set_start_method("spawn", force=True)          # as start.py does
from shared_buffer import ControlBuffer
from workers import AutoDRIVEWorker

def check(worker):
    assert worker.config["port"] == 4567

if __name__ == "__main__":
    act = ControlBuffer(slots=4, name=sys.argv[1])
    try:
        w = AutoDRIVEWorker(port=4567, command_buffer=act, reset_event=mp.Event(), lap_counter=mp.Value("i", 0))
        p = mp.Process(target=check, args=(w,))
        p.start()
        p.join(60)
        sys.exit(p.exitcode)
    finally:
        act.unlink()
"""


def test_worker_survives_spawn(tmp_path):
    """start.py uses the spawn start method: the worker as built must pickle into a child."""
    import subprocess
    import sys

    script = tmp_path / "spawn_check.py"
    script.write_text(_SPAWN_CHECK)
    r = subprocess.run([sys.executable, str(script), f"pk_{_uid()}"], cwd=REPO_ROOT, capture_output=True,
                       text=True, timeout=120, env={**os.environ, "PYTHONPATH": REPO_ROOT})
    assert r.returncode == 0, r.stderr[-2000:]


# wire contract against the fake sim


@pytest.fixture
def link(tmp_path):
    """AutoDRIVEWorker set up in-process + fake sim connected to it."""
    port = _free_port()
    act = ControlBuffer(slots=8, name=f"act_{_uid()}")
    pb = PoseBuffer(slots=8, name=f"pose_{_uid()}")
    reset = mp.Event()
    laps, coll = mp.Value("i", 0), mp.Value("i", 0)
    w = AutoDRIVEWorker(host="127.0.0.1", port=port, pose_buffer=pb, command_buffer=act,
                        reset_event=reset, lap_counter=laps, collision_counter=coll,
                        log_dir=str(tmp_path), command_timeout=0.15)
    w._logger = logging.getLogger("test.autodrive")
    w.setup()
    fake = FakeAutoDRIVE(port, rate_hz=25.0, lidar=False, camera=False).start()
    try:
        _wait(lambda: pb.latest_seq() > 3, 10.0, "telemetry into pose_buf")
        yield w, fake, act, pb, reset, laps, coll, tmp_path
    finally:
        w.teardown()
        fake.close()
        act.unlink()
        pb.unlink()


def test_wire_contract(link):
    w, fake, act, pb, reset, laps, coll, log_dir = link
    cmds = fake.stats.commands
    # nothing in the command buffer yet: neutral, string-encoded, exactly three keys
    assert cmds and all(set(c) == {"V1 Throttle", "V1 Steering", "V1 Reset"} for c in cmds)
    assert all(isinstance(v, str) for c in cmds for v in c.values())
    assert all(float(c["V1 Throttle"]) == 0.0 and float(c["V1 Steering"]) == 0.0 for c in cmds)

    # a fresh supervised command goes out as is; +steering turns the car left (CCW)
    yaw0 = fake.car.yaw
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        act.write(*signed_to_unit(0.5, 0.3))
        time.sleep(0.01)
    sent = [c for c in fake.stats.commands if float(c["V1 Throttle"]) > 0]
    assert sent and float(sent[-1]["V1 Steering"]) == pytest.approx(0.5)
    assert float(sent[-1]["V1 Throttle"]) == pytest.approx(0.3, abs=1e-3)
    dyaw = (fake.car.yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi
    assert dyaw > 0.05, "positive steering must turn left"

    # commands stop: once older than command_timeout the sim gets neutral
    time.sleep(0.4)
    n = len(fake.stats.commands)
    _wait(lambda: len(fake.stats.commands) > n + 3, 5.0)
    assert float(fake.stats.commands[-1]["V1 Throttle"]) == 0.0
    assert float(fake.stats.commands[-1]["V1 Steering"]) == 0.0

    # reset: 'True' (with neutral) held for reset_hold_s, one block, then 'False' again
    n = len(fake.stats.commands)
    reset.set()
    _wait(lambda: not reset.is_set(), 3.0, "reset sent")
    m = len(fake.stats.commands)
    _wait(lambda: len(fake.stats.commands) > m + 3, 5.0, "replies after the reset hold")
    flags = "".join("T" if c["V1 Reset"] == "True" else "F" for c in fake.stats.commands[n:])
    assert flags.strip("F").count("F") == 0 and flags.count("T") >= 2 and flags.endswith("F")
    assert all(float(c["V1 Throttle"]) == 0.0 for c in fake.stats.commands[n:] if c["V1 Reset"] == "True")
    assert fake.stats.resets == 1

    # race counters reach the shared values and sim_events.jsonl
    fake.lap_count, fake.collisions = 2, 1
    fake.lap_times = [12.5, 11.0]
    _wait(lambda: laps.value == 2 and coll.value == 1, 3.0, "counters")
    with open(log_dir / "sim_events.jsonl") as f:
        events = [json.loads(line) for line in f]
    kinds = [e["event"] for e in events]
    assert "reset_sent" in kinds and "lap" in kinds and "collision" in kinds
    assert next(e for e in events if e["event"] == "lap")["lap_time"] == pytest.approx(11.0)

    # echoes (same sim clock) are answered but not written twice
    rows = open(log_dir / "sim_messages.csv").read().splitlines()
    assert rows[0].startswith("t_rx,t_reply,sim_time,new_frame")
    new = [r.split(",")[3] for r in rows[1:]]
    assert new.count("0") > 0 and new.count("1") > 0
    assert pb.latest_seq() <= new.count("1")


def test_teardown_leaves_the_sim_neutral(link):
    w, fake, act, *_ = link
    t_end = time.monotonic() + 0.5
    while time.monotonic() < t_end:
        act.write(*signed_to_unit(-0.4, 0.5))
        time.sleep(0.01)
    assert float(fake.stats.commands[-1]["V1 Throttle"]) > 0
    w.teardown()
    w.teardown = lambda: None      # the fixture's teardown is a no-op now
    time.sleep(0.2)
    last = fake.stats.commands[-1]
    assert float(last["V1 Throttle"]) == 0.0 and float(last["V1 Steering"]) == 0.0


def test_actuator_output_buffer_is_neutral_until_armed():
    ctrl = ControlBuffer(slots=8, name=f"c_{_uid()}")
    out = ControlBuffer(slots=8, name=f"o_{_uid()}")
    armed = mp.Event()
    a = ActuatorWorker(input_buffer=ctrl, output_buffer=out, armed_event=armed,
                       safety={"speed_per_throttle": None, "throttle_rate": 100.0, "steering_rate": 100.0})
    a._logger = logging.getLogger("test.actuator")
    try:
        a.setup()

        def run(seconds):
            t_end = time.monotonic() + seconds
            while time.monotonic() < t_end:
                ctrl.write(*signed_to_unit(0.6, 0.3))
                a.step()

        run(0.2)
        _, s, t = out.read_latest()
        assert (s, t) == pytest.approx(signed_to_unit(0.0, 0.0))     # disarmed -> neutral
        armed.set()
        run(0.3)
        _, s, t = out.read_latest()
        assert (s, t) == pytest.approx(signed_to_unit(0.6, 0.3), abs=1e-3)
        a.teardown()
        _, s, t = out.read_latest()
        assert (s, t) == pytest.approx(signed_to_unit(0.0, 0.0))
    finally:
        ctrl.unlink()
        out.unlink()


# closed loop through the whole pipeline


@pytest.mark.slow
def test_closed_loop_pure_pursuit_on_a_saved_map(tmp_path, monkeypatch):
    """start.build_pipeline(config_autodrive.yaml) + the fake sim, no LiDAR: pure_pursuit laps
    on a saved map of the fake track, with the state estimated from the poses alone. (The
    mapping lap needs the camera's segmentation model, not in the repo: it is checked live.)"""
    import start

    suffix = _uid()
    for cls_name in ("FrameBuffer", "IMUBuffer", "SegmentationBuffer", "PoseBuffer", "ControlBuffer",
                     "PointCloudBuffer", "TrackBoundaryBuffer", "LidarBuffer", "StateBuffer"):
        cls = getattr(start, cls_name)
        monkeypatch.setattr(start, cls_name,
                            lambda *a, _cls=cls, **kw: _cls(*a, **{**kw, "name": f"{kw['name']}_{suffix}"}))
    port = _free_port()
    config = start.load_config(os.path.join(REPO_ROOT, "config_autodrive.yaml"))
    fake = FakeAutoDRIVE(port, rate_hz=20.0, camera=False, lidar=False)
    os.makedirs(tmp_path / "map")
    np.save(tmp_path / "map" / "outer_line.npy", fake.outer)
    np.save(tmp_path / "map" / "inner_line.npy", fake.inner)
    config.update(sim_host="127.0.0.1", sim_port=port, enable_slam=False, use_saved_map=True,
                  saved_map_path=str(tmp_path / "map"), recording=True,
                  recording_path=str(tmp_path / "rec"), sim_camera=False)
    config.setdefault("track_control", {}).pop("raceline_path", None)   # the centre-line

    p = start.build_pipeline(config)
    assert p.mapping_done_event.is_set() and not any(w.name in ("vision", "seg_scan") for w in p.workers)
    procs = []
    try:
        for w in p.workers:
            proc = mp.Process(target=w.run, name=w.name)
            proc.start()
            procs.append(proc)
        fake.start()          # only after forking: a running client thread can deadlock a fork
        _wait(lambda: fake.stats.connected and fake.stats.sent > 10, 20.0, "sim connected")
        p.armed_event.set()
        laps_at_mapping = fake.lap_count
        # counted on the pilot_core side: the sim's lap counter reached the shared value
        _wait(lambda: p.counters["lap"].value >= laps_at_mapping + 2, 90.0, "two pure_pursuit laps")
        assert fake.collisions == 0, f"{fake.collisions} collisions"
        with open(os.path.join(p.recording_dir, "sim_events.jsonl")) as f:
            assert sum(json.loads(line)["event"] == "lap" for line in f) >= 2
    finally:
        p.armed_event.clear()
        for w in p.workers:
            w.stop()
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        fake.close()
        for b in p.buffers:
            b.unlink()
    assert all(proc.exitcode == 0 for proc in procs), [(q.name, q.exitcode) for q in procs]


def test_a_frame_that_fails_still_writes_the_pose():
    # e.g. frame_buf sized for 192x108 while sim_depth writes 384x216: the camera error
    # must not hold back the pose, or the controller never sees the car move
    fake = FakeAutoDRIVE(port=1)
    fake.car.x, fake.car.y = 2.0, -1.5
    names = [f"ad_{_uid()}" for _ in range(2)]
    pb, fb = PoseBuffer(slots=4, name=names[0]), FrameBuffer(slots=2, name=names[1], max_payload=1000)
    w = AutoDRIVEWorker(host="127.0.0.1", port=0, pose_buffer=pb, frame_buffer=fb,
                        sim_depth={"camera": "sim_camera/camera.yaml", "scale": 2})
    w.setup()

    class _WS:
        def send(self, msg):
            pass

    try:
        w._on_bridge(_WS(), "42" + json.dumps(["Bridge", fake.telemetry()]))
        assert w._frame_errors == 1
        _, pose, _, _ = pb.read_latest()
        assert pose[:2, 3] == pytest.approx([2.0, -1.5], abs=1e-4)
    finally:
        w.teardown()
        for b in (pb, fb):
            b.unlink()
