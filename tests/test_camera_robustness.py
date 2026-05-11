"""CameraWorker without a camera: re-open logic, settings-vs-calibration check."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import workers.camera_worker as cam_mod
from workers.base_worker import FatalWorkerError
from workers.camera_worker import CameraWorker, intrinsics_mismatch, read_settings_camera


def _worker(monkeypatch, step_results):
    """CameraWorker with the device parts replaced by fakes."""
    monkeypatch.setattr(cam_mod.time, "sleep", lambda s: None)
    w = CameraWorker(name="camera")
    w._logger = MagicMock()
    w._frame_timeout, w._max_step_errors, w._max_reopen = 1.0, 3, 2
    w._last_frame_t = time.monotonic()
    w._got_frame = True                 # past start-up: the 1 s frame timeout applies
    results = list(step_results)

    def fake_step():
        r = results.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    w._step_device = fake_step
    w._start_device = MagicMock()
    w._stop_pipeline = MagicMock()
    return w


def test_reopens_after_repeated_step_errors(monkeypatch):
    w = _worker(monkeypatch, [RuntimeError("X_LINK_ERROR")] * 3 + [True])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            w.step()
    w.step()                                   # third error in a row -> re-open
    w._stop_pipeline.assert_called_once()
    w._start_device.assert_called_once()
    w.step()
    assert w._step_errors == 0


def test_reopens_when_frames_stop(monkeypatch):
    w = _worker(monkeypatch, [False, False])
    w.step()
    w._start_device.assert_not_called()
    w._last_frame_t -= 2.0                     # nothing for 2 s
    w.step()
    w._start_device.assert_called_once()


def test_gives_up_and_stops_after_failed_reopens(monkeypatch):
    w = _worker(monkeypatch, [False])
    w._start_device.side_effect = RuntimeError("no device")
    w._last_frame_t -= 2.0
    with pytest.raises(FatalWorkerError, match="could not be re-opened"):
        w.step()
    assert w._start_device.call_count == 2


def test_settings_check(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text(
        '%YAML:1.0\n# Device: 14442C10F13ED2D200   Resolution: 640x400   FPS: 15\n'
        'File.version: "1.0"\nCamera1.fx: 396.0\nCamera1.fy: 395.8\n'
        'Camera1.cx: 327.0\nCamera1.cy: 189.0\n'
    )
    s = read_settings_camera(str(p))
    assert s["device_id"] == "14442C10F13ED2D200"
    K = np.array([[396.5, 0, 326.0], [0, 396.0, 190.0], [0, 0, 1]])
    assert intrinsics_mismatch(s, K) == []
    K[1, 2] = 200.0                            # cy 5 % off
    assert len(intrinsics_mismatch(s, K)) == 1

    p.write_text(p.read_text() + 'Pilot.deviceId: "ABC"\n')
    assert read_settings_camera(str(p))["device_id"] == "ABC"
    assert read_settings_camera(str(tmp_path / "missing.yaml")) is None


def test_first_frame_gets_the_startup_timeout(monkeypatch):
    w = _worker(monkeypatch, [False, False])
    w._got_frame = False
    w._last_frame_t = time.monotonic() - 2.0      # past frame_timeout, inside startup_timeout
    w.step()
    w._start_device.assert_not_called()
    w._last_frame_t = time.monotonic() - 6.0
    w.step()
    w._start_device.assert_called_once()


def test_synthetic_track_scene_publishes_projected_frames():
    import uuid

    from shared_buffer import FrameBuffer

    fb = FrameBuffer(slots=2, name=f"cr_f_{uuid.uuid4().hex[:8]}", create=True)
    try:
        w = CameraWorker(name="camera", output_buffer=fb, synthetic=True, synthetic_scene="track",
                         resolution=[160, 100], fps=1000)
        w._logger = MagicMock()
        w.setup()
        w.step()
        _, rgb, depth = fb.read_latest()
        assert rgb.shape == (100, 160, 3) and depth.shape == (100, 160)
        assert np.all(depth[-1] > 0) and depth[-1].max() < 2.0    # floor just ahead of the camera
    finally:
        fb.unlink()
