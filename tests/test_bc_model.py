"""BC model (control/bc.py) and the imitation helpers in control/imitation.py.

Full training (scripts/train_bc_policy.py) is too slow to run here, the script
reports its own closed-loop results. Needs torch, onnx and onnxruntime.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from control import VehicleParams  # noqa: E402
from control.bc import build_model, export_onnx  # noqa: E402
from control.imitation import (  # noqa: E402
    INPUT_H, INPUT_W, DepthPolicyFollower, bench_intrinsics, collect_depth_demos,
)


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def test_bc_model_shape_and_learns():
    torch.manual_seed(0)
    model = build_model()
    x = torch.rand(128, 1, INPUT_H, INPUT_W)
    y = torch.tanh(x[:, :, :, :32].mean(dim=(1, 2, 3)) * 8 - x[:, :, :, 32:].mean(dim=(1, 2, 3)) * 8)[:, None]
    assert model(x).shape == (128, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(80):
        loss = torch.mean((model(x) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < 0.3 * losses[0]


def test_demonstrations_shapes_and_mirroring():
    x, y = collect_depth_demos(VehicleParams.load(), bench_intrinsics(), n_tracks=1, seconds=2.0,
                               speed=0.8, max_range=4.0, verbose=False)
    n = len(x) // 2
    assert x.shape[1:] == (1, INPUT_H, INPUT_W) and y.shape == (2 * n, 1)
    np.testing.assert_array_equal(x[n:], x[:n][..., ::-1])   # mirrored frames...
    np.testing.assert_array_equal(y[n:], -y[:n])             # ...with negated steering
    assert 0.0 <= x.min() and x.max() <= 1.0


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(1)
    model = build_model().eval()
    path = str(tmp_path_factory.mktemp("bc") / "bc.onnx")
    export_onnx(model, path, {"max_range": 4.0})
    return path, model


def test_onnx_matches_torch(exported):
    import onnxruntime as ort

    path, model = exported
    x = torch.rand(4, 1, INPUT_H, INPUT_W)
    with torch.no_grad():
        ref = model(x).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"depth": x.numpy()})[0]
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_control_worker_bc_mode(exported):
    from shared_buffer import ControlBuffer, FrameBuffer
    from workers import ControlWorker

    fb = FrameBuffer(slots=2, name=f"bc_f_{_uid()}", create=True)
    cb = ControlBuffer(slots=2, name=f"bc_c_{_uid()}", create=True)
    try:
        w = ControlWorker(name="control", mode="bc", output_buffer=cb, frame_buffer=fb,
                          bc_config={"model": exported[0]})
        w._logger = MagicMock()
        w.setup()
        assert isinstance(w._depth_follower, DepthPolicyFollower)
        fb.write(np.zeros((400, 640, 3), np.uint8), np.full((400, 640), 2.0, np.float32))
        w.step()
        _, steer_u, thr_u = cb.read_latest()
        assert 0.0 <= steer_u <= 1.0 and thr_u > 0.5
    finally:
        fb.unlink()
        cb.unlink()
