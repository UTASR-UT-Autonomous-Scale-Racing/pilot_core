"""
VAE control model (control/vae.py). Training itself (scripts/train_vae_policy.py)
is too slow for unit tests; the script reports closed-loop results.
Skipped unless torch, onnx and onnxruntime are installed.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from control.vae import (  # noqa: E402
    INPUT_H, INPUT_W, VAEPolicyFollower, build_models, export_onnx, preprocess_depth, vae_loss,
)


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def test_preprocess_shape_range_and_invalid():
    d = np.full((400, 640), 2.0, np.float32)
    d[:10] = np.nan
    d[10:20] = 50.0
    x = preprocess_depth(d, max_range=4.0)
    assert x.shape == (INPUT_H, INPUT_W) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


def test_model_shapes_and_training_reduces_loss():
    torch.manual_seed(0)
    vae, head = build_models(latent_dim=8)
    x = torch.rand(64, 1, INPUT_H, INPUT_W) * 0.5
    x[:, :, :, :32] += 0.4                    # some structure to learn
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape and mu.shape == (64, 8)
    assert head(mu).shape == (64, 1)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    first = None
    for _ in range(60):
        recon, mu, logvar = vae(x)
        loss, _, _ = vae_loss(recon, x, mu, logvar)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.5 * first


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(1)
    vae, head = build_models(latent_dim=8)
    path = str(tmp_path_factory.mktemp("vae") / "vae.onnx")
    export_onnx(vae.eval(), head.eval(), path, {"max_range": 4.0, "latent_dim": 8})
    return path, vae, head


def test_onnx_matches_torch(exported):
    import onnxruntime as ort

    path, vae, head = exported
    x = torch.rand(5, 1, INPUT_H, INPUT_W)
    with torch.no_grad():
        ref = head(vae.encode(x)[0]).numpy()
    out = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"depth": x.numpy()})[0]
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_follower_command(exported):
    f = VAEPolicyFollower(exported[0])
    cmd = f.step(np.random.uniform(0.3, 4.0, (400, 640)).astype(np.float32))
    assert -1.0 <= cmd.steering <= 1.0
    assert 0.0 < cmd.throttle <= f.cfg.cruise_speed / f.cfg.speed_per_throttle + 1e-9


def test_control_worker_vae_mode(exported):
    from shared_buffer import ControlBuffer, FrameBuffer
    from workers import ControlWorker

    fb = FrameBuffer(slots=2, name=f"vm_f_{_uid()}", create=True)
    cb = ControlBuffer(slots=2, name=f"vm_c_{_uid()}", create=True)
    try:
        w = ControlWorker(name="control", mode="vae", output_buffer=cb, frame_buffer=fb,
                          vae_config={"model": exported[0]})
        w._logger = MagicMock()
        w.setup()
        fb.write(np.zeros((400, 640, 3), np.uint8), np.full((400, 640), 2.0, np.float32))
        w.step()
        _, steer_u, thr_u = cb.read_latest()
        assert 0.0 <= steer_u <= 1.0 and thr_u > 0.5
    finally:
        fb.unlink()
        cb.unlink()
