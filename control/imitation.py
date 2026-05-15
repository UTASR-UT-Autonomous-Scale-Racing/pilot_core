"""
Shared parts of the learned depth controllers (vae, bc): preprocessing, DART demos, ONNX export, on-car follower.

Ref: M. Laskey et al., "DART: Noise Injection for Robust Imitation Learning", CoRL 2017.
ONNX contract: "depth" (N, 1, INPUT_H, INPUT_W) in [0, 1] -> "steering" (N, 1), +1 = left; sidecar needs max_range.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .types import Command

INPUT_H, INPUT_W = 40, 64


def preprocess_depth(depth: np.ndarray, max_range: float) -> np.ndarray:
    """Depth (H, W) in metres -> (INPUT_H, INPUT_W) float32 in [0, 1]."""
    d = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0)
    d = np.clip(d, 0.0, max_range) / max_range
    return cv2.resize(d, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)


# demo collection

RENDER_W, RENDER_H = 160, 100   # rendered depth size (px)
CAM_HEIGHT = 0.12               # m, camera height above ground on the bench


def bench_intrinsics(settings: str = "orbslam_oakd_rgbd.yaml"):
    """OAK-D RGB intrinsics, scaled from 640x400 down to the 160x100 render."""
    from .depth_ftg import Intrinsics

    K = Intrinsics.from_orbslam_settings(settings)
    return Intrinsics(K.fx * RENDER_W / 640, K.fy * RENDER_H / 400,
                      K.cx * RENDER_W / 640, K.cy * RENDER_H / 400)


def collect_depth_demos(params, K, n_tracks: int, seconds: float, speed: float, max_range: float,
                        seed: int = 0, noise_std: float = 0.3, dt: float = 0.01, rate: float = 15.0,
                        verbose: bool = True):
    """LQR expert demos with DART noise -> x (N, 1, INPUT_H, INPUT_W) float32, y (N, 1), mirrors included."""
    from .controllers import build_track_follower
    from .sim import BicyclePlant, PlantConfig, random_track, render_depth
    from .track import TrackModel
    from .types import VehicleState

    rng = np.random.default_rng(seed)
    frames, labels = [], []
    for k in range(n_tracks):
        outer, inner, _ = random_track(rng)
        expert = build_track_follower("lqr", TrackModel(outer, inner), params, speed_cfg={"v_max": speed})
        tr = expert.track
        car = BicyclePlant(params, PlantConfig(), *tr.points[0], tr.heading[0])
        noise, t, next_frame = 0.0, 0.0, 0.0
        while t < seconds:
            cmd = expert.step(VehicleState(t, car.x, car.y, car.yaw, car.v))
            if t >= next_frame:
                d = render_depth(car.x, car.y, car.yaw, [outer, inner], K, RENDER_W, RENDER_H,
                                 cam_height=CAM_HEIGHT, max_range=max_range)
                frames.append(preprocess_depth(d, max_range))
                labels.append(cmd.steering)
                next_frame += 1.0 / rate
            noise = 0.95 * noise + rng.normal(0, noise_std * np.sqrt(1 - 0.95 ** 2))
            cmd.steering = float(np.clip(cmd.steering + noise, -1, 1))
            car.step(cmd, t, dt)
            t += dt
        if verbose:
            print(f"  track {k + 1}/{n_tracks}: {len(frames)} frames")
    x = np.stack(frames)[:, None]
    y = np.array(labels, dtype=np.float32)[:, None]
    return np.concatenate([x, x[..., ::-1]]).astype(np.float32), np.concatenate([y, -y])


def export_steering_onnx(model, path: str, meta: dict) -> str:
    """torch module -> ONNX (depth (N,1,H,W) -> steering (N,1)), plus the JSON sidecar."""
    import torch

    torch.onnx.export(model.eval(), torch.zeros(1, 1, INPUT_H, INPUT_W), path,
                      input_names=["depth"], output_names=["steering"],
                      dynamic_axes={"depth": {0: "batch"}, "steering": {0: "batch"}},
                      opset_version=17, dynamo=False)
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump({"input": [INPUT_H, INPUT_W], **meta}, f, indent=2)
    return path


@dataclass
class DepthPolicyConfig:
    cruise_speed: float = 0.8          # m/s
    speed_per_throttle: float = 3.0    # m/s per unit throttle, shared calibration
    steer_slowdown: float = 0.5        # speed *= 1 - steer_slowdown * |steering|
    min_valid_fraction: float = 0.2    # below this share of valid depth pixels -> neutral
    stop_distance: float = 0.35        # m, obstacle this close ahead (upper-centre of the image) -> neutral


class DepthPolicyFollower:
    """Runs an exported depth -> steering ONNX model on the car. No map needed."""

    def __init__(self, onnx_path: str, cfg: DepthPolicyConfig | None = None) -> None:
        import onnxruntime as ort

        from control.types import resolve_path

        onnx_path = resolve_path(onnx_path)
        with open(os.path.splitext(onnx_path)[0] + ".json") as f:
            self.meta = json.load(f)
        self.max_range = float(self.meta["max_range"])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts, providers=["CPUExecutionProvider"])
        self.cfg = cfg or DepthPolicyConfig()

    def reset(self) -> None:
        pass

    def steer(self, depth: np.ndarray) -> float:
        x = preprocess_depth(depth, self.max_range)[None, None]
        return float(self.session.run(None, {"depth": x})[0][0, 0])

    def step(self, depth: np.ndarray) -> Command:
        c = self.cfg
        valid = np.isfinite(depth) & (depth > 0.05)
        if valid.mean() < c.min_valid_fraction:
            return Command(0.0, 0.0)
        # centre columns, above the image middle, so the floor in front doesn't count
        h, w = depth.shape
        win = (slice(h // 4, h // 2), slice(w // 3, 2 * w // 3))
        centre = depth[win][valid[win]]
        if centre.size and np.percentile(centre, 5) < c.stop_distance:
            return Command(0.0, 0.0)
        steering = float(np.clip(self.steer(depth), -1.0, 1.0))
        speed = c.cruise_speed * (1.0 - c.steer_slowdown * abs(steering))
        return Command(steering, speed / c.speed_per_throttle).clipped()
