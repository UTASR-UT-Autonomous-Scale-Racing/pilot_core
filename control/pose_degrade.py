"""
Degrade the sim's ground-truth pose to look like a visual SLAM pose on the car.

The sim's pose is exact and on time; ORB-SLAM3's is late, noisy, drifts, and stops while
tracking is lost. Each pose is published late (a frame's transfer plus a log-normal tracking
time), in order, still stamped with its capture time; it carries white noise and a slowly
wandering position bias; and during dropouts nothing is published.

The defaults come from ORB-SLAM3 RGB-D (the CUDA fork) replaying rendered sequences of the
compete track in real time (bench/2026-10-29_vio_sim/inertial_fix.md): tracking time p50
8.4 ms / p95 41.4 ms, ATE 0.107 m over three laps, and lost stretches around featureless
frames. They are not measured on the car; refit them from car recordings once those exist.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PoseDegradeConfig:
    enabled: bool = False
    base_latency_s: float = 0.033     # frame transfer, one 30 fps period
    latency_median_s: float = 0.0084  # tracking time
    latency_p95_s: float = 0.0414
    xy_sigma: float = 0.01            # m, white noise per pose
    yaw_sigma: float = 0.005          # rad, white noise per pose
    bias_rw: float = 0.03             # m / sqrt(s), random walk of the position bias
    dropout_rate: float = 0.15        # tracking losses per second
    dropout_s: float = 0.3            # s, length of each
    seed: int = 0


class PoseDegrader:
    """add() each ground-truth pose (Z-up world, 4x4) as it arrives; due() returns the poses
    whose publication time has come, as (capture time, pose), oldest first."""

    def __init__(self, cfg: PoseDegradeConfig | None = None) -> None:
        self.cfg = cfg or PoseDegradeConfig()
        c = self.cfg
        self._rng = np.random.default_rng(c.seed)
        self._mu = math.log(max(c.latency_median_s, 1e-6))
        self._sigma = max(math.log(max(c.latency_p95_s, c.latency_median_s) / max(c.latency_median_s, 1e-6)) / 1.645,
                          0.0)
        self._queue: list[tuple[float, int, float, np.ndarray]] = []    # (release, n, capture, pose)
        self._n = 0
        self._last_release = -math.inf
        self._last_t: float | None = None
        self._bias = np.zeros(2)
        self._lost_until = -math.inf
        self.published = 0
        self.dropped = 0

    def _latency(self) -> float:
        return self.cfg.base_latency_s + float(math.exp(self._mu + self._sigma * self._rng.standard_normal()))

    def add(self, t: float, pose: np.ndarray) -> None:
        c = self.cfg
        dt = 0.0 if self._last_t is None else max(t - self._last_t, 0.0)
        self._last_t = t
        self._bias += self._rng.normal(0.0, c.bias_rw * math.sqrt(dt), 2)
        if t < self._lost_until:
            self.dropped += 1
            return
        if dt > 0 and self._rng.random() < 1.0 - math.exp(-c.dropout_rate * dt):
            self._lost_until = t + c.dropout_s
            self.dropped += 1
            return
        out = np.array(pose, dtype=np.float32, copy=True)
        out[0, 3] += self._bias[0] + self._rng.normal(0.0, c.xy_sigma)
        out[1, 3] += self._bias[1] + self._rng.normal(0.0, c.xy_sigma)
        a = self._rng.normal(0.0, c.yaw_sigma)
        ca, sa = math.cos(a), math.sin(a)
        Rz = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        out[:3, :3] = Rz @ out[:3, :3]
        # tracking is serial: a pose is never published before the one captured before it
        release = max(t + self._latency(), self._last_release)
        self._last_release = release
        self._n += 1
        heapq.heappush(self._queue, (release, self._n, t, out))

    def due(self, now: float) -> list[tuple[float, np.ndarray]]:
        ready = []
        while self._queue and self._queue[0][0] <= now:
            _, _, t, pose = heapq.heappop(self._queue)
            ready.append((t, pose))
        self.published += len(ready)
        return ready
