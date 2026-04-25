"""
Optional: POSTs live telemetry (pose, trajectory, track boundaries, control)
as JSON to some HTTP endpoint every so often. Meant for a pit-lane dashboard
or similar.

Positions are the (x, z) translation of the camera-to-world pose from
PoseBuffer, i.e. the ground plane of the ORB-SLAM3 camera frame.

Config:
    pose_buffer         Twc poses from SLAM
    boundary_buffer     optional, left out when SLAM is disabled
    control_buffer      optional; steering/throttle go out as-is, in [0, 1]
                        with 0.5 = neutral
    endpoint            full URL, e.g. http://10.0.0.1:8080/api/telemetry
    publish_interval    seconds, default 1.0
    timeout             POST timeout in seconds, default 2.0
    vehicle_id          any string, default "pilot_1"
    max_trajectory_pts  only the newest N trajectory points are sent (500)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

import numpy as np

from .base_worker import BaseWorker

# type hints only, the buffers themselves come in as kwargs
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared_buffer.pose_buffer import PoseBuffer
    from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer
    from shared_buffer.control_buffer import ControlBuffer


class WebPublishWorker(BaseWorker):
    """Telemetry POSTer. Config is described in the module docstring.

    A payload looks like::

        {
            "vehicle_id": "pilot_1",
            "timestamp": 1718900000.123,
            "pose": {"x": 1.23, "z": 4.56},
            "trajectory": [[x, z], ...],
            "control": {"steering": 0.52, "throttle": 0.60},
            "track": {
                "outer": [[x, z], ...],
                "inner": [[x, z], ...]
            }
        }
    """

    def __init__(self, name: str = "web_publish", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        self._pose_buf: Optional["PoseBuffer"] = None
        self._boundary_buf: Optional["TrackBoundaryBuffer"] = None
        self._ctrl_buf: Optional["ControlBuffer"] = None

        self._endpoint: str = ""
        self._interval: float = 1.0
        self._timeout: float = 2.0
        self._vehicle_id: str = "pilot_1"
        self._max_traj: int = 500

        self._trajectory: List[List[float]] = []
        self._last_pose_seq: int = 0
        self._last_publish: float = 0.0
        self._consecutive_errors: int = 0
        self._MAX_BACKOFF: float = 30.0  # s, cap on the failure back-off

    def setup(self) -> None:
        self._pose_buf = self.config.get("pose_buffer")
        self._boundary_buf = self.config.get("boundary_buffer")
        self._ctrl_buf = self.config.get("control_buffer")
        self._endpoint = self.config.get("endpoint", "http://localhost:8080/api/telemetry")
        self._interval = float(self.config.get("publish_interval", 1.0))
        self._timeout = float(self.config.get("timeout", 2.0))
        self._vehicle_id = self.config.get("vehicle_id", "pilot_1")
        self._max_traj = int(self.config.get("max_trajectory_pts", 500))
        self._bev_axes = tuple(self.config.get("bev_axes", [0, 2]))

        if not self._endpoint:
            raise ValueError("WebPublishWorker requires a non-empty 'endpoint'")

        self.logger.info(
            f"WebPublishWorker ready — endpoint={self._endpoint}, "
            f"interval={self._interval}s, vehicle={self._vehicle_id}"
        )

    def step(self) -> None:
        self._ingest_pose()

        now = time.monotonic()
        if now - self._last_publish < self._interval:
            time.sleep(0.05)
            return

        payload = self._build_payload()
        self._post(payload)
        self._last_publish = now

    def teardown(self) -> None:
        self.logger.info(
            f"WebPublishWorker shutting down — "
            f"{len(self._trajectory)} trajectory pts accumulated"
        )

    def _ingest_pose(self) -> None:
        """Append the newest pose's ground-plane position, if there is a new one."""
        if self._pose_buf is None:
            return
        result = self._pose_buf.read_latest()
        if result is None:
            return
        header, pose, _, _ = result
        if header.seq_no <= self._last_pose_seq:
            return
        self._last_pose_seq = header.seq_no

        a0, a1 = self._bev_axes
        self._trajectory.append([float(pose[a0, 3]), float(pose[a1, 3])])

        if len(self._trajectory) > self._max_traj:
            self._trajectory = self._trajectory[-self._max_traj:]

    def _read_control(self) -> Optional[Dict[str, float]]:
        if self._ctrl_buf is None:
            return None
        result = self._ctrl_buf.read_latest()
        if result is None:
            return None
        _header, steering, throttle = result
        return {"steering": round(float(steering), 4),
                "throttle": round(float(throttle), 4)}

    def _read_boundaries(self) -> Optional[Dict[str, list]]:
        if self._boundary_buf is None:
            return None
        result = self._boundary_buf.read_latest()
        if result is None:
            return None
        _header, outer, inner = result
        return {
            "outer": _ndarray_to_list(outer),
            "inner": _ndarray_to_list(inner),
        }

    def _build_payload(self) -> Dict[str, Any]:
        # pose/control/track keys are left out when there's nothing to send
        payload: Dict[str, Any] = {
            "vehicle_id": self._vehicle_id,
            "timestamp": time.time(),
        }

        if self._trajectory:
            last = self._trajectory[-1]
            payload["pose"] = {"x": last[0], "z": last[1]}

        payload["trajectory"] = self._trajectory

        ctrl = self._read_control()
        if ctrl is not None:
            payload["control"] = ctrl

        track = self._read_boundaries()
        if track is not None:
            payload["track"] = track

        return payload

    def _post(self, payload: Dict[str, Any]) -> None:
        """POST to the endpoint with plain urllib (no requests dependency).

        Failures back off exponentially, capped at _MAX_BACKOFF, so a flaky
        network doesn't flood the log.
        """
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        req = Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=self._timeout) as resp:
                status = resp.status
            self._consecutive_errors = 0
            self.logger.debug(
                f"Published {len(body)} bytes → {self._endpoint} "
                f"(HTTP {status})"
            )
        except (URLError, OSError, TimeoutError) as exc:
            self._consecutive_errors += 1
            backoff = min(
                self._interval * (2 ** self._consecutive_errors),
                self._MAX_BACKOFF,
            )
            self.logger.warning(
                f"POST failed ({exc}) — "
                f"back-off {backoff:.1f}s "
                f"(consecutive errors: {self._consecutive_errors})"
            )
            time.sleep(backoff)

    # lazy so it also works before run()
    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(f"worker.{self.name}")
        return self._logger


def _ndarray_to_list(arr: np.ndarray, decimals: int = 4) -> list:
    """(N, 2) array -> nested list, rounded to keep the payload small."""
    if arr is None or len(arr) == 0:
        return []
    return np.round(arr, decimals=decimals).tolist()


def _json_default(obj: Any) -> Any:
    """json.dumps default= hook for numpy scalars/arrays."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
