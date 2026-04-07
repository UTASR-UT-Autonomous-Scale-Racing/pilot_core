"""
WebPublishWorker — optional worker that periodically publishes the car's
live telemetry (pose, trajectory, track boundaries, control state) to a
remote HTTP endpoint as JSON.

Designed for a pit-lane dashboard or spectator web-app that visualises
the vehicle's position on the mapped track in near-real-time.

Config kwargs
-------------
buffers : dict[str, Any] | None
    Optional canonical buffer map (same naming as BufferRecorderWorker):
    ``pose``, ``imu``, ``control``, ``pointcloud``, ``boundary``.
    If present, values are used first.
pose_buffer : PoseBuffer
    Legacy pose buffer kwarg (used when ``buffers['pose']`` is absent).
imu_buffer : IMUBuffer | None
    Legacy IMU buffer kwarg (used when ``buffers['imu']`` is absent).
pointcloud_buffer : PointCloudBuffer | None
    Legacy pointcloud buffer kwarg (used when ``buffers['pointcloud']`` is absent).
boundary_buffer : TrackBoundaryBuffer | None
    Legacy boundary buffer kwarg (used when ``buffers['boundary']`` is absent).
control_buffer : ControlBuffer | None
    Legacy control buffer kwarg (used when ``buffers['control']`` is absent).
endpoint : str
    Full URL to POST JSON payloads to (e.g. ``http://10.0.0.1:8080/api/telemetry``).
publish_interval : float
    Seconds between publishes (default 1.0).
timeout : float
    HTTP POST timeout in seconds (default 2.0).
vehicle_id : str
    Arbitrary identifier for the vehicle (default ``"pilot_1"``).
max_trajectory_pts : int
    Cap on how many trajectory points to include per payload (default 500).
max_pointcloud_pts : int
    Cap on how many point-cloud points to include per payload (default 2000).
include_buffers : list[str] | str | None
    Optional list of payload sections to include. Supported values:
    ``pose, trajectory, control, boundary, imu, pointcloud``.
    ``track`` is accepted as a legacy alias for ``boundary``.
    If omitted, defaults to ``pose, trajectory, control, boundary``.
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

# ---------------------------------------------------------------------------
# Lazy imports for typed buffers — they are passed in at runtime via kwargs.
# We use TYPE_CHECKING to keep type hints available for editors without
# requiring the imports at runtime (workers receive buffers as kwargs).
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shared_buffer.imu_buffer import IMUBuffer
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
    from shared_buffer.pose_buffer import PoseBuffer
    from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer
    from shared_buffer.control_buffer import ControlBuffer


class WebPublishWorker(BaseWorker):
    """Periodically POST live telemetry to an HTTP endpoint.

    Example JSON payload::

        {
            "vehicle_id": "pilot_1",
            "timestamp": 1718900000.123,
            "pose": {
                "x": 1.23, "z": 4.56, "heading_deg": 90.0
            },
            "trajectory": [[x, z], ...],
            "control": {"steering": 0.52, "throttle": 0.30},
            "track": {
                "outer": [[x, z], ...],
                "inner": [[x, z], ...]
            }
        }

    Parameters are documented at module level.
    """

    _payload_number: int = 0

    def __init__(self, name: str = "web_publish", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)

        # Buffer handles (set in setup)
        self._pose_buf: Optional["PoseBuffer"] = None
        self._imu_buf: Optional["IMUBuffer"] = None
        self._pc_buf: Optional["PointCloudBuffer"] = None
        self._boundary_buf: Optional["TrackBoundaryBuffer"] = None
        self._ctrl_buf: Optional["ControlBuffer"] = None

        # Config
        self._endpoint: str = ""
        self._interval: float = 1.0
        self._timeout: float = 2.0
        self._vehicle_id: str = "pilot_1"
        self._max_traj: int = 500
        self._max_pc_pts: int = 2000
        self._include: set[str] = {"control", "imu"}

        # Internal state
        self._trajectory: List[List[float]] = []
        self._latest_pose: Optional[Dict[str, float]] = None
        self._last_pose_seq: int = 0
        self._last_publish: float = 0.0
        self._consecutive_errors: int = 0
        self._MAX_BACKOFF: float = 30.0  # max sleep on repeated failures

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        buffers_cfg = self.config.get("buffers") or {}

        self._pose_buf = buffers_cfg.get("pose", self.config.get("pose_buffer"))
        self._imu_buf = buffers_cfg.get("imu", self.config.get("imu_buffer"))
        self._pc_buf = buffers_cfg.get("pointcloud", self.config.get("pointcloud_buffer"))
        self._boundary_buf = buffers_cfg.get("boundary", self.config.get("boundary_buffer"))
        self._ctrl_buf = buffers_cfg.get("control", self.config.get("control_buffer"))
        self._endpoint = self.config.get("endpoint", "http://localhost:8080/api/telemetry")
        self._interval = float(self.config.get("publish_interval", 1.0))
        self._timeout = float(self.config.get("timeout", 2.0))
        self._vehicle_id = self.config.get("vehicle_id", "pilot_1")
        self._max_traj = int(self.config.get("max_trajectory_pts", 500))
        self._max_pc_pts = int(self.config.get("max_pointcloud_pts", 2000))

        include_cfg = self.config.get("include_buffers")
        if include_cfg is None:
            include_set = {"pose", "trajectory", "control", "boundary", "imu", "pointcloud"}#TODO: Abstract default include set to a class-level constant or config default
        else:
            if isinstance(include_cfg, str):
                parts = [p.strip() for p in include_cfg.split(",") if p.strip()]
            else:
                parts = [str(p).strip() for p in include_cfg if str(p).strip()]
            include_set = {p.lower() for p in parts}

        # Backward compatibility for older include names.
        include_aliases = {
            "track": "boundary",
            "boundaries": "boundary",
        }
        include_set = {include_aliases.get(p, p) for p in include_set}

        supported = {"pose", "trajectory", "control", "boundary", "imu", "pointcloud"}
        unknown = include_set - supported
        if unknown:
            self.logger.warning("Unknown include_buffers entries ignored: %s", sorted(unknown))
        self._include = include_set & supported
        if not self._include:
            self._include = {"control", "imu"}

        if not self._endpoint:
            raise ValueError("WebPublishWorker requires a non-empty 'endpoint'")

        self.logger.info(
            f"WebPublishWorker ready — endpoint={self._endpoint}, "
            f"interval={self._interval}s, vehicle={self._vehicle_id}, "
            f"include={sorted(self._include)}"
        )

    def step(self) -> None:
        # Accumulate trajectory from pose buffer every step
        if "pose" in self._include or "trajectory" in self._include:
            self._ingest_pose()

        # Only publish at the configured interval
        now = time.monotonic()
        if now - self._last_publish < self._interval:
            time.sleep(0.05)  # yield CPU
            return

        # Build & send payload
        payload = self._build_payload()
        self._post(payload)
        self._last_publish = now

    def teardown(self) -> None:
        self.logger.info(
            f"WebPublishWorker shutting down — "
            f"{len(self._trajectory)} trajectory pts accumulated"
        )

    # ------------------------------------------------------------------
    # Data ingestion helpers
    # ------------------------------------------------------------------

    def _ingest_pose(self) -> None:
        """Read the latest pose and append (X, Z) to the trajectory."""
        if self._pose_buf is None:
            return
        result = self._pose_buf.read_latest()
        if result is None:
            return
        header, pose, _ = result
        if header.seq_no <= self._last_pose_seq:
            return
        self._last_pose_seq = header.seq_no

        # Extract translation from 4×4 matrix
        tx, tz = float(pose[0, 3]), float(pose[2, 3])
        self._latest_pose = {"x": tx, "z": tz}

        if "trajectory" in self._include:
            self._trajectory.append([tx, tz])

        # Cap to the most recent N points
        if len(self._trajectory) > self._max_traj:
            self._trajectory = self._trajectory[-self._max_traj:]

    def _read_control(self) -> Optional[Dict[str, float]]:
        """Snapshot latest steering + throttle."""
        if self._ctrl_buf is None:
            return None
        result = self._ctrl_buf.read_latest()
        if result is None:
            return None
        _header, steering, throttle = result
        return {"steering": round(float(steering), 4),
                "throttle": round(float(throttle), 4)}

    def _read_boundaries(self) -> Optional[Dict[str, list]]:
        """Snapshot outer / inner track boundaries."""
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

    def _read_imu(self) -> Optional[Dict[str, Any]]:
        """Snapshot latest IMU vector."""
        if self._imu_buf is None:
            return None
        result = self._imu_buf.read_latest()
        if result is None:
            return None
        _header, imu_vec = result
        if len(imu_vec) < 9:
            return None

        # Canonical IMU layout (16 floats):
        # [w, x, y, z, ax, ay, az, gx, gy, gz, mx, my, mz, throttle, steering, timestamp]
        if len(imu_vec) >= 16:
            return {
                "quat": [
                    round(float(imu_vec[0]), 6),
                    round(float(imu_vec[1]), 6),
                    round(float(imu_vec[2]), 6),
                    round(float(imu_vec[3]), 6),
                ],
                "accel": [
                    round(float(imu_vec[4]), 6),
                    round(float(imu_vec[5]), 6),
                    round(float(imu_vec[6]), 6),
                ],
                "gyro": [
                    round(float(imu_vec[7]), 6),
                    round(float(imu_vec[8]), 6),
                    round(float(imu_vec[9]), 6),
                ],
                "mag": [
                    round(float(imu_vec[10]), 6),
                    round(float(imu_vec[11]), 6),
                    round(float(imu_vec[12]), 6),
                ],
                "raw_control": {
                    "throttle": round(float(imu_vec[13]), 6),
                    "steering": round(float(imu_vec[14]), 6),
                },
                "sensor_timestamp": round(float(imu_vec[15]), 6),
            }

        # Backward-compatible fallback for 9-float vectors.
        return {
            "quat": [
                round(float(imu_vec[0]), 6),
                round(float(imu_vec[1]), 6),
                round(float(imu_vec[2]), 6),
                round(float(imu_vec[3]), 6),
            ],
            "accel": [
                round(float(imu_vec[4]), 6),
                round(float(imu_vec[5]), 6),
                round(float(imu_vec[6]), 6),
            ],
            "raw_control": {
                "throttle": round(float(imu_vec[7]), 6),
                "steering": round(float(imu_vec[8]), 6),
            },
        }

    def _read_pointcloud(self) -> Optional[Dict[str, Any]]:
        """Snapshot latest point cloud, capped for network payload size."""
        if self._pc_buf is None:
            return None
        result = self._pc_buf.read_latest()
        if result is None:
            return None
        _header, points = result
        count = int(len(points))
        if count <= 0:
            return {"count": 0, "points": []}
        clipped = points[: self._max_pc_pts]
        return {
            "count": count,
            "points": _ndarray_to_list(clipped, decimals=4),
        }

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_payload(self) -> Dict[str, Any]:
        """Assemble the JSON-serialisable telemetry dict."""
        WebPublishWorker._payload_number += 1
        payload: Dict[str, Any] = {
            "vehicle_id": self._vehicle_id,
            "timestamp": time.time(),
            "payload_number": WebPublishWorker._payload_number,
        }
            
        # Current pose
        if "pose" in self._include and self._latest_pose is not None:
            payload["pose"] = self._latest_pose

        # Trajectory
        if "trajectory" in self._include:
            payload["trajectory"] = self._trajectory

        # Control
        if "control" in self._include:
            ctrl = self._read_control()
            if ctrl is not None:
                payload["control"] = ctrl

        # Track boundaries (canonical key is "boundary").
        if "boundary" in self._include:
            boundary = self._read_boundaries()
            if boundary is not None:
                payload["boundary"] = boundary
                # Keep legacy key for existing consumers.
                payload["track"] = boundary

        # IMU
        if "imu" in self._include:
            imu = self._read_imu()
            if imu is not None:
                payload["imu"] = imu

        # Point cloud
        if "pointcloud" in self._include:
            pc = self._read_pointcloud()
            if pc is not None:
                payload["pointcloud"] = pc

        return payload

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def _post(self, payload: Dict[str, Any]) -> None:
        """POST JSON payload to the configured endpoint.

        Uses stdlib ``urllib`` to avoid an external dependency on
        ``requests``.  Applies exponential back-off on repeated failures
        so a flaky network doesn't flood the logs.
        """
        body = json.dumps(payload, default=_json_default).encode("utf-8")
        req = Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": "application/json", "x-api-key": "dev-secret"}, #TODO Set Env var for password
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

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        if self._logger is None:
            self._logger = logging.getLogger(f"worker.{self.name}")
        return self._logger


# ======================================================================
# Module-level helpers
# ======================================================================

def _ndarray_to_list(arr: np.ndarray, decimals: int = 4) -> list:
    """Convert an (N, 2) float32 ndarray to a JSON-friendly nested list.

    Rounds to *decimals* decimal places to keep payloads compact.
    """
    if arr is None or len(arr) == 0:
        return []
    return np.round(arr, decimals=decimals).tolist()


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for ``json.dumps`` — handles numpy scalars."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
