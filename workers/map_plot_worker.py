"""
MapPlotWorker — real-time matplotlib visualisation of the BEV track map.

Two visual phases:
    **Phase A - Mapping:**
        • Projected 3D point-cloud points rendered as **red** scatter dots.
        • Vehicle trajectory drawn as a thin grey line.
        • Title says "Mapping …"

    **Phase B - Driving (after loop closure):**
        • Plot is refreshed: inner/outer boundary lines drawn in white/cyan.
        • Car position shown as a green triangle.
        • Position trail drawn as a coloured line that **fades** after each lap
          (alpha decreases across successive laps).
        • Title shows current lap number.

The worker runs in a dedicated process and uses ``matplotlib.animation``
with a non-blocking backend (``Agg`` + ``FigureCanvasAgg`` blitted to a
Tk window when available, otherwise saves PNGs to disk as fallback).

Config kwargs
-------------
pose_buffer : PoseBuffer
pointcloud_buffer : PointCloudBuffer
boundary_buffer : TrackBoundaryBuffer
loop_event : mp.Event
mapping_done_event : mp.Event
fps : int            – plot refresh rate (default 10)
save_dir : str       – directory for saved snapshots (default /tmp/pilot_map)
max_trail_laps : int – number of laps before oldest trail fully fades (default 5)
save_video : bool    – record every frame into an MP4 video (default True)
video_filename : str – output video filename (default map_recording.mp4)
track_length : float – expected track length for lap detection heuristic
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import time
from typing import Any, Optional

import numpy as np

from .base_worker import BaseWorker
from shared_buffer import PoseBuffer

try:
    from shared_buffer.pointcloud_buffer import PointCloudBuffer
except ImportError:
    PointCloudBuffer = None  # type: ignore[misc]

try:
    from shared_buffer.track_boundary_buffer import TrackBoundaryBuffer
except ImportError:
    TrackBoundaryBuffer = None  # type: ignore[misc]


class MapPlotWorker(BaseWorker):
    """Real-time BEV map plotter running in its own process."""

    def __init__(self, name: str = "map_plot", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self._pose_buf: PoseBuffer | None = None
        self._pc_buf: Optional[Any] = None
        self._boundary_buf: Optional[Any] = None
        self._loop_event: Optional[mp.Event] = None
        self._mapping_done_event: Optional[mp.Event] = None

        # Internal state  populated during run
        self._fig = None
        self._ax = None

        # Data accumulation
        self._cloud_pts: list[np.ndarray] = []   # (N,2) BEV chunks
        self._traj_pts: list[np.ndarray] = []    # (2,) positions
        self._last_pose_seq = 0
        self._last_pc_seq = 0

        # Phase B state
        self._phase_b = False
        self._outer_line: Optional[np.ndarray] = None
        self._inner_line: Optional[np.ndarray] = None
        self._lap_trails: list[list[np.ndarray]] = [[]]  # per-lap trails
        self._current_lap = 0
        self._lap_start_pos: Optional[np.ndarray] = None
        self._lap_threshold = 1.0  # metres from start to count a new lap
        self._moved_away = False

        # Config
        self._target_dt = 0.1
        self._save_dir = "/tmp/pilot_map"
        self._max_trail_laps = 5
        self._last_render_time = 0.0
        self._headless = False

        # Video recording
        self._save_video = True
        self._video_writer = None
        self._video_path: str = ""
        self._video_fps = 10
        self._frame_count = 0

    def setup(self) -> None:
        self._pose_buf = self.config["pose_buffer"]
        self._pc_buf = self.config.get("pointcloud_buffer")
        self._boundary_buf = self.config.get("boundary_buffer")
        self._loop_event = self.config.get("loop_event")
        self._mapping_done_event = self.config.get("mapping_done_event")
        fps = self.config.get("fps", 10)
        self._target_dt = 1.0 / fps
        self._save_dir = self.config.get("save_dir", "/tmp/pilot_map")
        self._max_trail_laps = self.config.get("max_trail_laps", 5)
        self._save_video = self.config.get("save_video", True)
        self._video_fps = fps

        os.makedirs(self._save_dir, exist_ok=True)

        # Set up matplotlib — detect DISPLAY for interactive backend
        import matplotlib

        has_display = bool(os.environ.get("DISPLAY"))
        backend_set = False

        if has_display:
            for backend_name in ("TkAgg", "Qt5Agg"):
                try:
                    matplotlib.use(backend_name, force=True)
                    # Validate by importing the canvas
                    __import__(f"matplotlib.backends.backend_{backend_name.lower()}")
                    backend_set = True
                    break
                except Exception:
                    continue

        if not backend_set:
            matplotlib.use("Agg", force=True)
            self._headless = True

        import matplotlib.pyplot as plt
        self._plt = plt

        if not self._headless:
            plt.ion()  # Interactive mode
        self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 8))
        self._fig.set_facecolor("#1a1a2e")
        self._ax.set_facecolor("#1a1a2e")
        self._ax.set_aspect("equal")
        self._ax.set_title("Mapping …", color="white", fontsize=14, fontweight="bold")
        self._ax.tick_params(colors="white")
        for spine in self._ax.spines.values():
            spine.set_color("#333366")
        self._ax.grid(True, alpha=0.15, color="white")
        self._ax.set_xlabel("X (m)", color="white")
        self._ax.set_ylabel("Z (m)", color="white")

        if not self._headless:
            plt.show(block=False)
            plt.pause(0.01)

        # Set up video writer (OpenCV MP4)
        if self._save_video:
            try:
                import cv2
                video_name = self.config.get(
                    "video_filename", "map_recording.mp4"
                )
                self._video_path = os.path.join(self._save_dir, video_name)
                # Rasterize one frame to get the pixel dimensions
                self._fig.canvas.draw()
                w, h = self._fig.canvas.get_width_height()
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(
                    self._video_path, fourcc, self._video_fps, (w, h)
                )
                if self._video_writer.isOpened():
                    self.logger.info(
                        f"Video recording → {self._video_path} "
                        f"({w}×{h} @ {self._video_fps} fps)"
                    )
                else:
                    self.logger.warning(
                        f"Could not open video writer for {self._video_path}"
                    )
                    self._video_writer = None
            except ImportError:
                self.logger.warning(
                    "cv2 not available — video recording disabled"
                )
                self._video_writer = None

        self.logger.info(
            f"MapPlotWorker ready "
            f"(backend={matplotlib.get_backend()}, "
            f"headless={self._headless})"  
        )

    def step(self) -> None:
        #  Read point cloud         
        if self._pc_buf is not None:
            pc_result = self._pc_buf.read_latest()
            if pc_result is not None:
                pc_header, pts_3d = pc_result
                if pc_header.seq_no > self._last_pc_seq and len(pts_3d) > 0:
                    bev = pts_3d[:, [0, 2]]  # drop Y
                    self._cloud_pts.append(bev)
                    self._last_pc_seq = pc_header.seq_no

        #  Read pose; 
        pose_result = self._pose_buf.read_latest()
        if pose_result is not None:
            pose_header, pose, _ = pose_result
            if pose_header.seq_no > self._last_pose_seq:
                pos = np.array([pose[0, 3], pose[2, 3]], dtype=np.float32)
                self._traj_pts.append(pos)
                self._last_pose_seq = pose_header.seq_no

                # In phase B, track laps
                if self._phase_b:
                    self._lap_trails[self._current_lap].append(pos)
                    self._detect_lap(pos)

        #  Detect phase transition         
        if (
            not self._phase_b
            and self._loop_event is not None
            and self._loop_event.is_set()
        ):
            # Wait until mapping_done_event fires too,
            # OR read boundary buffer directly
            if self._boundary_buf is not None:
                bnd = self._boundary_buf.read_latest()
                if bnd is not None:
                    _, self._outer_line, self._inner_line = bnd
                    self._phase_b = True
                    # Set lap start position
                    if self._traj_pts:
                        self._lap_start_pos = self._traj_pts[-1].copy()
                    self.logger.info(
                        "Phase B — boundaries loaded, "
                        f"outer={len(self._outer_line)}, "
                        f"inner={len(self._inner_line)}"
                    )

        #  Render 
        now = time.monotonic()
        if now - self._last_render_time >= self._target_dt:
            self._render()
            self._last_render_time = now
        else:
            remaining = self._target_dt - (now - self._last_render_time)
            time.sleep(min(max(0, remaining), self._target_dt))

    def _detect_lap(self, pos: np.ndarray) -> None:
        """Detect when the car returns to the lap start position."""
        if self._lap_start_pos is None:
            return

        dist = np.linalg.norm(pos - self._lap_start_pos)

        if not self._moved_away and dist > self._lap_threshold * 3:
            self._moved_away = True

        if self._moved_away and dist < self._lap_threshold:
            self._current_lap += 1
            self._lap_trails.append([])
            self._moved_away = False
            self.logger.info(f"Lap {self._current_lap} completed")

    def _render(self) -> None:
        """Redraw the plot based on the current phase."""
        ax = self._ax
        ax.clear()

        # Style
        ax.set_facecolor("#1a1a2e")
        ax.set_aspect("equal")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333366")
        ax.grid(True, alpha=0.15, color="white")
        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Z (m)", color="white")

        if not self._phase_b:
            self._render_phase_a(ax)
        else:
            self._render_phase_b(ax)

        self._fig.canvas.draw_idle()
        self._fig.canvas.draw()  # ensure rasterised for video capture

        # Capture frame to video
        if self._video_writer is not None:
            self._write_video_frame()

        if self._headless:
            self._fig.savefig(
                os.path.join(self._save_dir, "map_latest.png"),
                dpi=100, facecolor=self._fig.get_facecolor(),
                bbox_inches="tight",
            )
        else:
            self._plt.pause(0.001)

    def _render_phase_a(self, ax) -> None:
        """Phase A — mapping: red scatter for projected points."""
        ax.set_title(
            "Mapping …", color="white", fontsize=14, fontweight="bold"
        )

        # Draw accumulated point cloud in red
        if self._cloud_pts:
            all_pts = np.vstack(self._cloud_pts)
            ax.scatter(
                all_pts[:, 0], all_pts[:, 1],
                s=1.5, c="red", alpha=0.6, rasterized=True,
            )

        # Draw trajectory so far (thin grey)
        if len(self._traj_pts) > 1:
            traj = np.array(self._traj_pts)
            ax.plot(
                traj[:, 0], traj[:, 1],
                color="#888888", linewidth=0.8, alpha=0.5,
            )

        # Car position (latest)
        if self._traj_pts:
            pos = self._traj_pts[-1]
            ax.plot(pos[0], pos[1], "wo", markersize=5, zorder=10)

        n_pts = sum(len(c) for c in self._cloud_pts)
        ax.text(
            0.02, 0.98,
            f"Points: {n_pts:,}  |  Poses: {len(self._traj_pts)}",
            transform=ax.transAxes, color="white", fontsize=9,
            verticalalignment="top", fontfamily="monospace",
        )

    def _render_phase_b(self, ax) -> None:
        """Phase B — driving: boundaries + fading lap trails."""
        ax.set_title(
            f"Driving — Lap {self._current_lap}",
            color="white", fontsize=14, fontweight="bold",
        )

        # Draw boundaries
        if self._outer_line is not None and len(self._outer_line) > 0:
            outer = np.vstack([self._outer_line, self._outer_line[:1]])
            ax.plot(
                outer[:, 0], outer[:, 1],
                color="#00ffff", linewidth=2.0, label="Outer boundary",
            )
        if self._inner_line is not None and len(self._inner_line) > 0:
            inner = np.vstack([self._inner_line, self._inner_line[:1]])
            ax.plot(
                inner[:, 0], inner[:, 1],
                color="white", linewidth=2.0, label="Inner boundary",
            )

        # Draw fading lap trails
        n_laps = len(self._lap_trails)
        for lap_idx, trail in enumerate(self._lap_trails):
            if len(trail) < 2:
                continue
            pts = np.array(trail)
            # Newer laps are brighter; older laps fade
            age = n_laps - 1 - lap_idx  # 0 = current, higher = older
            if age >= self._max_trail_laps:
                continue  # Too old, skip entirely
            alpha = max(0.1, 1.0 - (age / self._max_trail_laps) * 0.85)
            linewidth = 2.0 if age == 0 else 1.2
            color = "#ffa500" if age == 0 else "#ff6600"
            ax.plot(
                pts[:, 0], pts[:, 1],
                color=color, linewidth=linewidth, alpha=alpha,
            )

        # Car position (green triangle)
        all_trail_pts = [
            p for trail in self._lap_trails for p in trail
        ]
        if all_trail_pts:
            pos = all_trail_pts[-1]
            ax.plot(
                pos[0], pos[1], "^",
                color="#00ff00", markersize=10, zorder=10,
                markeredgecolor="white", markeredgewidth=0.5,
            )

        # Start/finish marker
        if self._lap_start_pos is not None:
            ax.plot(
                self._lap_start_pos[0], self._lap_start_pos[1], "s",
                color="#ffff00", markersize=8, zorder=9,
                markeredgecolor="white", markeredgewidth=0.5, label="Start",
            )

        # Stats text
        ax.text(
            0.02, 0.98,
            f"Lap: {self._current_lap}  |  "
            f"Outer: {len(self._outer_line)} pts  |  "
            f"Inner: {len(self._inner_line)} pts",
            transform=ax.transAxes, color="white", fontsize=9,
            verticalalignment="top", fontfamily="monospace",
        )

        ax.legend(
            loc="lower right", fontsize=8,
            facecolor="#2a2a4a", edgecolor="#444488",
            labelcolor="white",
        )

    def _write_video_frame(self) -> None:
        """Rasterize the current figure to a numpy array and write to video."""
        import cv2

        buf = self._fig.canvas.buffer_rgba()
        rgba = np.asarray(buf)                       # (H, W, 4) uint8
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)  # VideoWriter needs BGR
        self._video_writer.write(bgr)
        self._frame_count += 1

    def teardown(self) -> None:
        # Finalise video
        if self._video_writer is not None:
            self._video_writer.release()
            duration = (
                self._frame_count / self._video_fps
                if self._video_fps > 0
                else 0
            )
            self.logger.info(
                f"Video saved → {self._video_path}  "
                f"({self._frame_count} frames, {duration:.1f}s)"
            )

        # Save final snapshot
        if self._fig is not None:
            path = os.path.join(self._save_dir, "map_final.png")
            self._fig.savefig(
                path, dpi=150, facecolor=self._fig.get_facecolor(),
                bbox_inches="tight",
            )
            self.logger.info(f"Final map saved to {path}")
            self._plt.close(self._fig)

        self.logger.info(
            f"MapPlotWorker shutting down — "
            f"{len(self._traj_pts)} poses, "
            f"{self._current_lap} laps"
        )
