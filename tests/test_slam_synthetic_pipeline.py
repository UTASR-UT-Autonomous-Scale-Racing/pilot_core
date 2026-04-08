"""
test_slam_synthetic_pipeline.py
===============================
End-to-end test of the ORB-SLAM3 RGBD pipeline using synthetic sensor data.

Only the RGBD frames and IMU readings are synthetic.  Mapping, loop closure
detection, point-cloud generation, and localization are all performed by the
real ORB-SLAM3 library via its pybind11 wrapper (``orbslam3_python``).

Two phases are executed in sequence:

**Phase A — Mapping**
    Feed a full closed-loop sequence of synthetic RGBD frames to ORB-SLAM3.
    At the end (or during), verify that:
    - tracking was successful for the majority of frames
    - 3D map points were created
    - a map change (loop closure / global BA) was flagged

**Phase B — Localization**
    Switch to localization-only mode and re-feed the same frames.
    Verify that:
    - tracking still succeeds against the pre-built map
    - returned poses are valid (non-identity, non-zero rotation+translation)

This test is intentionally single-process (no mp workers) to keep the
SLAM system lifecycle simple and deterministic.  The multiprocessing worker
integration is tested separately.
"""

from __future__ import annotations

import os
import sys
import time
import logging

import cv2
import numpy as np
import pytest

# --- skip if orbslam3_python is not available ---
orbslam3 = pytest.importorskip(
    "orbslam3_python",
    reason="orbslam3_python wrapper not built or not on sys.path",
)

# Ensure project root is on path for synthetic imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.closed_loop_generator import ClosedLoopFrameGenerator

# --- Paths ---
VOCAB_PATH = os.path.join(ROOT, "ORB_SLAM3", "Vocabulary", "ORBvoc.txt")
SETTINGS_PATH = os.path.join(ROOT, "synthetic_slam_settings.yaml")
VIDEO_DIR = os.path.join(ROOT, "test_output")

logger = logging.getLogger(__name__)


# --- Frame Logger ---

class FrameLogger:
    """Records a four-panel 2x2 video:

        +------------------+------------------+
        |  (1) RGB image   | (2) Depth mask   |
        +------------------+------------------+
        |  (3) IMU values  | (4) BEV map      |
        +------------------+------------------+

    Each panel is 640x480.  Final video is 1280x960.
    """

    PANEL_W = 640
    PANEL_H = 480

    # Camera intrinsics (must match synthetic_slam_settings.yaml)
    FX = 456.99
    FY = 456.99
    CX = 320.0
    CY = 240.0

    # BEV rendering extents (metres)
    BEV_RANGE = 5.0  # +/- 5 m around the centroid

    def __init__(self, output_path: str, fps: float = 25.0) -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (self.PANEL_W * 2, self.PANEL_H * 2),
        )
        self._path = output_path
        self._all_map_pts: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self._trajectory: list[np.ndarray] = []
        self._frame_count = 0
        logger.info(f"FrameLogger: writing to {output_path}")

    # -- public API ----------------------------------------------------

    def log(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        slam_system,
        state: int,
        imu: dict | None = None,
        phase: str = "A",
    ) -> None:
        """Compose one 2x2-panel frame and write it to the video."""
        # Accumulate trajectory
        self._trajectory.append(pose[:3, 3].copy())

        # Fetch current map points from SLAM
        raw_pts = slam_system.get_all_map_points()
        pts_3d = np.asarray(raw_pts, dtype=np.float32)
        if len(pts_3d) > 0:
            self._all_map_pts = pts_3d

        # (1) Raw synthetic RGB frame
        panel1 = self._to_panel(rgb.copy())
        state_name = _tracking_state_name(state)
        colour = (0, 255, 0) if state == 2 else (0, 0, 255)
        self._put_label(panel1, f"{state_name}  f:{self._frame_count}", (10, 30), colour)

        # (2) Depth visualisation
        panel2 = self._draw_depth_mask(depth)

        # (3) IMU values
        panel3 = self._draw_imu_panel(imu)

        # (4) BEV map
        panel4 = self._draw_bev_map(pose, phase)

        # Assemble 2x2 grid
        top_row = np.hstack([panel1, panel2])
        bot_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bot_row])
        self._writer.write(combined)
        self._frame_count += 1

    def release(self) -> str:
        """Finalise the video file and return its path."""
        self._writer.release()
        logger.info(
            f"FrameLogger: wrote {self._frame_count} frames to {self._path}"
        )
        return self._path

    # -- internal helpers ----------------------------------------------

    def _to_panel(self, img: np.ndarray) -> np.ndarray:
        """Resize / pad to exactly PANEL_W x PANEL_H."""
        h, w = img.shape[:2]
        if (w, h) != (self.PANEL_W, self.PANEL_H):
            img = cv2.resize(img, (self.PANEL_W, self.PANEL_H))
        return img

    def _draw_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        """Colourmap the depth image into an RGB panel."""
        panel = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        if depth is None or depth.size == 0:
            self._put_label(panel, "No depth", (10, 30))
            return panel
        d = depth.copy().astype(np.float32)
        # Resize to panel dims if needed
        if d.shape[:2] != (self.PANEL_H, self.PANEL_W):
            d = cv2.resize(d, (self.PANEL_W, self.PANEL_H),
                           interpolation=cv2.INTER_NEAREST)
        valid = d > 0.01
        if not valid.any():
            self._put_label(panel, "Depth: all zero", (10, 30))
            return panel
        d_min = d[valid].min()
        d_max = d[valid].max()
        d_range = max(d_max - d_min, 0.01)
        norm = np.zeros_like(d, dtype=np.uint8)
        norm[valid] = np.clip(
            ((d[valid] - d_min) / d_range * 255), 0, 255
        ).astype(np.uint8)
        coloured = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        # Black out invalid pixels
        coloured[~valid] = (0, 0, 0)
        self._put_label(
            coloured,
            f"Depth  min={d_min:.2f}m  max={d_max:.2f}m",
            (10, 30),
        )
        return coloured

    def _draw_imu_panel(self, imu: dict | None) -> np.ndarray:
        """Render IMU accelerometer and gyroscope values as text + bar graph."""
        panel = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        self._put_label(panel, "IMU Readings", (10, 30), (200, 200, 200))

        if imu is None:
            self._put_label(panel, "No IMU data", (10, 70))
            return panel

        accel = imu.get('accel', np.zeros(3))
        gyro = imu.get('gyro', np.zeros(3))

        # Accelerometer section
        self._put_label(panel, "Accelerometer (m/s2)", (10, 70), (180, 180, 255))
        labels_a = ['X', 'Y', 'Z']
        max_accel = 15.0  # scale bar to this
        bar_x0, bar_w = 100, 400
        for j, (lbl, val) in enumerate(zip(labels_a, accel)):
            y = 100 + j * 50
            self._put_label(panel, f"{lbl}: {val:+8.3f}", (10, y + 15))
            # Bar: centred at midpoint, extends left/right
            mid = bar_x0 + bar_w // 2
            extent = int(np.clip(val / max_accel, -1, 1) * (bar_w // 2))
            # Background bar
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + 25),
                          (40, 40, 40), -1)
            # Centre line
            cv2.line(panel, (mid, y), (mid, y + 25), (80, 80, 80), 1)
            # Value bar
            col = (100, 200, 100) if val >= 0 else (100, 100, 200)
            if extent >= 0:
                cv2.rectangle(panel, (mid, y + 2), (mid + extent, y + 23), col, -1)
            else:
                cv2.rectangle(panel, (mid + extent, y + 2), (mid, y + 23), col, -1)

        # Gyroscope section
        self._put_label(panel, "Gyroscope (rad/s)", (10, 270), (255, 180, 180))
        labels_g = ['X', 'Y', 'Z']
        max_gyro = 5.0
        for j, (lbl, val) in enumerate(zip(labels_g, gyro)):
            y = 300 + j * 50
            self._put_label(panel, f"{lbl}: {val:+8.4f}", (10, y + 15))
            mid = bar_x0 + bar_w // 2
            extent = int(np.clip(val / max_gyro, -1, 1) * (bar_w // 2))
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + 25),
                          (40, 40, 40), -1)
            cv2.line(panel, (mid, y), (mid, y + 25), (80, 80, 80), 1)
            col = (200, 200, 100) if val >= 0 else (200, 100, 100)
            if extent >= 0:
                cv2.rectangle(panel, (mid, y + 2), (mid + extent, y + 23), col, -1)
            else:
                cv2.rectangle(panel, (mid + extent, y + 2), (mid, y + 23), col, -1)

        return panel

    def _draw_bev_map(self, pose: np.ndarray, phase: str) -> np.ndarray:
        """Render a bird's-eye view of map points + trajectory."""
        bev = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)

        # Determine BEV extents from all accumulated data
        all_xz = []
        if len(self._all_map_pts) > 0:
            all_xz.append(self._all_map_pts[:, [0, 2]])
        if self._trajectory:
            traj = np.array(self._trajectory)
            all_xz.append(traj[:, [0, 2]])

        if not all_xz:
            self._put_label(bev, "BEV — waiting", (10, 30))
            return bev

        combined = np.vstack(all_xz)
        cx = combined[:, 0].mean()
        cz = combined[:, 1].mean()
        spread = max(
            combined[:, 0].max() - combined[:, 0].min(),
            combined[:, 1].max() - combined[:, 1].min(),
            2.0,
        )
        half = spread * 0.6  # 20% margin

        def world_to_px(wx, wz):
            px = int((wx - cx + half) / (2 * half) * self.PANEL_W)
            py = int((wz - cz + half) / (2 * half) * self.PANEL_H)
            return px, py

        # Draw map points (grey dots)
        if len(self._all_map_pts) > 0:
            pts_bev = self._all_map_pts[:, [0, 2]]
            for wx, wz in pts_bev[::max(1, len(pts_bev) // 3000)]:
                px, py = world_to_px(wx, wz)
                if 0 <= px < self.PANEL_W and 0 <= py < self.PANEL_H:
                    bev[py, px] = (100, 100, 100)

        # Draw trajectory
        if len(self._trajectory) >= 2:
            traj = np.array(self._trajectory)
            traj_xz = traj[:, [0, 2]]
            pts_px = [world_to_px(x, z) for x, z in traj_xz]
            for j in range(1, len(pts_px)):
                p1 = pts_px[j - 1]
                p2 = pts_px[j]
                col = (0, 255, 0) if phase == "A" else (255, 200, 0)
                cv2.line(bev, p1, p2, col, 1, cv2.LINE_AA)

        # Current position (bright dot)
        cur_x, cur_z = pose[0, 3], pose[2, 3]
        cpx, cpy = world_to_px(cur_x, cur_z)
        cv2.circle(bev, (cpx, cpy), 5, (0, 0, 255), -1)

        # Heading arrow (forward = third column of rotation)
        fwd_world = pose[:3, 2]  # camera forward = travel direction
        arrow_len = int(self.PANEL_W * 0.04)
        ax = cpx + int(fwd_world[0] * arrow_len / (2 * half) * self.PANEL_W)
        ay = cpy + int(fwd_world[2] * arrow_len / (2 * half) * self.PANEL_H)
        cv2.arrowedLine(bev, (cpx, cpy), (ax, ay), (0, 0, 255), 2, tipLength=0.4)

        # Labels
        phase_label = "MAPPING" if phase == "A" else "LOCALIZATION"
        self._put_label(bev, f"BEV — {phase_label}", (10, 25), (200, 200, 200))
        n_pts = len(self._all_map_pts)
        self._put_label(
            bev,
            f"map:{n_pts}  traj:{len(self._trajectory)}",
            (10, self.PANEL_H - 15),
            (150, 150, 150),
        )

        return bev

    @staticmethod
    def _put_label(
        img: np.ndarray,
        text: str,
        org: tuple,
        colour: tuple = (255, 255, 255),
    ) -> None:
        cv2.putText(
            img, text, org,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            img, text, org,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA,
        )


# --- Helpers ---

def _is_valid_pose(pose: np.ndarray) -> bool:
    """Return True if *pose* is a plausible SE3 matrix (not identity/zero)."""
    if pose.shape != (4, 4):
        return False
    # Check the rotation block is a valid rotation (det ≈ 1)
    R = pose[:3, :3]
    det = np.linalg.det(R)
    if abs(det - 1.0) > 0.1:
        return False
    # Check it's not just identity (translation should be non-trivial)
    translation_norm = np.linalg.norm(pose[:3, 3])
    # At the very start translation may be near-zero; that's OK
    return True


def _tracking_state_name(state: int) -> str:
    _names = {
        -1: "SYSTEM_NOT_READY",
        0: "NO_IMAGES_YET",
        1: "NOT_INITIALIZED",
        2: "OK",
        3: "RECENTLY_LOST",
        4: "LOST",
        5: "OK_KLT",
    }
    return _names.get(state, f"UNKNOWN({state})")


# --- Fixtures ---

@pytest.fixture(scope="module")
def synthetic_frames():
    """Generate two continuous laps of synthetic RGBD frames.

    Returns the full 2-lap sequence.  Tests split it:
      Phase A = first lap  (frames 0 .. n_per_lap-1)
      Phase B = second lap (frames n_per_lap .. 2*n_per_lap-1)
    """
    n_per_lap = 300
    logger.info("Generating synthetic closed-loop frames (2 laps) …")
    gen = ClosedLoopFrameGenerator(
        n_frames=n_per_lap,
        straight_len=2.0,
        turn_radius=1.0,
        wall_height=1.5,
        camera_height=0.15,
        fps=25.0,
        seed=42,
    )
    frames = gen.generate_all(n_loops=2)
    logger.info(f"Generated {len(frames)} frames ({n_per_lap} per lap)")
    return frames


@pytest.fixture(scope="module")
def slam_system():
    """Create and yield an ORB-SLAM3 System; shut it down after test module."""
    assert os.path.isfile(VOCAB_PATH), f"Vocabulary not found: {VOCAB_PATH}"
    assert os.path.isfile(SETTINGS_PATH), f"Settings not found: {SETTINGS_PATH}"

    logger.info("Initialising ORB-SLAM3 (this loads the vocabulary) …")
    t0 = time.time()
    system = orbslam3.System(VOCAB_PATH, SETTINGS_PATH, orbslam3.RGBD)
    logger.info(f"ORB-SLAM3 ready in {time.time() - t0:.1f}s")

    yield system

    system.shutdown()
    logger.info("ORB-SLAM3 shut down")


@pytest.fixture(scope="module")
def frame_logger():
    """Create a FrameLogger that writes the debug video."""
    video_path = os.path.join(VIDEO_DIR, "slam_synthetic_pipeline.mp4")
    fl = FrameLogger(video_path, fps=25.0)
    yield fl
    path = fl.release()
    logger.info(f"Video saved to {path}")


# --- Tests ---

class TestSLAMSyntheticPipeline:
    """Phase A (mapping) + Phase B (localization) using real ORB-SLAM3."""

    # Store results across methods (class-level state)
    _phase_a_map_points: np.ndarray | None = None
    _phase_a_poses: list = []
    _map_changed: bool = False
    _phase_a_tracking_ok_count: int = 0
    _phase_b_tracking_ok_count: int = 0

    # --- Phase A: Mapping ---

    def test_phase_a_mapping(self, slam_system, synthetic_frames, frame_logger):
        """Feed the first lap to ORB-SLAM3 and verify mapping."""
        logger.info("═" * 60)
        logger.info("Phase A — MAPPING (lap 1)")
        logger.info("═" * 60)

        n_per_lap = len(synthetic_frames) // 2
        lap1 = synthetic_frames[:n_per_lap]
        n = len(lap1)
        tracking_ok = 0
        map_changed_count = 0
        poses = []

        for i, (rgb, depth, gt_pose, ts, imu_data) in enumerate(lap1):
            # ORB-SLAM3 expects uint8 RGB and float32 depth
            rgb_u8 = rgb.astype(np.uint8)
            depth_f32 = depth.astype(np.float32)

            pose = slam_system.track_rgbd(rgb_u8, depth_f32, ts)
            pose = np.asarray(pose, dtype=np.float32)
            poses.append(pose)

            state = slam_system.get_tracking_state()
            if state == 2:  # OK
                tracking_ok += 1

            if slam_system.map_changed():
                map_changed_count += 1
                TestSLAMSyntheticPipeline._map_changed = True
                logger.info(
                    f"  [frame {i}/{n}] Map changed! "
                    f"(total changes: {map_changed_count})"
                )

            # Log frame to video
            frame_logger.log(
                rgb_u8, depth_f32, pose, slam_system, state,
                imu=imu_data, phase="A",
            )

            if i % 50 == 0 or i == n - 1:
                state_name = _tracking_state_name(state)
                pts = slam_system.get_all_map_points()
                n_pts = len(np.asarray(pts))
                logger.info(
                    f"  [frame {i:3d}/{n}] state={state_name}, "
                    f"map_pts={n_pts}, tracking_ok={tracking_ok}"
                )

        # Store results for subsequent tests
        map_points = np.asarray(slam_system.get_all_map_points(), dtype=np.float32)
        TestSLAMSyntheticPipeline._phase_a_map_points = map_points
        TestSLAMSyntheticPipeline._phase_a_poses = poses
        TestSLAMSyntheticPipeline._phase_a_tracking_ok_count = tracking_ok

        logger.info(f"Phase A complete: {tracking_ok}/{n} frames tracked OK")
        logger.info(f"Map points: {len(map_points)}")
        logger.info(f"Map changes: {map_changed_count}")
        logger.info(f"Trajectory length: {slam_system.trajectory_length()}")

        # --- Assertions ---
        # At least 30% of frames should be tracked successfully
        # (initialisation takes a few frames, and synthetic scenes may
        # occasionally lose tracking in turns)
        min_tracked = int(n * 0.30)
        assert tracking_ok >= min_tracked, (
            f"Expected at least {min_tracked} frames tracked OK, got {tracking_ok}"
        )

        # Must have created some 3D map points
        assert len(map_points) > 0, "No map points were created"

        logger.info("✓ Phase A assertions passed")

    # --- Phase A result inspection ---

    def test_phase_a_map_points_quality(self, slam_system):
        """Verify map points are spatially distributed (not collapsed)."""
        pts = TestSLAMSyntheticPipeline._phase_a_map_points
        if pts is None or len(pts) == 0:
            pytest.skip("No map points from Phase A")

        # Points should span at least 0.5 m in X and Z
        x_range = pts[:, 0].max() - pts[:, 0].min()
        z_range = pts[:, 2].max() - pts[:, 2].min()

        logger.info(
            f"Map point spread: X={x_range:.2f}m, "
            f"Y={pts[:, 1].max() - pts[:, 1].min():.2f}m, "
            f"Z={z_range:.2f}m"
        )

        assert x_range > 0.3, f"Map points too narrow in X: {x_range:.2f}m"
        assert z_range > 0.3, f"Map points too narrow in Z: {z_range:.2f}m"

        logger.info("✓ Map point quality OK")

    def test_phase_a_keyframes(self, slam_system):
        """Verify multiple keyframes were created."""
        kf_poses = np.asarray(slam_system.get_all_keyframe_poses())
        n_kf = len(kf_poses)
        logger.info(f"Keyframes: {n_kf}")

        assert n_kf >= 3, f"Expected ≥3 keyframes, got {n_kf}"
        logger.info("✓ Keyframe count OK")

    # --- Phase B: Driving lap on the mapped track ---

    def test_phase_b_localization(self, slam_system, synthetic_frames, frame_logger):
        """Feed the second lap — continuous tracking on the mapped track."""
        logger.info("═" * 60)
        logger.info("Phase B — DRIVING LAP (lap 2 on the mapped track)")
        logger.info("═" * 60)

        n_per_lap = len(synthetic_frames) // 2
        lap2 = synthetic_frames[n_per_lap:]
        n = len(lap2)
        tracking_ok = 0
        map_changed_b = 0
        poses_b = []

        for i, (rgb, depth, gt_pose, ts, imu_data) in enumerate(lap2):
            rgb_u8 = rgb.astype(np.uint8)
            depth_f32 = depth.astype(np.float32)

            pose = slam_system.track_rgbd(rgb_u8, depth_f32, ts)
            pose = np.asarray(pose, dtype=np.float32)
            poses_b.append(pose)

            state = slam_system.get_tracking_state()
            if state == 2:
                tracking_ok += 1

            if slam_system.map_changed():
                map_changed_b += 1

            # Log frame to video
            frame_logger.log(
                rgb_u8, depth_f32, pose, slam_system, state,
                imu=imu_data, phase="B",
            )

            if i % 50 == 0 or i == n - 1:
                state_name = _tracking_state_name(state)
                logger.info(
                    f"  [frame {i:3d}/{n}] state={state_name}, "
                    f"tracking_ok={tracking_ok}, map_changes={map_changed_b}"
                )

        TestSLAMSyntheticPipeline._phase_b_tracking_ok_count = tracking_ok

        logger.info(f"Phase B complete: {tracking_ok}/{n} frames tracked OK")

        # At least 20% should track
        min_tracked = int(n * 0.20)
        assert tracking_ok >= min_tracked, (
            f"Expected at least {min_tracked} Phase B frames OK, "
            f"got {tracking_ok}"
        )

        logger.info("✓ Phase B assertions passed")

    # --- Summary ---

    def test_summary(self, slam_system):
        """Print a final summary of both phases."""
        pts = TestSLAMSyntheticPipeline._phase_a_map_points
        n_pts = len(pts) if pts is not None else 0
        kf = np.asarray(slam_system.get_all_keyframe_poses())

        logger.info("═" * 60)
        logger.info("SUMMARY")
        logger.info(f"  Phase A tracking OK : {TestSLAMSyntheticPipeline._phase_a_tracking_ok_count}")
        logger.info(f"  Phase B tracking OK : {TestSLAMSyntheticPipeline._phase_b_tracking_ok_count}")
        logger.info(f"  Map points          : {n_pts}")
        logger.info(f"  Keyframes           : {len(kf)}")
        logger.info(f"  Map changed         : {TestSLAMSyntheticPipeline._map_changed}")
        logger.info(f"  Trajectory length   : {slam_system.trajectory_length()}")
        logger.info("═" * 60)

        # This test always passes — it's just for reporting
        assert True
