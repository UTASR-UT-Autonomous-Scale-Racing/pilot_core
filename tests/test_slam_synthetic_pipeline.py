"""
Real ORB-SLAM3 (through the orbslam3_python pybind11 wrapper) on synthetic
RGBD + IMU data: two laps of a closed track from synthetic.closed_loop_generator.

Skipped unless orbslam3_python imports. Also needs ORB_SLAM3/Vocabulary/ORBvoc.txt
and synthetic_slam_settings.yaml.

The tests share one SLAM system and have to run in order. Lap 1 is the mapping
phase (A): >= 30% of frames tracked, map points spread in X and Z, at least 3
keyframes. Map changes (loop closure, global BA) are only logged. Lap 2 (B)
drives over the mapped track and needs >= 20% tracked.

Writes a 2x2 debug video (RGB, depth, IMU, BEV) to
test_output/slam_synthetic_pipeline.mp4. Kept single-process, no workers, so
the SLAM lifecycle stays simple; workers are covered elsewhere.
"""

from __future__ import annotations

import os
import sys
import time
import logging

import cv2
import numpy as np
import pytest

orbslam3 = pytest.importorskip(
    "orbslam3_python",
    reason="orbslam3_python wrapper not built or not on sys.path",
)

# project root on sys.path for the synthetic imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from synthetic.closed_loop_generator import ClosedLoopFrameGenerator

VOCAB_PATH = os.path.join(ROOT, "ORB_SLAM3", "Vocabulary", "ORBvoc.txt")
SETTINGS_PATH = os.path.join(ROOT, "synthetic_slam_settings.yaml")
VIDEO_DIR = os.path.join(ROOT, "test_output")

logger = logging.getLogger(__name__)


class FrameLogger:
    """2x2 debug video: RGB | depth on top, IMU | BEV map below.
    Panels are 640x480 (frames are resized), so the video is 1280x960."""

    PANEL_W = 640
    PANEL_H = 480

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
        self._trajectory.append(pose[:3, 3].copy())

        raw_pts = slam_system.get_all_map_points()
        pts_3d = np.asarray(raw_pts, dtype=np.float32)
        if len(pts_3d) > 0:
            self._all_map_pts = pts_3d

        panel1 = self._to_panel(rgb.copy())
        state_name = _tracking_state_name(state)
        colour = (0, 255, 0) if state == 2 else (0, 0, 255)
        self._put_label(panel1, f"{state_name}  f:{self._frame_count}", (10, 30), colour)

        panel2 = self._draw_depth_mask(depth)

        panel3 = self._draw_imu_panel(imu)

        panel4 = self._draw_bev_map(pose, phase)

        top_row = np.hstack([panel1, panel2])
        bot_row = np.hstack([panel3, panel4])
        combined = np.vstack([top_row, bot_row])
        self._writer.write(combined)
        self._frame_count += 1

    def release(self) -> str:
        self._writer.release()
        logger.info(
            f"FrameLogger: wrote {self._frame_count} frames to {self._path}"
        )
        return self._path

    def _to_panel(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if (w, h) != (self.PANEL_W, self.PANEL_H):
            img = cv2.resize(img, (self.PANEL_W, self.PANEL_H))
        return img

    def _draw_depth_mask(self, depth: np.ndarray) -> np.ndarray:
        panel = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        if depth is None or depth.size == 0:
            self._put_label(panel, "No depth", (10, 30))
            return panel
        d = depth.copy().astype(np.float32)
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
        coloured[~valid] = (0, 0, 0)
        self._put_label(
            coloured,
            f"Depth  min={d_min:.2f}m  max={d_max:.2f}m",
            (10, 30),
        )
        return coloured

    def _draw_imu_panel(self, imu: dict | None) -> np.ndarray:
        """Accel and gyro values as text plus a bar each"""
        panel = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)
        self._put_label(panel, "IMU Readings", (10, 30), (200, 200, 200))

        if imu is None:
            self._put_label(panel, "No IMU data", (10, 70))
            return panel

        accel = imu.get('accel', np.zeros(3))
        gyro = imu.get('gyro', np.zeros(3))

        # accel
        self._put_label(panel, "Accelerometer (m/s2)", (10, 70), (180, 180, 255))
        labels_a = ['X', 'Y', 'Z']
        max_accel = 15.0
        bar_x0, bar_w = 100, 400
        for j, (lbl, val) in enumerate(zip(labels_a, accel)):
            y = 100 + j * 50
            self._put_label(panel, f"{lbl}: {val:+8.3f}", (10, y + 15))
            # bar is centred, grows left for negative values
            mid = bar_x0 + bar_w // 2
            extent = int(np.clip(val / max_accel, -1, 1) * (bar_w // 2))
            cv2.rectangle(panel, (bar_x0, y), (bar_x0 + bar_w, y + 25),
                          (40, 40, 40), -1)
            cv2.line(panel, (mid, y), (mid, y + 25), (80, 80, 80), 1)
            col = (100, 200, 100) if val >= 0 else (100, 100, 200)
            if extent >= 0:
                cv2.rectangle(panel, (mid, y + 2), (mid + extent, y + 23), col, -1)
            else:
                cv2.rectangle(panel, (mid + extent, y + 2), (mid, y + 23), col, -1)

        # gyro
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
        bev = np.zeros((self.PANEL_H, self.PANEL_W, 3), dtype=np.uint8)

        # fit the view to everything seen so far
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
        half = spread * 0.6

        def world_to_px(wx, wz):
            px = int((wx - cx + half) / (2 * half) * self.PANEL_W)
            py = int((wz - cz + half) / (2 * half) * self.PANEL_H)
            return px, py

        # map points, subsampled to ~3000
        if len(self._all_map_pts) > 0:
            pts_bev = self._all_map_pts[:, [0, 2]]
            for wx, wz in pts_bev[::max(1, len(pts_bev) // 3000)]:
                px, py = world_to_px(wx, wz)
                if 0 <= px < self.PANEL_W and 0 <= py < self.PANEL_H:
                    bev[py, px] = (100, 100, 100)

        if len(self._trajectory) >= 2:
            traj = np.array(self._trajectory)
            traj_xz = traj[:, [0, 2]]
            pts_px = [world_to_px(x, z) for x, z in traj_xz]
            for j in range(1, len(pts_px)):
                p1 = pts_px[j - 1]
                p2 = pts_px[j]
                col = (0, 255, 0) if phase == "A" else (255, 200, 0)
                cv2.line(bev, p1, p2, col, 1, cv2.LINE_AA)

        cur_x, cur_z = pose[0, 3], pose[2, 3]
        cpx, cpy = world_to_px(cur_x, cur_z)
        cv2.circle(bev, (cpx, cpy), 5, (0, 0, 255), -1)

        fwd_world = pose[:3, 2]  # camera forward (3rd column) = travel direction
        arrow_len = int(self.PANEL_W * 0.04)
        ax = cpx + int(fwd_world[0] * arrow_len / (2 * half) * self.PANEL_W)
        ay = cpy + int(fwd_world[2] * arrow_len / (2 * half) * self.PANEL_H)
        cv2.arrowedLine(bev, (cpx, cpy), (ax, ay), (0, 0, 255), 2, tipLength=0.4)

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


@pytest.fixture(scope="module")
def synthetic_frames():
    """Two continuous laps. Phase A uses the first half, phase B the second."""
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
    video_path = os.path.join(VIDEO_DIR, "slam_synthetic_pipeline.mp4")
    fl = FrameLogger(video_path, fps=25.0)
    yield fl
    path = fl.release()
    logger.info(f"Video saved to {path}")


class TestSLAMSyntheticPipeline:
    # phase A results, kept on the class so later tests can see them
    _phase_a_map_points: np.ndarray | None = None
    _phase_a_poses: list = []
    _map_changed: bool = False
    _phase_a_tracking_ok_count: int = 0
    _phase_b_tracking_ok_count: int = 0

    # phase A: mapping

    def test_phase_a_mapping(self, slam_system, synthetic_frames, frame_logger):
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
            # ORB-SLAM3 wants uint8 RGB, float32 depth
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

        map_points = np.asarray(slam_system.get_all_map_points(), dtype=np.float32)
        TestSLAMSyntheticPipeline._phase_a_map_points = map_points
        TestSLAMSyntheticPipeline._phase_a_poses = poses
        TestSLAMSyntheticPipeline._phase_a_tracking_ok_count = tracking_ok

        logger.info(f"Phase A complete: {tracking_ok}/{n} frames tracked OK")
        logger.info(f"Map points: {len(map_points)}")
        logger.info(f"Map changes: {map_changed_count}")

        # Only 30%: init eats a few frames and the synthetic scene can lose
        # tracking in the turns
        min_tracked = int(n * 0.30)
        assert tracking_ok >= min_tracked, (
            f"Expected at least {min_tracked} frames tracked OK, got {tracking_ok}"
        )

        assert len(map_points) > 0, "No map points were created"

        logger.info("✓ Phase A assertions passed")

    def test_phase_a_map_points_quality(self, slam_system):
        """Map points should be spread out, not collapsed to a blob"""
        pts = TestSLAMSyntheticPipeline._phase_a_map_points
        if pts is None or len(pts) == 0:
            pytest.skip("No map points from Phase A")

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
        kf_poses = np.asarray(slam_system.get_all_keyframe_poses())
        n_kf = len(kf_poses)
        logger.info(f"Keyframes: {n_kf}")

        assert n_kf >= 3, f"Expected ≥3 keyframes, got {n_kf}"
        map_id, init_kf, n_kf_map = slam_system.map_info()
        assert n_kf_map >= 3           # current map (the local mapper may still be culling)
        logger.info("✓ Keyframe count OK")

    # phase B: second lap over the mapped track

    def test_phase_b_localization(self, slam_system, synthetic_frames, frame_logger):
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

        min_tracked = int(n * 0.20)
        assert tracking_ok >= min_tracked, (
            f"Expected at least {min_tracked} Phase B frames OK, "
            f"got {tracking_ok}"
        )

        logger.info("✓ Phase B assertions passed")

    def test_summary(self, slam_system):
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
        logger.info(f"  Map info            : {slam_system.map_info()}")
        logger.info("═" * 60)

        # just reporting, always passes
        assert True
