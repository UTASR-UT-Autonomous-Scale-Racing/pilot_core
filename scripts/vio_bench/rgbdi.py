"""
The `rgbdi` sequence format shared by the RGB-D-inertial benchmark (README.md here): one
directory per sequence, written from the sim or from a real-car recording, read by both
runners and the scorer.

    rgb/<t_ns>.png         8-bit RGB
    depth/<t_ns>.png       uint16 millimetres, pixel-aligned with rgb (0 = no depth)
    frames.csv             t_ns,rgb,depth                                   (time order)
    imu.csv                t_ns,ax,ay,az,gx,gy,gz    IMU frame, m/s^2 and rad/s, one
                           row per paired accel + gyro sample
    groundtruth.csv        t_ns,x,y,z,qx,qy,qz,qw    camera pose in the world (optional:
                           the real track has markers.csv instead)
    calib.yaml             width, height, fx, fy, cx, cy (undistorted pinhole),
                           T_imu_cam (4x4, p_imu = T p_cam), imu: {rate_hz, noise_gyro,
                           noise_acc, gyro_walk, acc_walk}, source, notes

All times are nanoseconds on one clock. Poses use the camera optical frame (x right,
y down, z forward).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


@dataclass
class Sequence:
    root: str
    t_ns: np.ndarray                 # (F,) int64 frame times
    rgb: list                        # F paths
    depth: list                      # F paths
    imu: np.ndarray                  # (M, 7) float64: t_ns, ax, ay, az, gx, gy, gz
    calib: dict
    gt_t_ns: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    gt_T: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 4)))   # T_world_cam

    @property
    def K(self) -> tuple:
        c = self.calib
        return c["fx"], c["fy"], c["cx"], c["cy"]

    @property
    def T_imu_cam(self) -> np.ndarray:
        return np.asarray(self.calib["T_imu_cam"], dtype=np.float64).reshape(4, 4)


def load(root: str) -> Sequence:
    with open(os.path.join(root, "calib.yaml")) as f:
        calib = yaml.safe_load(f)
    t, rgb, depth = [], [], []
    with open(os.path.join(root, "frames.csv")) as f:
        for r in csv.DictReader(f):
            t.append(int(r["t_ns"]))
            rgb.append(os.path.join(root, r["rgb"]))
            depth.append(os.path.join(root, r["depth"]))
    imu = np.loadtxt(os.path.join(root, "imu.csv"), delimiter=",", skiprows=1, ndmin=2)
    seq = Sequence(root, np.array(t, np.int64), rgb, depth, imu, calib)
    gt_path = os.path.join(root, "groundtruth.csv")
    if os.path.exists(gt_path):
        g = np.loadtxt(gt_path, delimiter=",", skiprows=1, ndmin=2)
        seq.gt_t_ns = g[:, 0].astype(np.int64)
        seq.gt_T = poses_from_rows(g[:, 1:])
    return seq


def poses_from_rows(rows: np.ndarray) -> np.ndarray:
    """(N, 7) x, y, z, qx, qy, qz, qw -> (N, 4, 4)."""
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(rows[:, 3:7]).as_matrix()
    T[:, :3, 3] = rows[:, :3]
    return T


def rows_from_poses(T: np.ndarray) -> np.ndarray:
    return np.hstack([T[:, :3, 3], Rotation.from_matrix(T[:, :3, :3]).as_quat()])


def write_poses(path: str, t_ns: np.ndarray, T: np.ndarray, header: str = "t_ns,x,y,z,qx,qy,qz,qw") -> None:
    rows = rows_from_poses(T) if len(T) else np.zeros((0, 7))
    with open(path, "w") as f:
        f.write(header + "\n")
        for t, r in zip(t_ns, rows):
            f.write(f"{int(t)}," + ",".join(f"{v:.9f}" for v in r) + "\n")


def read_poses(path: str) -> tuple:
    g = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if g.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, 4, 4))
    return g[:, 0].astype(np.int64), poses_from_rows(g[:, 1:8])


def write_imu(path: str, imu: np.ndarray) -> None:
    with open(path, "w") as f:
        f.write("t_ns,ax,ay,az,gx,gy,gz\n")
        for r in imu:
            f.write(f"{int(r[0])}," + ",".join(f"{v:.7f}" for v in r[1:]) + "\n")


def interpolate_poses(t_src: np.ndarray, T_src: np.ndarray, t_query: np.ndarray) -> tuple:
    """Poses at t_query (linear position, slerp rotation); mask of queries inside t_src."""
    from scipy.spatial.transform import Slerp

    inside = (t_query >= t_src[0]) & (t_query <= t_src[-1])
    out = np.tile(np.eye(4), (len(t_query), 1, 1))
    if inside.any():
        tq = t_query[inside].astype(np.float64)
        ts = t_src.astype(np.float64)
        out[inside, :3, :3] = Slerp(ts, Rotation.from_matrix(T_src[:, :3, :3]))(tq).as_matrix()
        for k in range(3):
            out[inside, k, 3] = np.interp(tq, ts, T_src[:, k, 3])
    return out, inside
