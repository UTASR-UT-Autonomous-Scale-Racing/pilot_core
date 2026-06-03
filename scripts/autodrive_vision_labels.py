#!/usr/bin/env python3
"""Ground-truth drivable-track masks for AutoDRIVE sim recordings, to train VisionWorker's
network (scripts/train_vision.py) and to score its output.

A pixel is track when the LiDAR depth adapter (workers/lidar_depth.py, the frame's own
scan) says its ray meets the floor before any wall, and that floor point, placed with
the sim's ground-truth pose, lies between the mapped track boundaries. The car body
(sim_camera/car_mask.png) is ignored (mask value 255 in the dataset, left out of IoU).
Recordings must be from runs with sim_depth on, so frames are what VisionWorker sees
live (upscaled, car painted out, contrast-equalised).

    python scripts/autodrive_vision_labels.py dataset <rec dir> [<rec dir> ...] --map maps/autodrive_practice \\
        --out datasets/sim_track [--stride 2]
    python scripts/autodrive_vision_labels.py score <rec dir> --map maps/autodrive_practice   # recorded seg vs GT
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

IGNORE = 255


class Labeller:
    def __init__(self, map_dir: str, camera: str = "sim_camera/camera.yaml", scale: float = 2.0) -> None:
        from matplotlib.path import Path

        from control.types import resolve_path
        from workers.lidar_depth import CameraModel, LidarDepth

        self.cam = CameraModel.from_yaml(camera, scale=scale)
        self.ld = LidarDepth(self.cam)
        d = resolve_path(map_dir)
        self.outer = Path(np.load(os.path.join(d, "outer_line.npy"))[:, :2])
        self.inner = Path(np.load(os.path.join(d, "inner_line.npy"))[:, :2])

    def mask(self, ranges: np.ndarray, Twv: np.ndarray) -> np.ndarray:
        """uint8 (H, W): 1 track, 0 not, IGNORE car body."""
        cam, ld = self.cam, self.ld
        D = ld(ranges)
        z = cam.cam_height + D * ld.d_up
        floor = (D > 0) & (z < 0.01)
        out = np.zeros(D.shape, np.uint8)
        v, u = np.nonzero(floor)
        if len(v):
            t = D[v, u]
            pv = np.stack([cam.cam_x + t * ld.d_fwd[v, u], t * ld.d_left[v, u]], 1)   # vehicle x, y
            pw = pv @ Twv[:2, :2].T + Twv[:2, 3]
            inside = self.outer.contains_points(pw) != self.inner.contains_points(pw)
            out[v[inside], u[inside]] = 1
        if ld.mask is not None:
            out[ld.mask] = IGNORE
        return out


def _paired(rec: str, with_seg: bool = False):
    """(frame record, lidar ranges, pose 4x4, seg record or None) per recorded frame."""
    from read_recording import list_files, load_buffer

    F = load_buffer(rec, "frame")
    L, P = load_buffer(rec, "lidar"), load_buffer(rec, "pose")
    S = load_buffer(rec, "segmentation") if with_seg and list_files(rec, "segmentation") else []
    lt = np.array([x.produced_ts for x in L])
    pt = np.array([x.produced_ts for x in P])
    st = np.array([x.produced_ts for x in S]) if S else None
    for f in F:
        i, j = int(np.argmin(abs(lt - f.produced_ts))), int(np.argmin(abs(pt - f.produced_ts)))
        if abs(lt[i] - f.produced_ts) > 0.02 or abs(pt[j] - f.produced_ts) > 0.02:
            continue
        s = None
        if st is not None:
            k = int(np.argmin(abs(st - f.produced_ts)))
            s = S[k] if abs(st[k] - f.produced_ts) < 0.2 else None
        yield f, L[i].value, P[j].value[0], s


def build_dataset(recs: list[str], map_dir: str, out: str, stride: int, camera: str) -> int:
    import cv2

    lab = Labeller(map_dir, camera)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "masks"), exist_ok=True)
    n = 0
    for r, rec in enumerate(recs):
        for k, (f, ranges, Twv, _) in enumerate(_paired(rec)):
            if k % stride:
                continue
            rgb, m = f.value[0], lab.mask(ranges, Twv)
            if rgb.shape[:2] != m.shape:
                raise SystemExit(f"{rec}: frames are {rgb.shape[1]}x{rgb.shape[0]}, not the sim_depth size "
                                 f"{lab.cam.size}: record with sim_depth on")
            name = f"r{r:02d}_{k:05d}"
            cv2.imwrite(os.path.join(out, "images", name + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            # train_vision reads non-zero as track: the car body goes in as 0 (not track)
            cv2.imwrite(os.path.join(out, "masks", name + ".png"), (m == 1).astype(np.uint8) * 255)
            n += 1
    with open(os.path.join(out, "source.json"), "w") as fp:
        json.dump({"recordings": recs, "map": map_dir, "camera": camera, "stride": stride, "images": n}, fp, indent=1)
    return n


def score(rec: str, map_dir: str, camera: str) -> dict:
    lab = Labeller(map_dir, camera)
    tp = fp = fn = 0
    ious, n = [], 0
    for f, ranges, Twv, s in _paired(rec, with_seg=True):
        if s is None:
            continue
        gt = lab.mask(ranges, Twv)
        pred = np.asarray(s.value[0] if isinstance(s.value, tuple) else s.value) > 0
        ok = gt != IGNORE
        g = gt == 1
        a, b, c = int((pred & g & ok).sum()), int((pred & ~g & ok).sum()), int((~pred & g & ok).sum())
        tp, fp, fn = tp + a, fp + b, fn + c
        ious.append(a / max(a + b + c, 1))
        n += 1
    return {"recording": rec, "frames": n, "iou": tp / max(tp + fp + fn, 1),
            "iou_per_frame_median": float(np.median(ious)) if ious else float("nan"),
            "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("recordings", nargs="+")
    d.add_argument("--map", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--stride", type=int, default=2)
    d.add_argument("--camera", default="sim_camera/camera.yaml")
    s = sub.add_parser("score")
    s.add_argument("recording")
    s.add_argument("--map", required=True)
    s.add_argument("--camera", default="sim_camera/camera.yaml")
    a = ap.parse_args()
    if a.cmd == "dataset":
        print(f"{build_dataset(a.recordings, a.map, a.out, a.stride, a.camera)} images -> {a.out}")
    else:
        print(json.dumps(score(a.recording, a.map, a.camera), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
