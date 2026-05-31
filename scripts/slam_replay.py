#!/usr/bin/env python3
"""Replay a TUM RGB-D sequence through SLAMWorker; print SLAM time per frame and ATE.

    python scripts/slam_replay.py <tum_sequence_dir> <associations.txt> <settings.yaml>
    ORB_SLAM3_CUDA=0 python scripts/slam_replay.py ...      # CPU paths of ORB-SLAM3
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
import uuid
from unittest.mock import MagicMock

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from shared_buffer import FrameBuffer, PoseBuffer  # noqa: E402
from workers.slam_worker import SLAMWorker  # noqa: E402


def ate_rmse(est: dict, gt_path: str) -> tuple[float, int]:
    """ATE RMSE (m) after SE(3) Umeyama alignment; est maps timestamp -> position."""
    gt = np.array([[float(v) for v in l.split()[:4]] for l in open(gt_path) if l.strip() and l[0] != "#"])
    ts = np.array(sorted(est))
    idx = np.clip(np.searchsorted(gt[:, 0], ts), 1, len(gt) - 1)
    idx = np.where(np.abs(gt[idx - 1, 0] - ts) < np.abs(gt[idx, 0] - ts), idx - 1, idx)
    keep = np.abs(gt[idx, 0] - ts) < 0.02
    model, data = gt[idx[keep], 1:4].T, np.array([est[t] for t in ts[keep]]).T
    mm, md = model.mean(1, keepdims=True), data.mean(1, keepdims=True)
    U, _, Vt = np.linalg.svd((model - mm) @ (data - md).T)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ S @ Vt
    err = np.linalg.norm(R @ (data - md) + mm - model, axis=0)
    return float(np.sqrt((err ** 2).mean())), int(keep.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sequence", help="TUM sequence folder (rgb/, depth/, groundtruth.txt)")
    ap.add_argument("associations", help="associations file: t_rgb rgb t_depth depth")
    ap.add_argument("settings", help="ORB-SLAM3 settings; DepthMapFactor is replaced by 1.0 (depth in metres)")
    ap.add_argument("--depth-scale", type=float, default=5000.0, help="raw depth units per metre (TUM: 5000)")
    ap.add_argument("--realtime", action="store_true", help="feed frames at the dataset rate")
    args = ap.parse_args()

    rows = [l.split() for l in open(args.associations) if l.strip() and l[0] != "#"]
    text = re.sub(r"DepthMapFactor:\s*[\d.]+", "DepthMapFactor: 1.0", open(args.settings).read())
    settings = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    settings.write(text)
    settings.close()

    first = cv2.imread(os.path.join(args.sequence, rows[0][1]))
    uid = uuid.uuid4().hex[:8]
    fb = FrameBuffer(slots=2, name=f"replay_f_{uid}", create=True,
                     max_payload=16 + first.shape[0] * first.shape[1] * 7)
    pb = PoseBuffer(slots=256, name=f"replay_p_{uid}", create=True)
    w = SLAMWorker(name="slam", input_buffer=fb, output_buffer=pb,
                   vocab_path=os.path.join(REPO_ROOT, "ORB_SLAM3/Vocabulary/ORBvoc.txt"),
                   settings_path=settings.name, level_world=False)
    w._logger = MagicMock()
    try:
        w.setup()
        t0 = time.monotonic() - float(rows[0][0])       # dataset time on the monotonic clock
        steps, est, seen = [], {}, 0
        start = time.monotonic()
        for r in rows:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(args.sequence, r[1])), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(args.sequence, r[3]), cv2.IMREAD_UNCHANGED).astype(np.float32)
            depth /= args.depth_scale
            if args.realtime:
                time.sleep(max(0.0, (float(r[0]) - float(rows[0][0])) - (time.monotonic() - start)))
            fb.write(rgb, depth, ts=t0 + float(r[0]))
            s = time.perf_counter()
            w.step()
            steps.append(1e3 * (time.perf_counter() - s))
            if pb.latest_seq() > seen:
                header, pose, _, _ = pb.read_latest()
                seen = header.seq_no
                est[header.produced_ts - t0] = pose[:3, 3].astype(np.float64)
        w.teardown()
        rmse, pairs = ate_rmse(est, os.path.join(args.sequence, "groundtruth.txt"))
        tracked = np.array(steps[w._warmup_frames:])
        print(f"frames {len(rows)}  poses {len(est)}  slam step median {np.median(tracked):.2f} ms  "
              f"mean {tracked.mean():.2f} ms  ate_rmse {rmse:.4f} m  pairs {pairs}")
    finally:
        fb.unlink()
        pb.unlink()
        os.unlink(settings.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
