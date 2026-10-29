#!/usr/bin/env python3
"""Accuracy on the real track from AprilTags (README.md here): no motion capture needed.

    python scripts/vio_bench/markers.py <seq dir> <result dir> [...] --size 0.16 \
        [--survey markers.yaml] [--closed] --out <dir>

Tags (tag36h11 by default) are found in every RGB frame and located in the camera frame
from their corners and printed size (IPPE). With a result's pose of that frame, each
sighting places the tag in that result's world. Scored per result:

  repeatability  the same tag placed on different passes (a pass = sightings with no gap
                 over --pass-gap s, i.e. one lap): RMS / max distance of each pass's
                 placement from the tag's mean, and the distance from the first pass by
                 pass number (drift growth over laps). Needs no survey.
  absolute       with --survey (tag centres measured on the track, >= 3 tags not in a
                 line): one rigid alignment of the tags' mean placements onto the
                 survey, then every sighting's distance from its surveyed position.
  closure        with --closed (the car was stopped where it started): distance between
                 the first and last tracked poses.

Only sightings within --max-dist m and at least --min-side px across are used: a 16 cm
tag at fx ~400 px is placed to ~0.6 cm at 1 m, ~1.5 cm at 1.5 m, ~5 cm at 2.5 m and
~13 cm at 3.5 m (rendered test images), so use big tags. Detections are cached in <seq>/tags_<family>_<size>.csv.

markers.yaml:
    family: 36h11
    size_m: 0.16            # black square edge, m
    tags:                   # optional survey: tag centre in the track frame, m
      0: [0.0, 0.0, 0.25]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rgbdi  # noqa: E402
from score import umeyama_se3  # noqa: E402

FAMILIES = {"36h11": cv2.aruco.DICT_APRILTAG_36h11, "25h9": cv2.aruco.DICT_APRILTAG_25h9,
            "16h5": cv2.aruco.DICT_APRILTAG_16h5}


def tag_object_points(size: float) -> np.ndarray:
    """Corners in the tag frame, in the order the detector returns them (TL, TR, BR, BL)."""
    h = size / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def detect_frame(gray: np.ndarray, detector, K: np.ndarray, size: float) -> list:
    """[(id, p_cam (3,), side_px, reproj_px)] for the tags in one undistorted image."""
    corners, ids, _ = detector.detectMarkers(gray)
    out = []
    if ids is None:
        return out
    obj = tag_object_points(size)
    for c, i in zip(corners, ids.ravel()):
        img = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
        reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img) ** 2, axis=1))))
        side = float(np.mean(np.linalg.norm(img - np.roll(img, -1, axis=0), axis=1)))
        out.append((int(i), tvec.ravel(), side, reproj))
    return out


def detect(seq: rgbdi.Sequence, family: str, size: float) -> np.ndarray:
    """(N, 7) t_ns, id, x, y, z (camera frame), side_px, reproj_px for every sighting; cached."""
    cache = os.path.join(seq.root, f"tags_{family}_{size:g}.csv")
    if os.path.exists(cache):
        return np.loadtxt(cache, delimiter=",", skiprows=1, ndmin=2).reshape(-1, 7)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(FAMILIES[family]), params)
    fx, fy, cx, cy = seq.K
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    rows = []
    for t, path in zip(seq.t_ns, seq.rgb):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for i, p, side, reproj in detect_frame(gray, detector, K, size):
            rows.append([t, i, *p, side, reproj])
    det = np.array(rows, dtype=np.float64).reshape(-1, 7)
    with open(cache, "w") as f:
        f.write("t_ns,id,x,y,z,side_px,reproj_px\n")
        for r in det:
            f.write(f"{int(r[0])},{int(r[1])}," + ",".join(f"{v:.6f}" for v in r[2:]) + "\n")
    return det


def passes(t_s: np.ndarray, gap: float) -> np.ndarray:
    """Pass number of each (time-sorted) sighting: a new pass after a gap > gap s."""
    return np.concatenate([[0], np.cumsum(np.diff(t_s) > gap)])


def score_result(det: np.ndarray, t_pose: np.ndarray, T_pose: np.ndarray, survey: dict | None,
                 closed: bool, pass_gap: float) -> dict:
    """Marker metrics for one trajectory (poses T_world_cam at t_pose, ns)."""
    pose_at = {int(t): T for t, T in zip(t_pose, T_pose)}
    seen = [(r, pose_at[int(r[0])]) for r in det if int(r[0]) in pose_at]
    out: dict = {"sightings": len(det), "sightings_tracked": len(seen)}
    if closed and len(t_pose) >= 2:
        out["closure_m"] = float(np.linalg.norm(T_pose[-1][:3, 3] - T_pose[0][:3, 3]))
    if not seen:
        return out

    by_tag: dict = {}
    for r, T in seen:
        p_w = (T @ np.append(r[2:5], 1.0))[:3]
        by_tag.setdefault(int(r[1]), []).append((r[0] * 1e-9, p_w))
    rep_dev, growth, tag_means = [], {}, {}
    for tag, obs in by_tag.items():
        obs.sort(key=lambda o: o[0])
        t = np.array([o[0] for o in obs])
        P = np.array([o[1] for o in obs])
        tag_means[tag] = P.mean(axis=0)
        k = passes(t, pass_gap)
        med = np.array([np.median(P[k == j], axis=0) for j in range(k.max() + 1)])
        if len(med) >= 2:
            rep_dev.extend(np.linalg.norm(med - med.mean(axis=0), axis=1))
            for j in range(1, len(med)):
                growth.setdefault(j, []).append(float(np.linalg.norm(med[j] - med[0])))
    out["tags_seen"] = len(by_tag)
    if rep_dev:
        rep = np.array(rep_dev)
        out["repeatability_m"] = {"rms": float(np.sqrt(np.mean(rep ** 2))), "max": float(rep.max()),
                                  "tags": sum(1 for o in by_tag.values() if len(o) > 1)}
        out["drift_from_first_pass_m"] = {int(j): float(np.mean(v)) for j, v in sorted(growth.items())}

    if survey:
        common = sorted(set(tag_means) & set(survey))
        if len(common) >= 3:
            A = umeyama_se3(np.array([tag_means[i] for i in common]), np.array([survey[i] for i in common]))
            err = []
            for tag in common:
                for _, p in by_tag[tag]:
                    err.append(np.linalg.norm((A @ np.append(p, 1.0))[:3] - np.asarray(survey[tag])))
            err = np.array(err)
            out["absolute_m"] = {"rmse": float(np.sqrt(np.mean(err ** 2))), "median": float(np.median(err)),
                                 "max": float(err.max()), "tags": len(common)}
        else:
            out["absolute_m"] = f"needs >= 3 surveyed tags seen, have {len(common)}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seq")
    ap.add_argument("results", nargs="+")
    ap.add_argument("--survey", help="markers.yaml (family, size_m, tags: surveyed centres)")
    ap.add_argument("--family", help="tag family, default 36h11 (or the survey's)")
    ap.add_argument("--size", type=float, help="black square edge in m (or the survey's size_m)")
    ap.add_argument("--closed", action="store_true", help="the car ended where it started")
    ap.add_argument("--pass-gap", type=float, default=2.0, help="s without a sighting that starts a new pass")
    ap.add_argument("--max-dist", type=float, default=2.5, help="m, farther sightings are dropped")
    ap.add_argument("--min-side", type=float, default=30.0, help="px, smaller sightings are dropped")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    survey_cfg = {}
    if args.survey:
        with open(args.survey) as f:
            survey_cfg = yaml.safe_load(f) or {}
    family = args.family or str(survey_cfg.get("family", "36h11"))
    size = args.size or survey_cfg.get("size_m")
    if not size:
        raise SystemExit("give the tag size: --size or size_m in the survey")
    survey = {int(k): np.asarray(v, dtype=np.float64) for k, v in (survey_cfg.get("tags") or {}).items()}

    seq = rgbdi.load(args.seq)
    det = detect(seq, family, float(size))
    good = (np.linalg.norm(det[:, 2:5], axis=1) <= args.max_dist) & (det[:, 5] >= args.min_side)
    det = det[good]
    print(f"{len(det)} usable sightings of {len(set(det[:, 1].astype(int)))} tags")

    rows = []
    for res in args.results:
        t, T = rgbdi.read_poses(os.path.join(res, "poses.csv"))
        with open(os.path.join(res, "meta.json")) as f:
            meta = json.load(f)
        rows.append({"name": os.path.basename(os.path.normpath(res)), "mode": meta.get("mode"),
                     **score_result(det, t, T, survey, args.closed, args.pass_gap)})
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "markers.json"), "w") as f:
        json.dump(rows, f, indent=1)

    def g(d, *keys, fmt="{:.3f}"):
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return "–"
            d = d[k]
        return fmt.format(d) if isinstance(d, (int, float)) else str(d)

    lines = ["| run | mode | sightings tracked | repeatability RMS / max (m) | drift from pass 1, last pass (m) | "
             "absolute RMSE / max (m) | closure (m) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        drift = r.get("drift_from_first_pass_m") or {}
        last = f"{drift[max(drift)]:.3f} (pass {max(drift) + 1})" if drift else "–"
        lines.append(f"| {r['name']} | {r['mode']} | {r['sightings_tracked']} / {r['sightings']} | "
                     f"{g(r, 'repeatability_m', 'rms')} / {g(r, 'repeatability_m', 'max')} | {last} | "
                     f"{g(r, 'absolute_m', 'rmse')} / {g(r, 'absolute_m', 'max')} | {g(r, 'closure_m')} |")
    with open(os.path.join(args.out, "markers.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
