#!/usr/bin/env python3
"""Turn a recording (start.py --recording) into training datasets.

Frames from the `frame` buffer are paired with the label sample nearest in
time (produced_ts, time.monotonic() clock) within --tolerance seconds; frames
without a label that close are dropped.

--bc OUT.npz   behaviour-cloning set for scripts/train_bc_policy.py and
               scripts/train_vae_policy.py (--data OUT.npz):
                 x (N, 1, 40, 64) float32  depth preprocessed with
                                           control.imitation.preprocess_depth
                 y (N, 1) float32          signed steering, +1 = full LEFT
               plus t (N,) frame timestamps (ignored by the trainers).
               --label-source picks where steering comes from:
                 imu      the steering pulse width the Arduino reports in IMU
                          slot 8 (I2C IMU path, IMUWorker). USE THIS FOR
                          HAND-DRIVEN LAPS: when you drive with the RC
                          transmitter nothing goes through ctrl_buf, so the
                          control buffer holds no driver commands. Pulse
                          widths map through the `pwm` config section:
                          unit = (pw - steer_min) / (steer_max - steer_min),
                          signed = 1 - 2 * unit. Needs the I2C IMU (not the
                          OAK-D IMU, whose slot 8 is a gyro rate).
                 control  ctrl_buf values in [0, 1] converted with
                          workers.control_worker.unit_to_signed, i.e. what the
                          on-board controller commanded (autonomous runs).
               Mirrored copies (depth flipped left-right, steering negated)
               are added unless --no-mirror, like the synthetic demos.

--vision DIR   images/ (+ masks/) layout for scripts/train_vision.py
               (vision/data.py). Masks are not recorded. --bootstrap-masks
               writes masks from the training-free GroundPlaneSegmenter
               (RANSAC floor + free space, needs the intrinsics from
               --camera-settings). THESE ARE AUTO-LABELS: review and fix them
               before training; DIR/auto_labels.json lists every mask that
               was generated rather than hand-labelled.

    python scripts/recording_to_dataset.py /tmp/pilot_recordings --bc laps.npz --label-source imu
    python scripts/recording_to_dataset.py /tmp/pilot_recordings --vision track_ds --stride 5 \\
        --bootstrap-masks --camera-settings orbslam_oakd_rgbd.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.pwm import PWMConfig  # noqa: E402
from scripts.read_recording import iter_records, load_buffer  # noqa: E402

LABEL_SOURCES = ("imu", "control")
IMU_STEERING_SLOT = 8
PWM_MARGIN_US = 100      # pulse widths further than this outside [steer_min, steer_max] are invalid


# pairing

def nearest_indices(t_query: np.ndarray, t_ref: np.ndarray, tolerance: float) -> np.ndarray:
    """For each t_query, the index of the nearest t_ref (sorted), or -1 if
    the nearest is further than tolerance seconds away."""
    t_query = np.asarray(t_query, dtype=np.float64)
    t_ref = np.asarray(t_ref, dtype=np.float64)
    if len(t_ref) == 0:
        return np.full(len(t_query), -1, dtype=np.int64)
    right = np.clip(np.searchsorted(t_ref, t_query), 0, len(t_ref) - 1)
    left = np.clip(right - 1, 0, len(t_ref) - 1)
    pick = np.where(np.abs(t_ref[left] - t_query) <= np.abs(t_ref[right] - t_query), left, right)
    pick = pick.astype(np.int64)
    pick[np.abs(t_ref[pick] - t_query) > tolerance] = -1
    return pick


# labels

def load_pwm_config(config_path: Optional[str]) -> PWMConfig:
    """PWMConfig from the `pwm` section of a pilot config yaml (defaults if None)."""
    if not config_path:
        return PWMConfig()
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    pwm = cfg.get("pwm", {}) or {}
    fields = PWMConfig.__dataclass_fields__
    return PWMConfig(**{k: int(v) for k, v in pwm.items() if k in fields})


def pwm_to_signed_steering(pw: np.ndarray, cfg: PWMConfig) -> np.ndarray:
    """Steering pulse width (us) -> signed steering, +1 = full left."""
    unit = (np.asarray(pw, dtype=np.float64) - cfg.steer_min) / (cfg.steer_max - cfg.steer_min)
    return 1.0 - 2.0 * np.clip(unit, 0.0, 1.0)


def steering_labels(
    recording: str, source: str, pwm: Optional[PWMConfig] = None, session: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(timestamps, signed steering) sorted by time, invalid samples removed."""
    if source == "imu":
        pwm = pwm or PWMConfig()
        recs = load_buffer(recording, "imu", session)
        t = np.array([r.produced_ts for r in recs], dtype=np.float64)
        pw = np.array([float(r.value[IMU_STEERING_SLOT]) for r in recs], dtype=np.float64)
        ok = (pw >= pwm.steer_min - PWM_MARGIN_US) & (pw <= pwm.steer_max + PWM_MARGIN_US)
        return t[ok], pwm_to_signed_steering(pw[ok], pwm)
    if source == "control":
        from workers.control_worker import unit_to_signed

        recs = load_buffer(recording, "control", session)
        t = np.array([r.produced_ts for r in recs], dtype=np.float64)
        y = np.array([unit_to_signed(*r.value)[0] for r in recs], dtype=np.float64)
        return t, y
    raise ValueError(f"label source must be one of {LABEL_SOURCES}, got {source!r}")


def _depth_metres(depth: np.ndarray) -> np.ndarray:
    # FrameBuffer carries float32 metres; uint16 would be millimetres
    return depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth


def _frames(recording: str, session: Optional[str], stride: int):
    for i, rec in enumerate(iter_records(recording, "frame", session)):
        if i % stride == 0:
            yield rec


# --bc

def build_bc_dataset(
    recording: str,
    out_path: str,
    label_source: str = "imu",
    tolerance: float = 0.05,
    max_range: float = 4.0,
    stride: int = 1,
    mirror: bool = True,
    label_offset: float = 0.0,
    pwm: Optional[PWMConfig] = None,
    session: Optional[str] = None,
) -> dict:
    """Write the .npz used by train_bc_policy.py / train_vae_policy.py and
    return counts. label_offset shifts the label lookup (frame_ts + offset),
    e.g. to compensate for the driver's reaction time."""
    from control.imitation import preprocess_depth

    t_lab, y_lab = steering_labels(recording, label_source, pwm, session)
    xs, ys, ts = [], [], []
    n_frames = 0
    for rec in _frames(recording, session, stride):
        n_frames += 1
        idx = nearest_indices([rec.produced_ts + label_offset], t_lab, tolerance)[0]
        if idx < 0:
            continue
        _, depth = rec.value
        xs.append(preprocess_depth(_depth_metres(depth), max_range))
        ys.append(y_lab[idx])
        ts.append(rec.produced_ts)
    if not xs:
        raise RuntimeError(
            f"no frame had a {label_source} label within {tolerance} s "
            f"({n_frames} frames, {len(t_lab)} valid labels)"
        )
    x = np.stack(xs)[:, None].astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)[:, None]
    t = np.asarray(ts, dtype=np.float64)
    n_paired = len(x)
    if mirror:
        x = np.concatenate([x, x[..., ::-1]])
        y = np.concatenate([y, -y])
        t = np.concatenate([t, t])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, x=np.ascontiguousarray(x), y=y, t=t)
    return {"frames": n_frames, "labels": len(t_lab), "paired": n_paired, "samples": len(x)}


# --vision

def _settings_size(path: str) -> Optional[Tuple[int, int]]:
    w = h = None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("Camera.width:"):
                w = int(float(s.split(":", 1)[1]))
            elif s.startswith("Camera.height:"):
                h = int(float(s.split(":", 1)[1]))
    return (w, h) if w and h else None


def _ground_plane_segmenter(camera_settings: str, frame_w: int, frame_h: int):
    from control.depth_ftg import Intrinsics
    from vision.segmenters import GroundPlaneSegmenter

    K = Intrinsics.from_orbslam_settings(camera_settings)
    size = _settings_size(camera_settings)
    if size and size != (frame_w, frame_h):
        sx, sy = frame_w / size[0], frame_h / size[1]
        K = Intrinsics(K.fx * sx, K.fy * sy, K.cx * sx, K.cy * sy)
    return GroundPlaneSegmenter(K)


def build_vision_dataset(
    recording: str,
    out_dir: str,
    stride: int = 1,
    bootstrap_masks: bool = False,
    camera_settings: str = "orbslam_oakd_rgbd.yaml",
    session: Optional[str] = None,
) -> dict:
    """Write out_dir/images/*.png (and out_dir/masks/*.png auto-labels with
    bootstrap_masks) and return counts."""
    import cv2

    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    seg = None
    auto, empty = [], 0
    n = 0
    for rec in _frames(recording, session, stride):
        rgb, depth = rec.value
        stem = f"{rec.session or rec.buffer}_{rec.seq_no:08d}"
        cv2.imwrite(os.path.join(img_dir, stem + ".png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        n += 1
        if not bootstrap_masks:
            continue
        if seg is None:
            os.makedirs(mask_dir, exist_ok=True)
            seg = _ground_plane_segmenter(camera_settings, depth.shape[1], depth.shape[0])
        mask = seg.get_mask(_depth_metres(depth))
        if mask is None or not mask.any():
            empty += 1          # no floor found: leave this image unlabelled
            continue
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(mask_dir, stem + ".png"), (mask > 0).astype(np.uint8) * 255)
        auto.append(stem + ".png")
    if bootstrap_masks:
        with open(os.path.join(out_dir, "auto_labels.json"), "w") as f:
            json.dump({
                "note": "Masks generated by vision.segmenters.GroundPlaneSegmenter (training-free). "
                        "They are automatic labels, not ground truth: review and correct them before training.",
                "camera_settings": camera_settings,
                "auto_masks": auto,
            }, f, indent=1)
    return {"images": n, "auto_masks": len(auto), "no_floor": empty}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("recording", help="recording directory")
    ap.add_argument("--session", help="only this session id (the number in the file names)")
    ap.add_argument("--bc", metavar="OUT.npz", help="write a behaviour-cloning dataset")
    ap.add_argument("--vision", metavar="DIR", help="write an images/ (+ masks/) segmentation dataset")
    ap.add_argument("--label-source", choices=LABEL_SOURCES, default="imu",
                    help="steering labels for --bc: imu (hand-driven laps) or control (autonomous runs)")
    ap.add_argument("--config", default=None,
                    help="pilot config yaml whose `pwm` section converts IMU pulse widths (default: PWMConfig defaults)")
    ap.add_argument("--tolerance", type=float, default=0.05, help="max frame/label time difference, s (default 0.05)")
    ap.add_argument("--label-offset", type=float, default=0.0, help="look labels up at frame time + offset, s")
    ap.add_argument("--max-range", type=float, default=4.0, help="depth clip / normalisation for --bc, m (default 4.0, as the trainers)")
    ap.add_argument("--no-mirror", action="store_true", help="--bc: don't add left-right mirrored copies")
    ap.add_argument("--stride", type=int, default=1, help="use every N-th recorded frame")
    ap.add_argument("--bootstrap-masks", action="store_true", help="--vision: auto-label masks with GroundPlaneSegmenter (review them!)")
    ap.add_argument("--camera-settings", default="orbslam_oakd_rgbd.yaml", help="ORB-SLAM3 settings with the intrinsics")
    args = ap.parse_args(argv)

    if not args.bc and not args.vision:
        ap.error("nothing to do: give --bc and/or --vision")
    if args.bc:
        stats = build_bc_dataset(
            args.recording, args.bc, args.label_source, args.tolerance, args.max_range,
            max(1, args.stride), not args.no_mirror, args.label_offset,
            load_pwm_config(args.config), args.session,
        )
        print(f"--bc: {stats['paired']} of {stats['frames']} frames paired with {args.label_source} labels "
              f"({stats['labels']} valid), {stats['samples']} samples -> {args.bc}")
    if args.vision:
        stats = build_vision_dataset(
            args.recording, args.vision, max(1, args.stride), args.bootstrap_masks,
            args.camera_settings, args.session,
        )
        msg = f"--vision: {stats['images']} images -> {args.vision}/images"
        if args.bootstrap_masks:
            msg += (f", {stats['auto_masks']} AUTO-LABEL masks -> {args.vision}/masks "
                    f"({stats['no_floor']} frames without a floor left unlabelled); review them before training")
        print(msg)
    return 0


__all__ = ["build_bc_dataset", "build_vision_dataset", "nearest_indices",
           "pwm_to_signed_steering", "steering_labels", "load_pwm_config"]

if __name__ == "__main__":
    sys.exit(main())
