#!/usr/bin/env python3
import argparse
import glob
import math
import os
import struct
from bisect import bisect_right

import cv2
import numpy as np

FILE_HEADER_SIZE = 64
REC_HDR_FMT = "<QddIHH"
REC_HDR_SIZE = struct.calcsize(REC_HDR_FMT)


def read_records(path):
    recs = []
    with open(path, "rb") as f:
        f.seek(FILE_HEADER_SIZE)
        while True:
            h = f.read(REC_HDR_SIZE)
            if len(h) < REC_HDR_SIZE:
                break
            seq, produced_ts, read_ts, payload_size, schema_ver, dtype_tag = struct.unpack(REC_HDR_FMT, h)
            payload = f.read(payload_size)
            if len(payload) < payload_size:
                break
            recs.append((seq, produced_ts, payload))
    return recs


def latest_file(recording_dir, prefix):
    files = sorted(glob.glob(os.path.join(recording_dir, f"{prefix}_*.bin")))
    return files[-1] if files else None


def decode_frame(payload):
    rh, rw, dh, dw = struct.unpack_from("<IIII", payload, 0)
    meta_size = struct.calcsize("<IIII")
    rgb_size = rh * rw * 3
    rgb = np.frombuffer(payload, dtype=np.uint8, count=rgb_size, offset=meta_size).reshape((rh, rw, 3)).copy()
    depth_off = meta_size + rgb_size
    depth_bytes = len(payload) - depth_off
    if depth_bytes == dh * dw * 4:
        depth = np.frombuffer(payload, dtype=np.float32, count=dh * dw, offset=depth_off).reshape((dh, dw)).copy()
    else:
        depth = np.frombuffer(payload, dtype=np.uint16, count=dh * dw, offset=depth_off).reshape((dh, dw)).astype(np.float32)
    return rgb, depth


def decode_seg(payload):
    if len(payload) < struct.calcsize("<II"):
        return np.empty((0, 0), dtype=np.float32)
    h, w = struct.unpack_from("<II", payload, 0)
    off = struct.calcsize("<II")
    needed = off + h * w * 4
    if h <= 0 or w <= 0 or len(payload) < needed:
        return np.empty((0, 0), dtype=np.float32)
    seg = np.frombuffer(payload, dtype=np.float32, count=h * w, offset=off).reshape((h, w)).copy()
    return seg


def decode_pose(payload):
    pose = np.frombuffer(payload, dtype=np.float32, count=16, offset=0).reshape((4, 4)).copy()
    n = 0
    mpts = np.empty((0, 3), dtype=np.float32)
    if len(payload) >= 68:
        (n,) = struct.unpack_from("<I", payload, 64)
        off = 68
        need = off + n * 3 * 4
        if n > 0 and len(payload) >= need:
            mpts = np.frombuffer(payload, dtype=np.float32, count=n * 3, offset=off).reshape((n, 3)).copy()
    return pose, mpts


def decode_imu(payload):
    vec = np.frombuffer(payload, dtype=np.float32).copy()
    if vec.size < 16:
        out = np.zeros(16, dtype=np.float32)
        out[0] = 1.0
        out[: vec.size] = vec
        vec = out
    q = vec[0:4]
    n = np.linalg.norm(q)
    if n < 1e-8:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        q = q / n
    return q, vec


def decode_control(payload):
    if len(payload) < 8:
        return 0.0, 0.0
    return struct.unpack_from("<ff", payload, 0)


def decode_pointcloud(payload):
    if len(payload) < 4:
        return np.empty((0, 3), dtype=np.float32)
    (n,) = struct.unpack_from("<I", payload, 0)
    off = 4
    need = off + n * 3 * 4
    if n <= 0 or len(payload) < need:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.frombuffer(payload, dtype=np.float32, count=n * 3, offset=off).reshape((n, 3)).copy()
    return pts


def decode_boundary(payload):
    off = 0
    if len(payload) < 4:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    (n_outer,) = struct.unpack_from("<I", payload, off)
    off += 4
    need_outer = off + n_outer * 2 * 4
    if n_outer > 0 and len(payload) >= need_outer:
        outer = np.frombuffer(payload, dtype=np.float32, count=n_outer * 2, offset=off).reshape((n_outer, 2)).copy()
    else:
        outer = np.empty((0, 2), dtype=np.float32)
    off = need_outer
    if len(payload) < off + 4:
        return outer, np.empty((0, 2), dtype=np.float32)
    (n_inner,) = struct.unpack_from("<I", payload, off)
    off += 4
    need_inner = off + n_inner * 2 * 4
    if n_inner > 0 and len(payload) >= need_inner:
        inner = np.frombuffer(payload, dtype=np.float32, count=n_inner * 2, offset=off).reshape((n_inner, 2)).copy()
    else:
        inner = np.empty((0, 2), dtype=np.float32)
    return outer, inner


def depth_to_bgr(depth, size):
    d = depth.astype(np.float32)
    valid = np.isfinite(d)
    if np.any(valid):
        dv = d[valid]
        lo, hi = np.percentile(dv, 2), np.percentile(dv, 98)
        if hi <= lo:
            hi = lo + 1.0
        d = np.clip((d - lo) / (hi - lo), 0, 1)
    else:
        d = np.zeros_like(d)
    d8 = (d * 255).astype(np.uint8)
    d8 = cv2.resize(d8, size, interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(d8, cv2.COLORMAP_VIRIDIS)


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def render_orientation(q, size):
    h, w = size
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    verts = np.array([
        [-1.2, -0.6, -0.2], [1.2, -0.6, -0.2], [1.2, 0.6, -0.2], [-1.2, 0.6, -0.2],
        [-1.2, -0.6, 0.2], [1.2, -0.6, 0.2], [1.2, 0.6, 0.2], [-1.2, 0.6, 0.2],
    ], dtype=np.float32)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    r = quat_to_rot(q)
    rv = (r @ verts.T).T
    cam_z = 5.0
    f = min(h, w) * 0.42

    def proj(p):
        x, y, z = p
        zc = z + cam_z
        return int(w * 0.5 + f * (x / zc)), int(h * 0.5 - f * (y / zc))

    pts = [proj(p) for p in rv]
    for i, j in edges:
        cv2.line(img, pts[i], pts[j], (40, 40, 200), 2, cv2.LINE_AA)

    axes = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 0.9]], dtype=np.float32)
    ar = (r @ axes.T).T
    o = proj(ar[0])
    cv2.line(img, o, proj(ar[1]), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, o, proj(ar[2]), (0, 180, 0), 2, cv2.LINE_AA)
    cv2.line(img, o, proj(ar[3]), (255, 0, 0), 2, cv2.LINE_AA)
    return img


def draw_pose_distance_scale(panel, world_min_x, world_max_x):
    span_m = float(world_max_x - world_min_x)
    if span_m <= 1e-6:
        return

    usable_px = max(1, panel.shape[1] - 40)
    meters_per_px = span_m / usable_px
    target_m = meters_per_px * 110.0
    if target_m <= 0:
        return

    base = 10 ** math.floor(math.log10(target_m))
    bar_m = 5.0 * base
    for cand in (1.0, 2.0, 5.0):
        candidate_m = cand * base
        if candidate_m >= target_m:
            bar_m = candidate_m
            break

    bar_px = int(round(bar_m / meters_per_px))
    bar_px = max(30, min(usable_px - 10, bar_px))

    x0 = 20
    y0 = panel.shape[0] - 28
    x1 = x0 + bar_px
    cv2.line(panel, (x0, y0), (x1, y0), (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(panel, (x0, y0 - 6), (x0, y0 + 6), (30, 30, 30), 2, cv2.LINE_AA)
    cv2.line(panel, (x1, y0 - 6), (x1, y0 + 6), (30, 30, 30), 2, cv2.LINE_AA)

    if bar_m >= 10:
        label = f"{bar_m:.0f} m"
    elif bar_m >= 1:
        label = f"{bar_m:.1f} m"
    else:
        label = f"{bar_m:.2f} m"
    cv2.putText(panel, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)


def panel_with_title(panel, title, subtitle=""):
    out = panel.copy()
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def blank_panel(size, text):
    h, w = size
    img = np.full((h, w, 3), 240, dtype=np.uint8)
    cv2.putText(img, text, (12, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
    return img


def latest_at(ts, recs, rec_ts):
    if not recs:
        return None
    i = bisect_right(rec_ts, ts) - 1
    if i < 0:
        return None
    return recs[i]


def draw_surface_grid(img, margin=20, step=40):
    h, w = img.shape[:2]
    for x in range(margin, w - margin + 1, step):
        cv2.line(img, (x, margin), (x, h - margin), (230, 230, 230), 1, cv2.LINE_AA)
    for y in range(margin, h - margin + 1, step):
        cv2.line(img, (margin, y), (w - margin, y), (230, 230, 230), 1, cv2.LINE_AA)


def build_video(recording_dir, output_path, fps=20, max_frames=200):
    # Discover buffer files (latest file per buffer type).
    prefixes = ["frame", "segmentation", "pose", "imu", "imu_aux", "control", "pointcloud", "boundary"]
    files = {k: latest_file(recording_dir, k) for k in prefixes}

    if files["frame"] is None:
        raise RuntimeError("No frame_*.bin file found")

    recs = {k: (read_records(v) if v else []) for k, v in files.items()}
    ts = {k: [t for _, t, _ in recs[k]] for k in prefixes}

    frame_recs = recs["frame"]
    if not frame_recs:
        raise RuntimeError("No frame records found")

    # Path extents from all pose data.
    all_pose_xy = []
    for _, _, p in recs["pose"]:
        pose, _ = decode_pose(p)
        all_pose_xy.append((float(pose[0, 3]), float(pose[2, 3])))
    if all_pose_xy:
        arr = np.asarray(all_pose_xy, dtype=np.float32)
        min_x, min_z = arr.min(axis=0)
        max_x, max_z = arr.max(axis=0)
    else:
        min_x = min_z = -1.0
        max_x = max_z = 1.0
    full_pose_xy = list(all_pose_xy)

    # Only show panels for buffers that have files in this recording.
    # RGB and depth are rendered as separate panels from frame data.
    panel_keys = []
    if files["frame"] is not None:
        panel_keys.extend(["rgb", "depth"])
    for k in prefixes:
        if k == "frame":
            continue
        if files[k] is not None:
            panel_keys.append(k)

    cols = 4
    rows = int(math.ceil(len(panel_keys) / cols))
    panel_w, panel_h = 480, 300
    canvas_w, canvas_h = cols * panel_w, rows * panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    vw = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))

    path_so_far = []
    pc_seen_seq = set()
    pc_agg_xy = np.empty((0, 2), dtype=np.float32)
    pc_min_x = None
    pc_max_x = None
    pc_min_z = None
    pc_max_z = None
    pc_new_points = 0

    if max_frames is None or int(max_frames) <= 0:
        total_frames = len(frame_recs)
    else:
        total_frames = min(len(frame_recs), int(max_frames))
    for idx, (_, frame_ts, frame_payload) in enumerate(frame_recs[:total_frames]):
        rgb, depth = decode_frame(frame_payload)

        panels = {}

        # frame buffer panels (RGB and depth rendered separately)
        rgb_panel = cv2.resize(rgb, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
        panels["rgb"] = panel_with_title(rgb_panel, "rgb", f"seq {idx+1}/{total_frames}")

        depth_panel = depth_to_bgr(depth, (panel_w, panel_h))
        valid = np.isfinite(depth)
        if np.any(valid):
            dmin = float(np.nanmin(depth[valid]))
            dmax = float(np.nanmax(depth[valid]))
            depth_subtitle = f"{dmin:.2f}m to {dmax:.2f}m"
        else:
            depth_subtitle = "no valid depth"
        panels["depth"] = panel_with_title(depth_panel, "depth", depth_subtitle)

        # segmentation panel
        seg_rec = latest_at(frame_ts, recs["segmentation"], ts["segmentation"])
        if seg_rec is None:
            seg_panel = blank_panel((panel_h, panel_w), "no segmentation records")
        else:
            seg = decode_seg(seg_rec[2])
            if seg.size > 0:
                seg_bin = (seg > 0.5).astype(np.uint8) * 255
                seg_img = cv2.cvtColor(cv2.resize(seg_bin, (panel_w, panel_h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            else:
                seg_img = blank_panel((panel_h, panel_w), "segmentation empty payload")
            seg_panel = seg_img
        panels["segmentation"] = panel_with_title(seg_panel, "segmentation")

        # pose panel (path)
        pose_rec = latest_at(frame_ts, recs["pose"], ts["pose"])
        path_panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
        cv2.rectangle(path_panel, (1, 1), (panel_w - 2, panel_h - 2), (220, 220, 220), 1)
        if pose_rec is None:
            cv2.putText(path_panel, "no pose records", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
        else:
            pose, _ = decode_pose(pose_rec[2])
            path_so_far.append((float(pose[0, 3]), float(pose[2, 3])))
            if len(path_so_far) >= 2:
                sx = (panel_w - 40) / max(1e-6, (max_x - min_x))
                sz = (panel_h - 40) / max(1e-6, (max_z - min_z))
                pts = []
                for px, pz in path_so_far:
                    u = int(20 + (px - min_x) * sx)
                    v = int(panel_h - 20 - (pz - min_z) * sz)
                    pts.append((u, v))
                for a, b in zip(pts[:-1], pts[1:]):
                    cv2.line(path_panel, a, b, (255, 80, 50), 2, cv2.LINE_AA)
                cv2.circle(path_panel, pts[0], 4, (0, 180, 0), -1)
                cv2.circle(path_panel, pts[-1], 4, (0, 0, 255), -1)
            draw_pose_distance_scale(path_panel, min_x, max_x)
        panels["pose"] = panel_with_title(path_panel, "pose (trajectory)")

        # imu panel (orientation + stats)
        imu_rec = latest_at(frame_ts, recs["imu"], ts["imu"])
        if imu_rec is None:
            imu_panel = blank_panel((panel_h, panel_w), "no imu records")
        else:
            q, vec = decode_imu(imu_rec[2])
            imu_panel = render_orientation(q, (panel_h, panel_w))
            txt = f"acc=({vec[4]:+.2f},{vec[5]:+.2f},{vec[6]:+.2f}) gyro=({vec[7]:+.2f},{vec[8]:+.2f},{vec[9]:+.2f})"
            cv2.putText(imu_panel, txt, (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.putText(imu_panel, f"thr={vec[13]:+.2f} steer={vec[14]:+.2f}", (10, panel_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        panels["imu"] = panel_with_title(imu_panel, "imu")

        # imu_aux panel (I2C throttle/steering + inertial stats)
        imu_aux_rec = latest_at(frame_ts, recs["imu_aux"], ts["imu_aux"])
        imu_aux_panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
        if imu_aux_rec is None:
            cv2.putText(imu_aux_panel, "no imu_aux records", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
        else:
            q2, vec2 = decode_imu(imu_aux_rec[2])
            cv2.putText(imu_aux_panel, "I2C aux stream", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 2, cv2.LINE_AA)
            cv2.putText(
                imu_aux_panel,
                f"acc=({vec2[4]:+.2f},{vec2[5]:+.2f},{vec2[6]:+.2f})",
                (12, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                imu_aux_panel,
                f"gyro=({vec2[7]:+.2f},{vec2[8]:+.2f},{vec2[9]:+.2f})",
                (12, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )

            steering2 = float(vec2[14])
            throttle2 = float(vec2[13])
            sx2 = int((steering2 + 1.0) * 0.5 * (panel_w - 40)) if abs(steering2) <= 1.0 else int(steering2 * (panel_w - 40))
            tx2 = int(max(0, min(panel_w - 40, throttle2 * (panel_w - 40))))
            cv2.rectangle(imu_aux_panel, (20, 170), (20 + sx2, 205), (20, 120, 255), -1)
            cv2.rectangle(imu_aux_panel, (20, 245), (20 + tx2, 280), (20, 180, 20), -1)
            cv2.rectangle(imu_aux_panel, (20, 170), (panel_w - 20, 205), (180, 180, 180), 1)
            cv2.rectangle(imu_aux_panel, (20, 245), (panel_w - 20, 280), (180, 180, 180), 1)
            cv2.putText(imu_aux_panel, f"steer={steering2:+.3f}", (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(imu_aux_panel, f"throttle={throttle2:+.3f}", (20, 237), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(
                imu_aux_panel,
                f"q=({q2[0]:+.2f},{q2[1]:+.2f},{q2[2]:+.2f},{q2[3]:+.2f})",
                (12, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )
        panels["imu_aux"] = panel_with_title(imu_aux_panel, "imu_aux throttle+steer")

        # control panel (bars)
        ctrl_rec = latest_at(frame_ts, recs["control"], ts["control"])
        control_panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
        if ctrl_rec is None:
            cv2.putText(control_panel, "no control records", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
        else:
            steering, throttle = decode_control(ctrl_rec[2])
            sx = int((steering + 1.0) * 0.5 * (panel_w - 40)) if abs(steering) <= 1.0 else int(steering * (panel_w - 40))
            tx = int(max(0, min(panel_w - 40, throttle * (panel_w - 40))))
            cv2.rectangle(control_panel, (20, 90), (20 + sx, 130), (20, 120, 255), -1)
            cv2.rectangle(control_panel, (20, 200), (20 + tx, 240), (20, 180, 20), -1)
            cv2.rectangle(control_panel, (20, 90), (panel_w - 20, 130), (180, 180, 180), 1)
            cv2.rectangle(control_panel, (20, 200), (panel_w - 20, 240), (180, 180, 180), 1)
            cv2.putText(control_panel, f"steering={steering:+.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
            cv2.putText(control_panel, f"throttle={throttle:+.3f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        panels["control"] = panel_with_title(control_panel, "control")

        # pointcloud panel
        pc_rec = latest_at(frame_ts, recs["pointcloud"], ts["pointcloud"])
        pc_panel = np.full((panel_h, panel_w, 3), 250, dtype=np.uint8)
        draw_surface_grid(pc_panel)
        if pc_rec is None:
            cv2.putText(pc_panel, "no pointcloud records", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
        else:
            seq = int(pc_rec[0])
            if seq not in pc_seen_seq:
                pts = decode_pointcloud(pc_rec[2])
                if len(pts) > 0:
                    # Keep an aggregated 2D cloud over time (X,Z surface projection).
                    stride = max(1, len(pts) // 5000)
                    xy = pts[::stride][:, [0, 2]].astype(np.float32)
                    if len(xy) > 0:
                        pc_agg_xy = np.vstack([pc_agg_xy, xy])
                        if len(pc_agg_xy) > 180000:
                            keep = np.linspace(0, len(pc_agg_xy) - 1, 120000, dtype=np.int32)
                            pc_agg_xy = pc_agg_xy[keep]
                        bmin = xy.min(axis=0)
                        bmax = xy.max(axis=0)
                        pc_min_x = float(bmin[0]) if pc_min_x is None else min(pc_min_x, float(bmin[0]))
                        pc_max_x = float(bmax[0]) if pc_max_x is None else max(pc_max_x, float(bmax[0]))
                        pc_min_z = float(bmin[1]) if pc_min_z is None else min(pc_min_z, float(bmin[1]))
                        pc_max_z = float(bmax[1]) if pc_max_z is None else max(pc_max_z, float(bmax[1]))
                        pc_new_points = len(xy)
                pc_seen_seq.add(seq)

            if len(pc_agg_xy) == 0 or pc_min_x is None:
                cv2.putText(pc_panel, "pointcloud empty", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
            else:
                sx = (panel_w - 40) / max(1e-6, pc_max_x - pc_min_x)
                sz = (panel_h - 40) / max(1e-6, pc_max_z - pc_min_z)
                sample = pc_agg_xy[:: max(1, len(pc_agg_xy) // 20000)]
                for xz in sample:
                    u = int(20 + (xz[0] - pc_min_x) * sx)
                    v = int(panel_h - 20 - (xz[1] - pc_min_z) * sz)
                    if 0 <= u < panel_w and 0 <= v < panel_h:
                        pc_panel[v, u] = (200, 60, 30)
                cv2.putText(pc_panel, f"aggN={len(pc_agg_xy)}  +{pc_new_points}", (10, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        # Overlay pose trajectory on the same aggregate map panel.
        map_min_x = min_x
        map_max_x = max_x
        map_min_z = min_z
        map_max_z = max_z
        if pc_min_x is not None:
            map_min_x = min(map_min_x, pc_min_x)
            map_max_x = max(map_max_x, pc_max_x)
            map_min_z = min(map_min_z, pc_min_z)
            map_max_z = max(map_max_z, pc_max_z)

        sx = (panel_w - 40) / max(1e-6, map_max_x - map_min_x)
        sz = (panel_h - 40) / max(1e-6, map_max_z - map_min_z)

        def to_uv(px, pz):
            u = int(20 + (px - map_min_x) * sx)
            v = int(panel_h - 20 - (pz - map_min_z) * sz)
            return (u, v)

        if len(full_pose_xy) >= 2:
            full_pts = [to_uv(px, pz) for px, pz in full_pose_xy]
            for a, b in zip(full_pts[:-1], full_pts[1:]):
                cv2.line(pc_panel, a, b, (165, 165, 165), 1, cv2.LINE_AA)

        if len(path_so_far) >= 2:
            traveled_pts = [to_uv(px, pz) for px, pz in path_so_far]
            for a, b in zip(traveled_pts[:-1], traveled_pts[1:]):
                cv2.line(pc_panel, a, b, (255, 80, 40), 2, cv2.LINE_AA)
            cv2.circle(pc_panel, traveled_pts[-1], 4, (0, 0, 255), -1)

        panels["pointcloud"] = panel_with_title(pc_panel, "pointcloud + pose (aggregate 2D)")

        # boundary panel
        bnd_rec = latest_at(frame_ts, recs["boundary"], ts["boundary"])
        b_panel = np.full((panel_h, panel_w, 3), 250, dtype=np.uint8)
        if bnd_rec is None:
            cv2.putText(b_panel, "no boundary records", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
        else:
            outer, inner = decode_boundary(bnd_rec[2])
            if len(outer) == 0 and len(inner) == 0:
                cv2.putText(b_panel, "boundary empty", (12, panel_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 60), 2, cv2.LINE_AA)
            else:
                all_pts = np.vstack([x for x in (outer, inner) if len(x) > 0])
                xmin, zmin = all_pts.min(axis=0)
                xmax, zmax = all_pts.max(axis=0)
                sx = (panel_w - 40) / max(1e-6, xmax - xmin)
                sz = (panel_h - 40) / max(1e-6, zmax - zmin)

                def draw_line(arr, color):
                    if len(arr) < 2:
                        return
                    pts2 = []
                    for px, pz in arr:
                        u = int(20 + (px - xmin) * sx)
                        v = int(panel_h - 20 - (pz - zmin) * sz)
                        pts2.append((u, v))
                    for a, b in zip(pts2[:-1], pts2[1:]):
                        cv2.line(b_panel, a, b, color, 2, cv2.LINE_AA)

                draw_line(outer, (40, 100, 255))
                draw_line(inner, (40, 200, 40))
        panels["boundary"] = panel_with_title(b_panel, "boundary")

        canvas = np.full((canvas_h, canvas_w, 3), 230, dtype=np.uint8)
        for pi, key in enumerate(panel_keys):
            r = pi // cols
            c = pi % cols
            y0 = r * panel_h
            x0 = c * panel_w
            panel = panels.get(key)
            if panel is None:
                panel = blank_panel((panel_h, panel_w), f"no renderer for {key}")
                panel = panel_with_title(panel, key)
            canvas[y0:y0 + panel_h, x0:x0 + panel_w] = panel

        cv2.putText(canvas, f"recording={os.path.basename(recording_dir)}  frame {idx+1}/{total_frames}", (12, canvas_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        vw.write(canvas)

    vw.release()
    return total_frames, panel_keys


def main():
    ap = argparse.ArgumentParser(description="Render full recording to a per-buffer summary MP4")
    ap.add_argument("--recording", required=True, help="Path to recording_<timestamp> directory")
    ap.add_argument("--output", required=True, help="Output .mp4 file")
    ap.add_argument("--fps", type=int, default=20, help="Output video FPS")
    ap.add_argument(
        "--max-frames",
        type=int,
        default=200,
        help="Maximum number of frames to render (<=0 renders all frames)",
    )
    args = ap.parse_args()

    n, panel_keys = build_video(
        args.recording,
        args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(f"saved_video={args.output}")
    print(f"frames_rendered={n}")
    print("panels=" + ",".join(panel_keys))


if __name__ == "__main__":
    main()
