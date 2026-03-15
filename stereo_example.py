#!/usr/bin/env python3

"""Stereo RGB + Depth (aligned) via DepthAI v3.

This example uses DepthAI's v3 builder API with three camera nodes (RGB + stereo
pair) and a `StereoDepth` node. Frames are synced (RGB + depth) using `Sync`.

On RVC4 devices, depth is aligned to RGB using `ImageAlign`.
On other platforms, alignment uses `stereo.inputAlignTo`.
"""

from __future__ import annotations

import argparse
from collections import deque
from datetime import timedelta
import time

import cv2
import depthai as dai
import numpy as np


FPS: float = 60.0
SIZE: tuple[int, int] = (640, 400)
STEREO_SIZE: tuple[int, int] = (1280, 800)
DEPTH_PRESET: str = "FAST_ACCURACY"

RGB_SOCKET = dai.CameraBoardSocket.CAM_A
LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C

IR_DOT_INTENSITY = 0.8
IR_FLOOD_INTENSITY = 0.8

RVC2_PRESETS: set[str] = {
    "FAST_ACCURACY",
    "FAST_DENSITY",
    "DEFAULT",
    "FACE",
    "HIGH_DETAIL",
    "ROBOTICS",
}
RVC4_PRESETS: set[str] = {"ACCURACY", "DENSITY"}

PRESET_CANDIDATES: dict[str, list[str]] = {
    "FAST_ACCURACY": ["FAST_ACCURACY"],
    "FAST_DENSITY": ["FAST_DENSITY"],
    "DEFAULT": ["DEFAULT"],
    "FACE": ["FACE"],
    "HIGH_DETAIL": ["HIGH_DETAIL", "HIGH_DENSITY"],
    "ROBOTICS": ["ROBOTICS"],
    "ACCURACY": ["ACCURACY", "HIGH_ACCURACY"],
    "DENSITY": ["DENSITY", "HIGH_DENSITY"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aligned stereo depth tuned for OAK-D Pro with preset selection"
    )
    parser.add_argument(
        "--depth-preset",
        default=DEPTH_PRESET,
        type=str,
        help=(
            "Stereo preset. RVC2: FAST_ACCURACY, FAST_DENSITY, DEFAULT, FACE, "
            "HIGH_DETAIL, ROBOTICS. RVC4: ACCURACY, DENSITY."
        ),
    )
    parser.add_argument(
        "--fps",
        default=FPS,
        type=float,
        help="Output FPS target for RGB and stereo streams.",
    )
    return parser.parse_args()


def is_rvc4_platform(platform: dai.Platform) -> bool:
    return "RVC4" in str(platform)


def resolve_depth_preset(
    requested_name: str,
    platform: dai.Platform,
) -> tuple[object, str, str]:
    mode_cls = dai.node.StereoDepth.PresetMode
    available: dict[str, object] = {
        name: getattr(mode_cls, name)
        for name in dir(mode_cls)
        if name.isupper() and not name.startswith("_")
    }

    requested = requested_name.strip().upper()
    if requested not in PRESET_CANDIDATES:
        requested = DEPTH_PRESET

    effective = requested
    if is_rvc4_platform(platform) and requested in RVC2_PRESETS:
        effective = "ACCURACY"
    elif not is_rvc4_platform(platform) and requested in RVC4_PRESETS:
        effective = "FAST_ACCURACY"

    for candidate in PRESET_CANDIDATES[effective]:
        if candidate in available:
            return available[candidate], effective, candidate

    for fallback in ("FAST_ACCURACY", "DEFAULT", "ACCURACY", "DENSITY"):
        for candidate in PRESET_CANDIDATES[fallback]:
            if candidate in available:
                return available[candidate], effective, candidate

    raise RuntimeError("No compatible StereoDepth preset found in this DepthAI build")


def ir_levels_from_luma(mean_luma: float) -> tuple[float, float]:
    """Map scene brightness to IR levels for better depth robustness."""
    if mean_luma >= 185.0:
        return 0.05, 0.10
    if mean_luma >= 150.0:
        return 0.20, 0.25
    if mean_luma >= 110.0:
        return 0.45, 0.50
    if mean_luma >= 75.0:
        return 0.65, 0.70
    return 0.85, 0.90


def colorize_depth_mm(depth_mm: np.ndarray) -> np.ndarray:
    """Colorize a depth frame in millimetres (uint16) into BGR (uint8)."""
    if depth_mm.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    invalid = depth_mm == 0
    valid = depth_mm[~invalid]
    if valid.size == 0:
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)

    near = np.percentile(valid, 3)
    far = np.percentile(valid, 97)
    if far <= near:
        far = near + 1

    depth8 = np.interp(depth_mm.astype(np.float32), (near, far), (255.0, 0.0)).astype(
        np.uint8
    )

    colored = cv2.applyColorMap(depth8, cv2.COLORMAP_JET)
    return colored


def latency_summary_ms(samples: deque[float], latest_ms: float) -> str:
    if not samples:
        return f"{latest_ms:.2f} ms"
    arr = np.asarray(samples, dtype=np.float32)
    return (
        f"{latest_ms:.2f} ms avg={float(arr.mean()):.2f} ms "
        f"std={float(arr.std()):.2f}"
    )


args = parse_args()
fps = float(args.fps)

with dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS) as device:
    pipeline = dai.Pipeline(device)
    pipeline.setXLinkChunkSize(0)
    platform = pipeline.getDefaultDevice().getPlatform()

    cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET)
    cam_left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
    cam_right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

    stereo = pipeline.create(dai.node.StereoDepth)
    sync = pipeline.create(dai.node.Sync)
    align = pipeline.create(dai.node.ImageAlign) if platform == dai.Platform.RVC4 else None

    selected_preset, effective_preset_name, sdk_preset_name = resolve_depth_preset(
        str(args.depth_preset),
        platform,
    )
    stereo.setDefaultProfilePreset(selected_preset)
    stereo.setRectification(True)

    # Keep preset-defined filtering and stereo behavior.

    sync.setSyncThreshold(timedelta(milliseconds=25)) #seconds=1.0 / (2.0 * FPS)))

    rgb_out = cam_rgb.requestOutput(SIZE, fps=fps, enableUndistortion=True)
    left_out = cam_left.requestOutput(STEREO_SIZE, fps=fps)
    right_out = cam_right.requestOutput(STEREO_SIZE, fps=fps)

    left_out.link(stereo.left)
    right_out.link(stereo.right)

    rgb_out.link(sync.inputs["rgb"])
    try:
        sync.inputs["rgb"].setBlocking(False)
    except Exception:
        pass

    if align is not None:
        stereo.depth.link(align.input)
        rgb_out.link(align.inputAlignTo)
        align.outputAligned.link(sync.inputs["depth"])
    else:
        stereo.depth.link(sync.inputs["depth"])
        try:
            rgb_out.link(stereo.inputAlignTo)
        except Exception:
            # If alignment isn't supported, we still keep synced depth+rgb streams.
            pass

    queue = sync.out.createOutputQueue(maxSize=4, blocking=False)

    with pipeline:
        pipeline.start()

        print(f"USB speed: {device.getUsbSpeed()}")
        print(
            f"Requested preset={str(args.depth_preset).upper()} "
            f"effective={effective_preset_name} sdk={sdk_preset_name} "
            f"platform={platform} fps={fps} rgb={SIZE} mono={STEREO_SIZE}"
        )
        try:
            device.setIrLaserDotProjectorIntensity(IR_DOT_INTENSITY)
            device.setIrFloodLightIntensity(IR_FLOOD_INTENSITY)
        except RuntimeError as ex:
            print(f"IR controls unavailable on this device: {ex}")

        last_print = time.time()
        last_ir_update = 0.0
        current_dot_ir = IR_DOT_INTENSITY
        current_flood_ir = IR_FLOOD_INTENSITY
        frames = 0
        rgb_latency_samples: deque[float] = deque(maxlen=240)
        depth_latency_samples: deque[float] = deque(maxlen=240)

        while pipeline.isRunning():
            message_group = queue.get()
            assert isinstance(message_group, dai.MessageGroup)
            frame_rgb = message_group["rgb"]
            frame_depth = message_group["depth"]
            assert isinstance(frame_rgb, dai.ImgFrame)
            assert isinstance(frame_depth, dai.ImgFrame)

            rgb = frame_rgb.getCvFrame()
            depth_mm = frame_depth.getFrame()
            rgb_latency_ms = (
                dai.Clock.now() - frame_rgb.getTimestamp()
            ).total_seconds() * 1000.0
            depth_latency_ms = (
                dai.Clock.now() - frame_depth.getTimestamp()
            ).total_seconds() * 1000.0
            rgb_latency_samples.append(float(rgb_latency_ms))
            depth_latency_samples.append(float(depth_latency_ms))

            now = time.time()
            if now - last_ir_update >= 0.5:
                mean_luma = float(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).mean())
                target_dot_ir, target_flood_ir = ir_levels_from_luma(mean_luma)
                if (
                    abs(target_dot_ir - current_dot_ir) >= 0.05
                    or abs(target_flood_ir - current_flood_ir) >= 0.05
                ):
                    try:
                        device.setIrLaserDotProjectorIntensity(target_dot_ir)
                        device.setIrFloodLightIntensity(target_flood_ir)
                        current_dot_ir = target_dot_ir
                        current_flood_ir = target_flood_ir
                    except RuntimeError:
                        pass
                last_ir_update = now

            depth_vis = colorize_depth_mm(depth_mm)

            cv2.imshow("rgb", rgb)
            cv2.imshow("depth_aligned", depth_vis)

            frames += 1
            if now - last_print >= 2.0:
                fps_meas = frames / (now - last_print)
                print(
                    f"FPS ~ {fps_meas:.1f}  "
                    f"rgb={rgb.shape} depth={depth_mm.shape} "
                    f"mono={STEREO_SIZE} platform={platform} "
                    f"ir(dot={current_dot_ir:.2f}, flood={current_flood_ir:.2f}) "
                    f"lat_rgb[{latency_summary_ms(rgb_latency_samples, rgb_latency_ms)}] "
                    f"lat_depth[{latency_summary_ms(depth_latency_samples, depth_latency_ms)}]"
                )
                frames = 0
                last_print = now

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

cv2.destroyAllWindows()
