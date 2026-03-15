"""Color camera latency measurement (DepthAI v3).

Originally based on Luxonis latency measurement example, but updated to the
DepthAI v3 builder API (`node.Camera.build(...)`, `requestOutput(...)`,
`createOutputQueue(...)`).
"""

from __future__ import annotations

import argparse
import depthai as dai
import numpy as np


FPS: float = 120.0
SIZE: tuple[int, int] = (640, 400)
RGB_SOCKET = dai.CameraBoardSocket.CAM_A


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure OAK-D RGB latency")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after N frames (0 = run continuously)",
    )
    return parser.parse_args()


def print_stats(latency_ms_samples: list[float], latency_ms: float) -> None:
    if not latency_ms_samples:
        print(f"Latency: {latency_ms:.2f} ms")
        return
    arr = np.asarray(latency_ms_samples, dtype=np.float32)
    print(
        "Latency: {:.2f} ms, Average latency: {:.2f} ms, Std: {:.2f}".format(
            latency_ms, float(arr.mean()), float(arr.std())
        )
    )

def _create_device():
    try:
        return dai.Device(maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)
    except TypeError:
        return dai.Device()


def _build_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    # This might improve reducing the latency on some systems
    pipeline.setXLinkChunkSize(0)

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("rgb")

    # DepthAI v3 builder API
    if hasattr(dai.node.Camera, "build"):
        cam_rgb = pipeline.create(dai.node.Camera).build(RGB_SOCKET, sensorFps=FPS)
        cam_out = cam_rgb.requestOutput(SIZE, fps=FPS)
        cam_out.link(xout.input)
        return pipeline

    # DepthAI v2 classic API fallback
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(RGB_SOCKET)
    cam_rgb.setInterleaved(False)
    cam_rgb.setPreviewSize(*SIZE)
    cam_rgb.setFps(FPS)
    cam_rgb.preview.link(xout.input)
    return pipeline


def main() -> int:
    args = parse_args()

    with _create_device() as device:
        pipeline = _build_pipeline()
        device.startPipeline(pipeline)
        q = device.getOutputQueue("rgb", maxSize=4, blocking=False)

        print(device.getUsbSpeed())

        samples: list[float] = []
        frames = 0
        while True:
            img_frame = q.get()
            latency_ms = (
                dai.Clock.now() - img_frame.getTimestamp()
            ).total_seconds() * 1000
            samples.append(float(latency_ms))
            print_stats(samples, float(latency_ms))

            frames += 1
            if args.max_frames > 0 and frames >= args.max_frames:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())