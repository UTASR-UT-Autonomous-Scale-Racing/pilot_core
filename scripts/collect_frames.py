#!/usr/bin/env python3
"""
scripts/collect_frames.py — Save RGB frames from the simulator ZMQ stream to disk.

Connects to the same ZMQ sensor port as ZMQSensorWorker (default: 5555) and
writes each sampled ``front_camera`` frame as frame_NNNNNN.jpg (the JPEG bytes
are written as received, no re-encoding).  Run this alongside the AutoDRIVE
simulator bridge; the pilot_core pipeline does NOT need to be running.

Usage (from the repo root; --output is relative to the current directory):
    python scripts/collect_frames.py                         # defaults → ./dataset/rgb
    python scripts/collect_frames.py --output ./dataset/rgb  # custom output dir
    python scripts/collect_frames.py --every 5 --max 2000    # every 5th frame, stop at 2000
    python scripts/collect_frames.py --host localhost --port 5555
"""

import argparse
import base64
import json
import os
import sys
import time

try:
    import zmq
except ImportError:
    sys.exit("pyzmq is required: pip install pyzmq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect RGB frames from simulator ZMQ stream")
    parser.add_argument("--host",   default="localhost",    help="ZMQ publisher host (default: localhost)")
    parser.add_argument("--port",   type=int, default=5555, help="ZMQ sensor port (default: 5555)")
    parser.add_argument("--output", default="./dataset/rgb",help="Output directory for JPEG files (default: ./dataset/rgb)")
    parser.add_argument("--every",  type=int, default=3,    help="Save every N-th frame (default: 3)")
    parser.add_argument("--max",    type=int, default=0,    help="Stop after saving this many frames (0 = unlimited)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    context = zmq.Context()
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 1000)   # 1s timeout so Ctrl+C is responsive
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.setsockopt(zmq.CONFLATE, 1)       # keep only the latest message
    sock.connect(f"tcp://{args.host}:{args.port}")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")

    print(f"Connected to tcp://{args.host}:{args.port}")
    print(f"Saving every {args.every}th frame → {args.output}")
    if args.max:
        print(f"Will stop after {args.max} saved frames")
    print("Press Ctrl+C to stop.\n")

    frames_seen  = 0   # total frames received from ZMQ
    frames_saved = 0   # frames written to disk
    start_time   = time.time()

    try:
        while True:
            # Stop when we've hit the user-specified cap
            if args.max and frames_saved >= args.max:
                print(f"\nReached {args.max} saved frames — done.")
                break

            try:
                raw = sock.recv_string()
            except zmq.Again:
                # No message within timeout — simulator may not be running yet
                print("Waiting for simulator...", end="\r")
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            fc = data.get("front_camera")
            if not fc:
                continue

            frames_seen += 1

            # Decimation: only save every N-th frame
            if frames_seen % args.every != 0:
                continue

            # The payload is already JPEG: decode base64 and write the bytes as-is
            try:
                jpeg_bytes = base64.b64decode(fc["data"])
            except Exception as e:
                print(f"Warning: failed to decode frame {frames_seen}: {e}")
                continue

            filename = f"frame_{frames_saved:06d}.jpg"
            filepath = os.path.join(args.output, filename)
            with open(filepath, "wb") as f:
                f.write(jpeg_bytes)

            frames_saved += 1

            # Progress report every 50 saved frames
            if frames_saved % 50 == 0:
                elapsed  = time.time() - start_time
                fps_recv = frames_seen  / elapsed
                fps_save = frames_saved / elapsed
                print(f"  Saved {frames_saved} frames | "
                      f"recv {fps_recv:.1f} fps | "
                      f"saved {fps_save:.1f} fps")

    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.time() - start_time
        print(f"\nDone. {frames_saved} frames saved to '{args.output}' "
              f"in {elapsed:.1f}s ({frames_saved/elapsed:.1f} fps saved)")
        sock.close()
        context.term()


if __name__ == "__main__":
    main()
