#!/usr/bin/env python3
"""Checks a machine before start.py drives the car: Jetson platform and power mode, CUDA, the
shared-memory room, the ring buffers, the ORB-SLAM3 bindings, the OAK-D on USB 3 and the
Arduino on I2C. Each check prints PASS, FAIL or SKIP; the exit code is 1 if any check failed.

    python scripts/jetson_selftest.py                       # every check
    python scripts/jetson_selftest.py --skip camera,i2c     # bench without the car's hardware
    python scripts/jetson_selftest.py --config config.yaml  # I2C address / bus from the config
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import uuid

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

MIN_SHM_MB = 512


class Skip(Exception):
    pass


def check_platform(cfg: dict) -> str:
    release = "/etc/nv_tegra_release"
    if not os.path.exists(release):
        raise Skip("not a Jetson (no /etc/nv_tegra_release)")
    with open(release) as f:
        line = f.readline().strip()
    if shutil.which("nvpmodel"):
        mode = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10).stdout.strip()
        mode = mode.splitlines()[0] if mode else "unknown"
    else:
        mode = "nvpmodel not found"
    return f"{line.split(',')[0]}; power mode: {mode}"


def check_cuda(cfg: dict) -> str:
    providers = []
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
    except ImportError:
        pass
    gpu = [p for p in providers if p in ("CUDAExecutionProvider", "TensorrtExecutionProvider")]
    devices = [d for d in ("/dev/nvhost-ctrl-gpu", "/dev/nvidia0") if os.path.exists(d)]
    if not devices:
        raise AssertionError("no GPU device node (/dev/nvhost-ctrl-gpu or /dev/nvidia0)")
    return f"GPU {devices[0]}; onnxruntime GPU providers: {', '.join(gpu) or 'none (CPU inference)'}"


def check_shm(cfg: dict) -> str:
    st = os.statvfs("/dev/shm")
    mb = st.f_bavail * st.f_frsize / 2**20
    if mb < MIN_SHM_MB:
        raise AssertionError(f"/dev/shm has {mb:.0f} MB free, need {MIN_SHM_MB} (Docker: --ipc=host)")
    return f"/dev/shm {mb:.0f} MB free"


def check_buffers(cfg: dict) -> str:
    import numpy as np

    from shared_buffer import FrameBuffer

    name = f"selftest_{uuid.uuid4().hex[:8]}"
    buf = FrameBuffer(slots=4, name=name, create=True)
    try:
        rgb = np.random.default_rng(0).integers(0, 255, (400, 640, 3), dtype=np.uint8)
        depth = np.full((400, 640), 1.5, np.float32)
        buf.write(rgb, depth, ts=1.0)
        header, rgb2, depth2 = buf.read_latest()
        if not (np.array_equal(rgb, rgb2) and np.array_equal(depth, depth2) and header.produced_ts == 1.0):
            raise AssertionError("ring buffer round trip changed the frame")
    finally:
        buf.unlink()
    return "640x400 RGB-D frame round trip"


def check_slam(cfg: dict) -> str:
    try:
        import orbslam3_python

        del orbslam3_python
    except ImportError as e:
        raise AssertionError(f"orbslam3_python does not import ({e}); build it (docs/slam.md)")
    vocab = os.path.join(REPO_ROOT, "ORB_SLAM3", "Vocabulary", "ORBvoc.txt")
    if not os.path.exists(vocab):
        raise AssertionError(f"{vocab} missing (tar xf ORBvoc.txt.tar.gz)")
    return "orbslam3_python imports; vocabulary present"


def check_camera(cfg: dict) -> str:
    try:
        import depthai as dai
    except ImportError:
        raise AssertionError("depthai not installed")
    devices = dai.Device.getAllAvailableDevices()
    if not devices:
        raise AssertionError("no OAK-D found")
    with dai.Device() as device:
        speed = device.getUsbSpeed()
        name = device.getDeviceName()
    if speed not in (dai.UsbSpeed.SUPER, dai.UsbSpeed.SUPER_PLUS):
        raise AssertionError(f"{name} on {speed.name}: 30 fps RGB-D needs USB 3 (docs/oakd.md)")
    return f"{name} on {speed.name}"


def check_i2c(cfg: dict) -> str:
    addr = cfg.get("i2c_address", 0x08)
    if addr is None:
        raise Skip("i2c_address is null in the config")
    bus = int(cfg.get("i2c_bus", 1))
    if not os.path.exists(f"/dev/i2c-{bus}"):
        raise AssertionError(f"/dev/i2c-{bus} missing")
    from smbus2 import SMBus

    if not os.access(f"/dev/i2c-{bus}", os.R_OK | os.W_OK):
        raise AssertionError(f"no access to /dev/i2c-{bus} (run as root, or add the user to the i2c group)")
    with SMBus(bus) as b:
        try:
            b.read_byte(int(addr))
        except OSError as e:
            raise AssertionError(f"no reply from 0x{int(addr):02x} on /dev/i2c-{bus} ({e})")
    return f"Arduino answers at 0x{int(addr):02x} on /dev/i2c-{bus}"


CHECKS = {
    "platform": check_platform,
    "cuda": check_cuda,
    "shm": check_shm,
    "buffers": check_buffers,
    "slam": check_slam,
    "camera": check_camera,
    "i2c": check_i2c,
}


def run(names: list[str], cfg: dict) -> int:
    failed = 0
    for name in names:
        try:
            msg = CHECKS[name](cfg)
            print(f"PASS  {name:9s} {msg}")
        except Skip as e:
            print(f"SKIP  {name:9s} {e}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name:9s} {e}")
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config.yaml"))
    ap.add_argument("--skip", default="", help=f"comma-separated checks to skip: {', '.join(CHECKS)}")
    ap.add_argument("--only", default="", help="comma-separated checks to run")
    a = ap.parse_args()
    import yaml

    with open(a.config) as f:
        cfg = yaml.safe_load(f) or {}
    names = [n for n in (a.only.split(",") if a.only else CHECKS) if n]
    unknown = [n for n in names + a.skip.split(",") if n and n not in CHECKS]
    if unknown:
        ap.error(f"unknown check(s): {', '.join(unknown)}")
    skip = set(filter(None, a.skip.split(",")))
    sys.exit(run([n for n in names if n not in skip], cfg))


if __name__ == "__main__":
    main()
