"""CPU budget for the benchmark runners: import before cv2 / numpy / the SLAM modules.

Containers often report every host core (nproc 255 on the vast.ai box) but cap the CPU
time (cgroup cpu.max: 30.6 CPUs there). Thread pools sized to nproc then exceed the cap
and the kernel freezes the whole process until the next 100 ms period, which shows up as
40-60 ms tracking calls. This pins the process to --cpus cores (default: the cgroup cap,
else all) and sizes the OpenMP / BLAS pools to match. --cpus 8 also approximates a
Jetson Orin's core count.
"""

from __future__ import annotations

import os
import sys


def cgroup_cpus() -> int | None:
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        return None if quota == "max" else max(1, int(int(quota) // int(period)))
    except (OSError, ValueError):
        return None


def _requested() -> int | None:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--cpus" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--cpus="):
            return int(a.split("=", 1)[1])
    return None


def apply() -> int:
    avail = sorted(os.sched_getaffinity(0))
    n = _requested() or cgroup_cpus() or len(avail)
    n = max(1, min(n, len(avail)))
    os.sched_setaffinity(0, avail[:n])
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
    return n


CPUS = apply()
