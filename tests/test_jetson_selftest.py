"""scripts/jetson_selftest.py on any machine: the hardware-free checks pass."""

from __future__ import annotations

import os
import subprocess
import sys

from tests.conftest import REPO_ROOT

SCRIPT = os.path.join(REPO_ROOT, "scripts", "jetson_selftest.py")


def _run(*args):
    return subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, timeout=60)


def test_hardware_free_checks_pass():
    r = _run("--only", "shm,buffers")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("PASS") == 2


def test_unknown_check_is_an_error():
    r = _run("--only", "warp_drive")
    assert r.returncode == 2 and "unknown check" in r.stderr
