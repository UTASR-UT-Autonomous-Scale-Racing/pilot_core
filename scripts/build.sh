#!/usr/bin/env bash
# Build the legacy C++ pipeline (Jetson container only — see scripts/READMECPP.md).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
