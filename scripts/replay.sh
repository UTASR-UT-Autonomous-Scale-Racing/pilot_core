#!/usr/bin/env bash
# Replay recorded data with the legacy C++ replay binary.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Starting replay ..."
exec ./build/bin/replay "$@"
