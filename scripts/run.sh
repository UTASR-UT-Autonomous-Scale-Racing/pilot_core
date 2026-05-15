#!/usr/bin/env bash
# Launch the pipeline from the repo root.
#   scripts/run.sh [start.py args]   # e.g. --config config_cpu.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .venv/bin/activate ] && source .venv/bin/activate
export LD_LIBRARY_PATH="$PWD/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

exec python3 start.py "$@"
