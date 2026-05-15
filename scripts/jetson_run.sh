#!/usr/bin/env bash
# Native (no Docker) Jetson run: max perf, ff-pull + pinned submodules, then start.py.
# Refuses to run with uncommitted changes.
#   scripts/jetson_run.sh [start.py args]   # config.yaml unless --config is given
set -euo pipefail
cd "$(dirname "$0")/.."

# jetson_clocks doesn't survive a reboot, so redo it every run
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# pilot_core -> latest commit of the current branch (fast-forward only)
if ! git diff --quiet --ignore-submodules || ! git diff --cached --quiet --ignore-submodules; then
    echo "Error: pilot_core has uncommitted changes; commit or stash them first." >&2
    exit 1
fi
git pull --ff-only

# submodules -> pinned commits; the car needs all of them, ORB_SLAM3 included
git submodule sync --quiet
git submodule update --init --recursive
echo "All repositories are up to date."
git submodule status

# Python environment (docs/slam.md, README quick start)
if [ ! -f .venv/bin/activate ]; then
    echo "Error: .venv not found; create it first (README quick start)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# config.yaml unless the caller passes --config
args=("$@")
has_config=0
for a in "${args[@]}"; do
    case "$a" in --config|--config=*) has_config=1 ;; esac
done
[ "$has_config" -eq 1 ] || args=(--config config.yaml "${args[@]}")

# sudo for I2C/USB; -E keeps LD_LIBRARY_PATH, and the venv's python is used explicitly
echo "Starting Pilot..."
export LD_LIBRARY_PATH="$PWD/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
exec sudo -E "$(command -v python)" start.py "${args[@]}"
