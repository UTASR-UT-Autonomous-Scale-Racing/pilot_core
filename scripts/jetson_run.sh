#!/usr/bin/env bash
# Native (no Docker) Jetson run: max perf, the release tag + pinned submodules, the
# self-test, then start.py. Refuses to run with uncommitted changes.
#   scripts/jetson_run.sh [start.py args]   # config.yaml unless --config is given
#   PILOT_REF=<ref> scripts/jetson_run.sh   # another tag, branch or commit
#   SKIP_SELFTEST=1 scripts/jetson_run.sh   # skip scripts/jetson_selftest.py
set -euo pipefail
cd "$(dirname "$0")/.."

# jetson_clocks doesn't survive a reboot, so redo it every run
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# pilot_core at the release tag (or PILOT_REF), submodules at its pinned commits
# shellcheck source=deploy_ref.sh
source scripts/deploy_ref.sh
update_repo

# Python environment (docs/slam.md, README quick start)
if [ ! -f .venv/bin/activate ]; then
    echo "Error: .venv not found; create it first (README quick start)." >&2
    exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# config.yaml unless the caller passes --config
args=("$@")
CONFIG_FILE=$(config_from_args "${args[@]}")
case " ${args[*]} " in *" --config"*) ;; *) args=(--config "$CONFIG_FILE" "${args[@]}") ;; esac

# root for I2C and USB. sudo drops LD_LIBRARY_PATH; the bindings find ORB-SLAM3 by their RPATH
PY="$(command -v python)"
if [ "${SKIP_SELFTEST:-0}" != 1 ]; then
    sudo "$PY" scripts/jetson_selftest.py --config "$CONFIG_FILE" || {
        echo "Self-test failed; fix it or run with SKIP_SELFTEST=1." >&2
        exit 1
    }
fi
echo "Starting Pilot..."
exec sudo "$PY" start.py "${args[@]}"
