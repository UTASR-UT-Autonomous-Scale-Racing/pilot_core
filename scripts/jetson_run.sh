#!/bin/bash
# Native (no Docker) run on the Jetson. Sets max perf, hard-resets pilot_core
# and the submodules to their remote branches, then starts start.py.
#   scripts/jetson_run.sh [start.py args]   # config.yaml unless --config is given
# careful: this throws away any uncommitted changes in pilot_core, pilot_vision
# and firmware
set -e
cd "$(dirname "$0")/.."

# jetson_clocks doesn't survive a reboot, so redo it every run
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# pilot_core -> origin/<current branch>
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch origin
git reset --hard origin/$CURRENT_BRANCH

# same for each submodule, on whatever branch it's on
for repo in pilot_vision firmware; do
    cd $repo
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git fetch origin
    git reset --hard origin/$BRANCH
    cd ..
done

echo "All repositories are up to date."

# sudo for I2C/USB; -E so we keep the venv and LD_LIBRARY_PATH
echo "Starting Pilot..."
export LD_LIBRARY_PATH="$PWD/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
sudo -E "$(which python3)" start.py --config config.yaml "$@"
