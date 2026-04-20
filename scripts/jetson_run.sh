#!/bin/bash
# Run the pipeline natively on the Jetson (no Docker): set max performance,
# hard-reset pilot_core + submodules to their remote branches, start start.py.
#   scripts/jetson_run.sh [start.py args]   # config.yaml unless --config is given
# WARNING: discards all uncommitted changes in pilot_core, pilot_vision and firmware.
set -e
cd "$(dirname "$0")/.."

# Max performance mode (jetson_clocks does not persist across reboots)
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# Update pilot_core to origin/<current branch>, discarding local changes
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch origin
git reset --hard origin/$CURRENT_BRANCH

# Same for each submodule, on its own current branch
for repo in pilot_vision firmware; do
    cd $repo
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    git fetch origin
    git reset --hard origin/$BRANCH
    cd ..
done

echo "All repositories are up to date."

# Start Pilot (sudo for I2C / USB access; -E keeps the venv + LD_LIBRARY_PATH)
echo "Starting Pilot..."
export LD_LIBRARY_PATH="$PWD/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"
sudo -E "$(which python3)" start.py --config config.yaml "$@"
