#!/bin/bash
# NOTE: This script is intended to be run on a Jetson device only.
set -e  # Exit on error
cd "$(dirname "$0")/.."

# Power Mode (Performance)
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks # Disables on reboot

# Run L4T-Nvidia toolkit container (BASE ONLY NOT REQUIRED)
# echo "Launching Docker container..."
# xhost +
# sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-jetpack:r35.3.1

# Function to update and reset repo
# Retrieve updates from repos (Overwrites all uncommitted changes)
# Update and reset main repo
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git fetch origin
git reset --hard origin/$CURRENT_BRANCH

# Update and reset submodules/repos
for repo in pilot_ingest pilot_vision pilot_controller firmware; do
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
