#!/bin/bash
# NOTE: This script is intended to be run on a Jetson device only. Do NOT run on your local machine.
set -e  # Exit on error

# Power Mode (Performance)
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# Run L4T-Nvidia toolkit container (BASE ONLY NOT REQUIRED)
# echo "Launching Docker container..."
# xhost +
# sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-jetpack:r35.3.1

# Function to update and reset repo
update_repo() {
    local dir=$1
    echo "Updating $dir..."
    cd "$dir"
    git fetch origin && git reset --hard origin/main && git clean -fd
    echo "$dir synced with origin/main."
}

# Retrieve updates from repos (Overwrites all uncommitted changes)
update_repo "$(pwd)/pilot_ingest"
update_repo "$(pwd)/pilot_vision"
update_repo "$(pwd)/pilot_control"

echo "All repositories are up to date."

# Start Pilot
echo "Starting Pilot..."
sudo -E $(which python) _start.py --segmentation_config pilot_vision/models/default/default.yaml --control_config pilot_controller/models/default/default.yaml -k -oakd_imu -d

