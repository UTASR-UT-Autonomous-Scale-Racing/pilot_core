#!/bin/bash

set -e  # Exit on error

# Power Mode (Performance)
echo "Setting performance mode..."
sudo nvpmodel -m 0
sudo jetson_clocks

# Run L4T-Nvidia toolkit container (BASE ONLY NOT REQUIRED)
echo "Launching Docker container..."
xhost +
sudo docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix nvcr.io/nvidia/l4t-jetpack:r35.3.1

# Function to update and reset repo
update_repo() {
    local dir=$1
    echo "Updating $dir..."
    cd "$dir"
    git fetch origin && git reset --hard origin/main && git clean -fd
    echo "$dir synced with origin/main."
}

# Retrieve updates from repos (Overwrites all uncommitted changes)
update_repo /home/forecr/Control-Model
update_repo /home/forecr/Data-Pipeline
update_repo /home/forecr/Jetson-Pilot

echo "All repositories are up to date."

# Start Pilot
echo "Starting Pilot..."
cd /home/forecr/Jetson-Pilot
./pilot_control.sh

