#!/bin/bash
# NOTE: This script is intended to be run on a Jetson device only.
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
# Retrieve updates from repos (Overwrites all uncommitted changes)
cd pilot_ingest && git fetch origin && git reset --hard origin/main && cd ..
cd pilot_vision && git fetch origin && git reset --hard origin/main && cd ..
cd pilot_control && git fetch origin && git reset --hard origin/main && cd ..
echo "All repositories are up to date."

# Read config paths from config.yaml
CONFIG_FILE="config.yaml"
SEGMENTATION_CONFIG=$(grep '^segmentation_config_path:' "$CONFIG_FILE" | awk '{print $2}')
CONTROL_CONFIG=$(grep '^control_config_path:' "$CONFIG_FILE" | awk '{print $2}')

# Start Pilot
echo "Starting Pilot..."
sudo -E $(which python) _start.py --segmentation_config "$SEGMENTATION_CONFIG" --control_config "$CONTROL_CONFIG" "$@" 
