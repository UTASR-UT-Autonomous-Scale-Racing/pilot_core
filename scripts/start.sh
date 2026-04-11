#!/bin/bash

# Exits on error
set -e
cd "$(dirname "$0")/.."

# ================================
# Updates repositories

git pull
git submodule update --recursive --remote

# ================================
# Checks configuration
CONFIG_FILE="config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

SEGMENTATION_CONFIG=$(grep '^segmentation_config_path:' "$CONFIG_FILE" | awk '{print $2}')
CONTROL_CONFIG=$(grep '^control_config_path:' "$CONFIG_FILE" | awk '{print $2}')

if [ ! -f "$SEGMENTATION_CONFIG" ]; then
    echo "Error: Segmentation config not found at $SEGMENTATION_CONFIG"
    exit 1
fi

if [ ! -f "$CONTROL_CONFIG" ]; then
    echo "Error: Control config not found at $CONTROL_CONFIG"
    exit 1
fi


echo "-> Successfully verified configuration files."
echo ""

# ================================
# Hardware Setup

# sets Jetson performance mode
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo "Set Jetson performance mode to MAX-N."
fi

echo 'Set SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "-> Successfully applied hardware settings."
echo ""

# ================================
# Builds Docker image
IMAGE_NAME="utasr-image"
sudo docker build -t ${IMAGE_NAME} -f Dockerfile .

echo "Successfully built Docker image."
echo ""

# ================================
# Runs Docker container (Review Required)
# previliged => usb access
# net=host => network access;
# device => i2c access;

# By Default not limit CPU, Ram and GPU usage
sudo docker run \
    --runtime nvidia \
    --rm \
    -it \
    --privileged \
    --net=host \
    --device /dev/i2c-1 \
    -e DISPLAY=$DISPLAY \
    --name utasr-container \
    ${IMAGE_NAME} \
    "$@"

echo ""
echo "Successfully started Docker container."

echo ""
echo "start.sh completed."


