#!/bin/bash
# Jetson Docker deploy: update repos, check config, set Jetson performance
# mode + OAK-D udev rule, build the image from Dockerfile and run it.
#   scripts/start.sh                    # container runs start.py --config config.yaml
#   scripts/start.sh --config X.yaml    # args replace the image CMD (passed to start.py)
set -e
cd "$(dirname "$0")/.."

# ================================
# Update repositories (submodules move to their remote branch tips)
git pull
git submodule update --recursive --remote

# ================================
# Check that the config and its segmentation model config exist
CONFIG_FILE="config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

SEGMENTATION_CONFIG=$(grep '^segmentation_config_path:' "$CONFIG_FILE" | awk '{print $2}')

if [ ! -f "$SEGMENTATION_CONFIG" ]; then
    echo "Error: Segmentation config not found at $SEGMENTATION_CONFIG"
    exit 1
fi

echo "-> Successfully verified configuration files."
echo ""

# ================================
# Hardware setup

# Jetson max performance mode (skipped on non-Jetson hosts)
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo "Set Jetson performance mode to MAX-N."
fi

# udev rule so the OAK-D (Movidius, vendor 03e7) is accessible without root
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "-> Successfully applied hardware settings."
echo ""

# ================================
# Build Docker image
IMAGE_NAME="utasr-image"
sudo docker build -t ${IMAGE_NAME} -f Dockerfile .

echo "Successfully built Docker image."
echo ""

# ================================
# Run Docker container (no CPU / RAM / GPU limits)
#   --privileged  USB access (OAK-D)
#   --net=host    host networking
#   --device      I2C bus to the Arduino
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
