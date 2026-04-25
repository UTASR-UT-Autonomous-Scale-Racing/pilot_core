#!/bin/bash
# Docker deploy on the Jetson. Pulls, sanity-checks the config, sets perf
# mode and the OAK-D udev rule, then builds and runs the image.
#   scripts/start.sh                    # container runs start.py --config config.yaml
#   scripts/start.sh --config X.yaml    # args replace the image CMD (passed to start.py)
set -e
cd "$(dirname "$0")/.."

# NB: submodules go to their remote branch tips, not the pinned commits
git pull
git submodule update --recursive --remote

# make sure the config and the segmentation config it points to exist
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

# hardware
# perf mode, skipped if this isn't a Jetson
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo "Set Jetson performance mode to MAX-N."
fi

# OAK-D (Movidius, vendor 03e7) without root
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "-> Successfully applied hardware settings."
echo ""

IMAGE_NAME="utasr-image"
sudo docker build -t ${IMAGE_NAME} -f Dockerfile .

echo "Successfully built Docker image."
echo ""

# no CPU/RAM/GPU limits. --privileged is for the OAK-D over USB, the
# --device is the I2C bus to the Arduino
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
