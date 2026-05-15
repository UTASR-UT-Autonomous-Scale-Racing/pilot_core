#!/usr/bin/env bash
# Docker deploy on the Jetson: pull, perf mode, OAK-D udev rule, build and run (foreground).
#   scripts/start.sh                              # start.py --config config.yaml
#   scripts/start.sh --config X.yaml [--enable-slam ...]   # args are passed to start.py
set -euo pipefail
cd "$(dirname "$0")/.."

# pilot_core fast-forward, submodules at the commits pilot_core pins
git pull --ff-only
git submodule sync --quiet
git submodule update --init --recursive

# start.py args: config.yaml unless the caller passes --config
args=("$@")
CONFIG_FILE=""
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --config) CONFIG_FILE="${args[$((i + 1))]:-}" ;;
        --config=*) CONFIG_FILE="${args[$i]#--config=}" ;;
    esac
done
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="config.yaml"
    args=(--config "$CONFIG_FILE" "${args[@]}")
fi

# paths are relative to the repo root, which is the container's workdir
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE" >&2
    exit 1
fi
echo "-> Verified configuration: $CONFIG_FILE"
echo ""

# perf mode, skipped if this isn't a Jetson
if command -v nvpmodel &> /dev/null; then
    sudo nvpmodel -m 0
    sudo jetson_clocks
    echo "Set Jetson performance mode to MAX-N."
fi

# OAK-D (Movidius, vendor 03e7) without root; only written if missing
UDEV_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
UDEV_FILE=/etc/udev/rules.d/80-movidius.rules
if ! grep -qxF "$UDEV_RULE" "$UDEV_FILE" 2>/dev/null; then
    echo "$UDEV_RULE" | sudo tee "$UDEV_FILE" > /dev/null
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo "Installed OAK-D udev rule ($UDEV_FILE)."
fi
echo "-> Hardware settings applied."
echo ""

IMAGE_NAME="utasr-image"
sudo docker build -t "$IMAGE_NAME" -f Dockerfile .
echo "Built Docker image $IMAGE_NAME."
echo ""

# --privileged: OAK-D over USB; --device: Arduino I2C; --ipc=host: host /dev/shm
# for the ring buffers (Docker's default is 64 MB).
echo "Started container utasr-container (logs follow; q or Ctrl+C to stop)."
exec sudo docker run \
    --runtime nvidia \
    --rm \
    -it \
    --privileged \
    --net=host \
    --ipc=host \
    --device /dev/i2c-1 \
    --name utasr-container \
    "$IMAGE_NAME" \
    "${args[@]}"
