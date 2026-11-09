#!/usr/bin/env bash
# Docker deploy on the Jetson: the release tag, perf mode, OAK-D udev rule, build, the
# self-test in the container, then run (foreground).
#   scripts/start.sh                              # start.py --config config.yaml
#   scripts/start.sh --config X.yaml [--enable-slam ...]   # args are passed to start.py
#   PILOT_REF=<ref> / SKIP_SELFTEST=1             # as for jetson_run.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# pilot_core at the release tag (or PILOT_REF), submodules at its pinned commits
# shellcheck source=deploy_ref.sh
source scripts/deploy_ref.sh
update_repo

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
RUN_FLAGS=(--runtime nvidia --rm --privileged --net=host --ipc=host --device /dev/i2c-1)
if [ "${SKIP_SELFTEST:-0}" != 1 ]; then
    sudo docker run "${RUN_FLAGS[@]}" --entrypoint python3 "$IMAGE_NAME" \
        scripts/jetson_selftest.py --config "$CONFIG_FILE" || {
        echo "Self-test failed; fix it or run with SKIP_SELFTEST=1." >&2
        exit 1
    }
fi
echo "Started container utasr-container (logs follow; q or Ctrl+C to stop)."
exec sudo docker run "${RUN_FLAGS[@]}" -it --name utasr-container "$IMAGE_NAME" "${args[@]}"
