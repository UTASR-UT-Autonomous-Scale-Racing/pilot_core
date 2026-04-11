#!/usr/bin/env bash
# Launch the legacy C++ pipeline (build with scripts/build.sh first).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG_FILE="config.yaml"
SEGMENTATION_CONFIG=$(grep '^segmentation_config_path:' "$CONFIG_FILE" | awk '{print $2}')
CONTROL_CONFIG=$(grep '^control_config_path:' "$CONFIG_FILE" | awk '{print $2}')

echo "Starting pilot (C++) ..."
exec ./build/bin/oakd_orbslam3 \
    --segmentation-config "$SEGMENTATION_CONFIG" \
    --control-config "$CONTROL_CONFIG" \
    ORB_SLAM3/Vocabulary/ORBvoc.txt oak_d_orbslam_settings.yaml "$@"
