# Read config paths from config.yaml
CONFIG_FILE="config.yaml"
SEGMENTATION_CONFIG=$(grep '^segmentation_config_path:' "$CONFIG_FILE" | awk '{print $2}')
CONTROL_CONFIG=$(grep '^control_config_path:' "$CONFIG_FILE" | awk '{print $2}')

# Start Pilot
echo "Starting Pilot..."
sudo -E $(which python3) _start.py --segmentation_config "$SEGMENTATION_CONFIG" --control_config "$CONTROL_CONFIG" "$@" 
