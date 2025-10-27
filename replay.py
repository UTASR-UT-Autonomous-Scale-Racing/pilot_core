import argparse
import sqlite3
import numpy as np
import cv2
import time
from utils import load_model_from_config, overlay_depth_mask
import sys


def parse_args():
    parser = argparse.ArgumentParser(
            description="Run the data processing pipeline.",
            add_help=False
        )
    parser.add_argument('--segmentation_config', required=True, help="Path to segmentation model YAML config")
    parser.add_argument('--control_config', required=True, help="Path to control model YAML config")
    parser.add_argument('--db_path', type=str, help="Path to SQLite database file")
    parser.add_argument('-help', action='store_true', help="Show help message")

    args, _ = parser.parse_known_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    return args

args = parse_args()

# Load segmentation model
segmentation_model, _ = load_model_from_config(args.segmentation_config)

# Load control model
control_model, _ = load_model_from_config(args.control_config)

# Connect to the SQLite database
conn = sqlite3.connect(args.db_path)
cursor = conn.cursor()

# Query all rows from the state_vector_data table, ordered by id
cursor.execute("SELECT * FROM state_vector_data ORDER BY id ASC")

# Set expected frame dimensions
width = 640
height = 400
last_ts = None
delay = 0
paused = False

print("Starting playback...")
i=0
# Iterate over each row in the query result
while row := cursor.fetchone():
    time.sleep(delay)  # Sleep for visualization delay
    # Convert binary data to numpy arrays for depth and RGB frames
    depth_frame = np.frombuffer(row[10], dtype=np.float32)
    rgb_frame = np.frombuffer(row[11], dtype=np.uint8)

    # Reshape frames to expected dimensions
    rgb_frame = rgb_frame.reshape((height, width, 3))
    depth_frame = depth_frame.reshape((height, width))  # Adjust shape as needed

    segments = segmentation_model.get_mask(depth_frame, rgb_frame)

    normalized_steering, normalized_throttle = control_model.get_action(row[0:8], segments, float(row[12]), row[9], row[8])

    combined_frame = overlay_depth_mask(segments, rgb_frame)

    # Prepare overlay text with state information and timestamp delta
    overlay_text = [
        f"id: {row[0]}  w: {row[1]:.2f}  x: {row[2]:.2f}  y: {row[3]:.2f}  z: {row[4]:.2f}",
        f"ax: {row[5]:.5f} ay: {row[6]:.5f}  az: {row[7]:.5f}",
        f"throttle: {row[8]:.2f}  steering: {row[9]:.2f}  ts: {float(row[12]):.2f}",
    ]
    
    # Draw the overlay text on the combined frame
    for text in overlay_text:
        cv2.putText(
            combined_frame,
            text,
            (10, 30 + overlay_text.index(text) * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )
 
    # Display the combined frame in a window
    cv2.imshow("Combined Frame", combined_frame)
    cv2.waitKey(400)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Exit loop if 'q' is pressed
    if key == ord('q'):
        break
    
    # Toggle pause if 'p' is pressed
    if key == ord('p'):
        paused = not paused
        print(f"Playback {'paused' if paused else 'resumed'}")
    
    # If paused, wait for key press to continue or exit
    while paused:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = False
            print("Playback resumed")
            break
    
    # Exit if 'q' was pressed during pause
    if key == ord('q'):
        break
    last_ts = row[9]
    if i == 10:
        break
    i+=1

# Clean up: close OpenCV windows and database connection
cv2.destroyAllWindows()
conn.close()

control_model.save_plot()
