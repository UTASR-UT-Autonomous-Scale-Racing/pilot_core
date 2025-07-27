import argparse
import logging
import signal
import sys
import time
import cv2
import numpy
import yaml
import importlib
from tqdm import tqdm

from pilot_ingest import I2CDataProcessor, SegmentProcessor, StateVectorProcessor
from pilot_controller.control_output import Controller
from pilot_controller.models import *
from pilot_vision.models import *
from torch import Tensor

# Global variable used to store the function that stops all running processes
process_stop_func = None

def handle_signal(signum, frame):
    """
    Handle termination signals to stop all processes gracefully.
    """
    logging.info("Stopping all processes")
    if process_stop_func:
        process_stop_func()
    sys.exit(0)

# Register SIGINT (Ctrl+C) to trigger the signal handler
signal.signal(signal.SIGINT, handle_signal)
# Register SIGTERM (kill command) to trigger the signal handler
signal.signal(signal.SIGTERM, handle_signal)
# Register SIGTSTP (Ctrl+Z) to trigger the signal handler
signal.signal(signal.SIGTSTP, handle_signal)

def start_processes(i2c_processor, segment_processor, state_vector_processor):
    """
    Start all necessary processes for data acquisition and processing.
    """
    i2c_processor.start_reading()
    segment_processor.start_stream()
    state_vector_processor.start_processing()

def stop_processes(i2c_processor, segment_processor, state_vector_processor, use_oakd_imu):
    """
    Stop all running processes and clean up resources.
    """
    state_vector_processor.stop_processing()
    if not use_oakd_imu:
        i2c_processor.write(["1461", "1397"])
    i2c_processor.stop_reading()
    segment_processor.close_stream() # Reset steering and throttle to neutral values

def system_setup(i2c_address, i2c_bus, recording, segmentation_model):
    """
    Set up the system components including I2C data processor, segment processor, and state vector processor.
    """

    use_oakd_imu = i2c_address is None or i2c_bus is None

    # Configure processors
    i2c_processor = I2CDataProcessor(
        i2c_address=i2c_address,
        i2c_bus=i2c_bus
    )
    segment_processor = SegmentProcessor(
        use_oakd_imu=use_oakd_imu,  # True if using OAK-D IMU, False if using board IMU
        segmentation_model=segmentation_model
    )

    # Configure the state vector processor
    state_vector_processor = StateVectorProcessor(
        designated_I2C_processor=i2c_processor,
        designated_segment_processor=segment_processor,
        recording=recording
    )
    state_vector_processor.pause_recording()

    # Start all underlying processes
    start_processes(i2c_processor, segment_processor, state_vector_processor)
    time.sleep(3)  # Allow time for connections to be established

    # Validate the structure of the state vector
    test_output = state_vector_processor.get_state_vector()
    while not test_output:
        test_output = state_vector_processor.get_state_vector()

    logging.info("Setup complete")
    try:
        assert test_output.sensors is not None, "Sensors data is None"
        assert test_output.frames is not None, "Frames data is None"
        assert test_output.timestamp is not None, "Timestamp is None"
    except AssertionError as e:
        logging.error(str(e))
        stop_processes(i2c_processor, segment_processor, state_vector_processor, use_oakd_imu)
        sys.exit(1)

    # Return the initialized processors
    return i2c_processor, segment_processor, state_vector_processor

def parse_args():
    """
    Parse command-line arguments for the data processing pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run the data processing pipeline.",
        add_help=False
    )
    parser.add_argument('--segmentation_config', required=True, help="Path to segmentation model YAML config")
    parser.add_argument('--control_config', required=True, help="Path to control model YAML config")
    parser.add_argument('-k', action='store_true', help="Use keyboard controller over program controls")
    parser.add_argument('-oakd_imu', action='store_true', help="Use OAK-D IMU over board IMU")
    parser.add_argument('-r', action='store_true', help="Record all data")
    parser.add_argument('-d', action='store_true', help="Debug mode, show all data (to be used with a desktop interface)")
    parser.add_argument('-help', action='store_true', help="Show help message")
    parser.add_argument('-p', action='store_true', help="Passthrough mode: use IMU-reported steering/throttle instead of model/controller output")

    args, _ = parser.parse_known_args()

    if args.help:
        parser.print_help()
        sys.exit(0)

    return args

def load_model_from_config(yaml_path):
    """
    Load a model class from a YAML configuration file.
    Expects each model folder to contain config.yaml and model.py
    """
    import os
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    seg_class_name = config.get("class")
    seg_kwargs = config.get("kwargs", {})

    # Build module path: assume model.py is in the same directory as config.yaml
    config_dir = os.path.dirname(yaml_path)
    # Get relative path from project root
    rel_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    # Convert filesystem path to Python package path and append .model
    package_path = rel_path.replace(os.sep, '.')
    seg_module_path = f"{package_path}.model"

    # Resolve relative paths in kwargs based on the config file's directory
    for key, value in seg_kwargs.items():
        if isinstance(value, str) and not os.path.isabs(value):
            # Convert relative path to absolute path based on config file location
            seg_kwargs[key] = os.path.join(config_dir, value)

    module = importlib.import_module(seg_module_path)
    ModelClass = getattr(module, seg_class_name)
    model = ModelClass(**seg_kwargs)
    logging.info(f"Loaded {seg_class_name} from {seg_module_path}")
    return model

def main():
    # Retrieve command-line arguments
    args = parse_args()

    # Set up logging level.
    logging.basicConfig(level=logging.DEBUG if args.d else logging.INFO)
    if args.r and not args.k:
        logging.warning("Recording is enabled but keyboard control is not active. You are recording autonomous driving data.")

    # Configure keyboard control and device options
    keyboard_controlled = True if args.k else None
    use_oakd_imu = args.oakd_imu
    debug_mode = args.d
    recording = args.r
    passthrough_mode = args.p

    # Action handler for controlling the vehicle
    control_handler = Controller()

    # Load segmentation model config and class
    segmentation_model = load_model_from_config(args.segmentation_config)

    if not keyboard_controlled:
        # Load control model config and class
        control_model = load_model_from_config(args.control_config)

    # Device-specific constants
    if use_oakd_imu:
        I2C_SLAVE_ADDR = None
        I2C_BUS = None
    else:
        I2C_SLAVE_ADDR = 0x08
        I2C_BUS = 1

    # Initialize system components
    i2c_processor, segment_processor, state_vector_processor = system_setup(
        I2C_SLAVE_ADDR, I2C_BUS, recording, segmentation_model
    )

    # Set the global stop function for the signal handler
    global process_stop_func
    process_stop_func = lambda: stop_processes(i2c_processor, segment_processor, state_vector_processor, use_oakd_imu)

    logging_string = '\n'

    # Main processing loop
    i = 0

    state_vector_processor.resume_recording()
  
    with tqdm(total=None, desc="Processing frames") as frame_counter:
        
        while True:
            state_vector = state_vector_processor.get_state_vector()
            if not state_vector:
                time.sleep(0.000001)  # Shorter sleep for faster polling
                continue
            
            sensors = state_vector.sensors.to_array()
            segments = None
            # Control logic
            if passthrough_mode:
                control_handler.throttle = sensors[-2]
                control_handler.steering = sensors[-1]
            elif keyboard_controlled:
                control_handler.get_keyboard_output()
            else:
                # Get the segment depth mask
                segments = segment_processor.segment_model.get_mask(*state_vector.frames.to_array())
                normalized_steering, normalized_throttle = control_model.get_action(sensors, segments, state_vector.timestamp)
                #TODO: Remove this line when the model is ready
                normalized_throttle, normalized_steering = segment_processor.throttle_value, segment_processor.steering_value
                control_handler.get_normalized_action_output(normalized_steering, normalized_throttle)
                control_handler.throttle = sensors[-2]
                control_handler.steering = sensors[-1]

            if not use_oakd_imu:
                i2c_processor.write([control_handler.steering, control_handler.throttle])

            # Debug mode: log sensor data and show processed frame
            if debug_mode:
                logging_string += f"Quaternion: w={sensors[0]}, x={sensors[1]}, y={sensors[2]}, z={sensors[3]}\n"
                logging_string += f"Steering: {control_handler.steering}, Throttle: {control_handler.throttle }\n"
                curr_time = time.time()
                delta_state_ts = curr_time - state_vector.timestamp
                delta_frames_ts = curr_time - state_vector.frames.timestamp
                delta_sensors_ts = curr_time - state_vector.sensors.timestamp
                logging_string += f"Delta State TS: {delta_state_ts:.4f} seconds\n"
                logging_string += f"Delta Frames TS: {delta_frames_ts:.4f} seconds\n"
                logging_string += f"Delta Sensors TS: {delta_sensors_ts:.4f} seconds\n"      

                if segments is not None:
                    if not isinstance(segments, numpy.ndarray):
                        segments = Tensor.cpu(segments).numpy()
                        segments = cv2.applyColorMap(segments, cv2.COLORMAP_HOT)

                    # Blend the two frames together.
                    combined_frame = cv2.addWeighted(segments, 0.5, state_vector.frames.to_array()[1], 0.5, 0)
                    cv2.imshow("Masked Disparity Frame", combined_frame)

                    # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
                logging.debug(logging_string)
            
            frame_counter.update(1)

            logging_string = ''
            
            i += 1
            if i % 100 == 0:
                control_model.save_plot()

    stop_processes(i2c_processor, segment_processor, state_vector_processor)
    sys.exit(0)

if __name__ == "__main__":
    main()
