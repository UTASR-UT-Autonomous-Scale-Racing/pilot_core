import os
import importlib
import yaml
import numpy as np
import cv2


def load_model_from_config(yaml_path):
    """
    Load a model class from a YAML configuration file.
    Expects each model folder to contain config.yaml and model.py

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        tuple: (model instance, model class name)
    """
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

    return model, seg_class_name


def overlay_depth_mask(segments, rgb_frame):
    """
    Overlay the depth mask on the RGB frame.
    Args:
        segments (np.ndarray): The depth mask to overlay.
        rgb_frame (np.ndarray): The RGB frame to overlay on.
    Returns:
        np.ndarray: The combined frame with the depth mask overlay.
    """

    # Convert segments to a color map for visualization
    segments = cv2.applyColorMap(segments.astype(np.uint8), cv2.COLORMAP_HOT)

    # Blend the two frames together.
    coloured_segments = ((segments / 6.5) * 255).astype(np.uint8)
    combined_frame = cv2.addWeighted(coloured_segments, 1.0, rgb_frame, 1.0, 0)

    return combined_frame