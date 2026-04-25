"""YAML config -> model instance. The vision worker uses this."""

import importlib
import os

import yaml


def load_model_from_config(yaml_path):
    """Build the model described by yaml_path. Returns (model, class_name).

    model.py has to sit in the same directory as the yaml, and that directory
    has to be importable as a package from the project root. The yaml gives
    `class` (a name in model.py) and optionally `kwargs`. Relative string
    kwargs whose key ends in something path-like (_path, _file, _dir, ...)
    get resolved against the config directory.
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    seg_class_name = config.get("class")
    seg_kwargs = config.get("kwargs", {})

    # import as <package.path>.model relative to the project root
    config_dir = os.path.dirname(yaml_path)
    rel_path = os.path.relpath(config_dir, os.path.dirname(__file__))
    package_path = rel_path.replace(os.sep, '.')
    seg_module_path = f"{package_path}.model"

    _PATH_SUFFIXES = ("_path", "_file", "_dir", "_weights", "_vocab", "_config")
    for key, value in seg_kwargs.items():
        if (
            isinstance(value, str)
            and not os.path.isabs(value)
            and key.lower().endswith(_PATH_SUFFIXES)
        ):
            seg_kwargs[key] = os.path.join(config_dir, value)

    module = importlib.import_module(seg_module_path)
    ModelClass = getattr(module, seg_class_name)
    model = ModelClass(**seg_kwargs)

    return model, seg_class_name
