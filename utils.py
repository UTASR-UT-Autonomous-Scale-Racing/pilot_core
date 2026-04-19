"""Load a model instance described by a YAML config (used by the vision worker)."""

import importlib
import os

import yaml


def load_model_from_config(yaml_path):
    """
    Instantiate the model described by ``yaml_path``.

    The config's directory must also contain ``model.py`` and be importable as
    a package relative to the project root. The YAML provides ``class`` (the
    class name in ``model.py``) and optional ``kwargs``; string kwargs whose
    key ends in a path-like suffix (``_path``, ``_file``, ``_dir``, ...) and
    that are relative are resolved against the config directory.

    Args:
        yaml_path (str): Path to the YAML configuration file.

    Returns:
        tuple: (model instance, model class name)
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    seg_class_name = config.get("class")
    seg_kwargs = config.get("kwargs", {})

    # model.py lives next to config.yaml; import it as <package.path>.model
    # relative to the project root.
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
