"""
Config helpers used throughout the foundation v3 pipeline.
"""

from copy import deepcopy
from pathlib import Path


def deep_update(base, update):
    base = deepcopy(base)
    if update is None:
        return base
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def normalize_label_mapping(mapping):
    if mapping is None:
        return {}
    return {str(key): value for key, value in mapping.items()}


def nonnull_overrides(data, keys):
    overrides = {}
    for key in keys:
        value = data.get(key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def resolve_relative_path(value, base_dpath):
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dpath / path
    return path.resolve()


def resolve_package_paths(data, base_dpath):
    data = deepcopy(data)

    def _recurse(node):
        if isinstance(node, dict):
            resolved = {}
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    resolved[key] = _recurse(value)
                elif key.endswith(('_fpath', '_dpath')):
                    resolved[key] = None if value is None else str(resolve_relative_path(value, base_dpath))
                else:
                    resolved[key] = value
            return resolved
        if isinstance(node, list):
            return [_recurse(item) for item in node]
        return node

    return _recurse(data)


def ensuredir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
