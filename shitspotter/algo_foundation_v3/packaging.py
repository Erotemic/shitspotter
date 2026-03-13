"""
Packaged model config helpers.
"""

from copy import deepcopy
from pathlib import Path

import yaml

from shitspotter.algo_foundation_v3 import model_registry
from shitspotter.algo_foundation_v3.config_utils import (
    deep_update,
    normalize_label_mapping,
    resolve_package_paths,
)


def load_package(package_fpath):
    package_fpath = Path(package_fpath).expanduser()
    data = yaml.safe_load(package_fpath.read_text()) or {}
    return data


def _expand_presets(data):
    data = deepcopy(data)

    detector = data.get('detector', None)
    if isinstance(detector, dict) and detector.get('preset', None):
        data['detector'] = deep_update(
            model_registry.resolve_detector_preset(detector['preset']),
            detector,
        )

    segmenter = data.get('segmenter', None)
    if isinstance(segmenter, dict) and segmenter.get('preset', None):
        data['segmenter'] = deep_update(
            model_registry.resolve_segmenter_preset(segmenter['preset']),
            segmenter,
        )

    baseline = data.get('baseline', None)
    if isinstance(baseline, dict) and baseline.get('preset', None):
        data['baseline'] = deep_update(
            model_registry.resolve_baseline_preset(baseline['preset']),
            baseline,
        )

    return data


def validate_package(data):
    backend = data.get('backend', None)
    if backend not in model_registry.BACKEND_DEFAULTS:
        raise KeyError(f'Unknown or missing backend={backend!r}')

    if backend == 'deimv2_sam2':
        if not isinstance(data.get('detector', None), dict):
            raise ValueError('deimv2_sam2 packages require a detector section')
        if not isinstance(data.get('segmenter', None), dict):
            raise ValueError('deimv2_sam2 packages require a segmenter section')
    elif backend == 'maskdino':
        if not isinstance(data.get('baseline', None), dict):
            raise ValueError('maskdino packages require a baseline section')

    data.setdefault('format_version', model_registry.PACKAGE_FORMAT_VERSION)
    data.setdefault('metadata', {})
    data.setdefault('postprocess', deepcopy(model_registry.DEFAULT_POSTPROCESS))
    data['label_mapping'] = normalize_label_mapping(data.get('label_mapping', {}))
    return data


def resolve_package(package_fpath=None, package_data=None, overrides=None):
    if package_fpath is None and package_data is None:
        raise ValueError('Specify package_fpath or package_data')

    raw = deepcopy(package_data) if package_data is not None else load_package(package_fpath)
    overrides = deepcopy(overrides) if overrides is not None else {}

    backend = overrides.get('backend', raw.get('backend', None))
    if backend is None:
        raise ValueError('Package backend must be specified')

    resolved = model_registry.default_package_for_backend(backend)
    resolved = deep_update(resolved, raw)
    resolved = _expand_presets(resolved)
    resolved = deep_update(resolved, overrides)
    resolved = _expand_presets(resolved)

    if package_fpath is not None:
        package_fpath = Path(package_fpath).expanduser().resolve()
        resolved = resolve_package_paths(resolved, package_fpath.parent)
        resolved['package_fpath'] = str(package_fpath)
    else:
        resolved['package_fpath'] = None

    return validate_package(resolved)


def dump_package(data, package_fpath):
    package_fpath = Path(package_fpath).expanduser()
    package_fpath.parent.mkdir(parents=True, exist_ok=True)
    package_fpath.write_text(yaml.safe_dump(data, sort_keys=False))
    return package_fpath


def build_package(
    backend,
    detector_preset=None,
    segmenter_preset=None,
    baseline_preset=None,
    detector_checkpoint_fpath=None,
    segmenter_checkpoint_fpath=None,
    baseline_checkpoint_fpath=None,
    metadata_name=None,
):
    package = model_registry.default_package_for_backend(backend)
    if detector_preset is not None:
        package.setdefault('detector', {})
        package['detector']['preset'] = detector_preset
    if segmenter_preset is not None:
        package.setdefault('segmenter', {})
        package['segmenter']['preset'] = segmenter_preset
    if baseline_preset is not None:
        package.setdefault('baseline', {})
        package['baseline']['preset'] = baseline_preset
    if detector_checkpoint_fpath is not None:
        package.setdefault('detector', {})
        package['detector']['checkpoint_fpath'] = str(detector_checkpoint_fpath)
    if segmenter_checkpoint_fpath is not None:
        package.setdefault('segmenter', {})
        package['segmenter']['checkpoint_fpath'] = str(segmenter_checkpoint_fpath)
    if baseline_checkpoint_fpath is not None:
        package.setdefault('baseline', {})
        package['baseline']['checkpoint_fpath'] = str(baseline_checkpoint_fpath)
    if metadata_name is not None:
        package.setdefault('metadata', {})
        package['metadata']['name'] = metadata_name
    return resolve_package(package_data=package)


def package_name(data):
    metadata = data.get('metadata', {})
    name = metadata.get('name', None)
    if name:
        return name
    package_fpath = data.get('package_fpath', None)
    if package_fpath:
        return Path(package_fpath).stem
    return data['backend']
