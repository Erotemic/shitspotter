"""
Preset definitions for the foundation v3 pipeline.
"""

from copy import deepcopy


PACKAGE_FORMAT_VERSION = 1
DEFAULT_CATEGORY_NAME = 'poop'
DEFAULT_LABEL_MAPPING = {
    '0': DEFAULT_CATEGORY_NAME,
    '1': DEFAULT_CATEGORY_NAME,
    DEFAULT_CATEGORY_NAME: DEFAULT_CATEGORY_NAME,
}

DEFAULT_POSTPROCESS = {
    'score_thresh': 0.20,
    'nms_thresh': 0.50,
    'crop_padding': 32,
    'polygon_simplify': 2.0,
    'min_component_area': 32,
    'keep_largest_component': True,
}

REPO_ENV_VARS = {
    'deimv2': 'SHITSPOTTER_DEIMV2_REPO_DPATH',
    'sam2': 'SHITSPOTTER_SAM2_REPO_DPATH',
    'maskdino': 'SHITSPOTTER_MASKDINO_REPO_DPATH',
}

DETECTOR_PRESETS = {
    'deimv2_m': {
        'backend': 'deimv2',
        'variant': 'deimv2_m',
        'repo_envvar': REPO_ENV_VARS['deimv2'],
        'config_relpath': 'configs/deimv2/deimv2_dinov3_m_coco.yml',
        'input_size': [640, 640],
        'device': 'cuda:0',
    },
    'deimv2_s': {
        'backend': 'deimv2',
        'variant': 'deimv2_s',
        'repo_envvar': REPO_ENV_VARS['deimv2'],
        'config_relpath': 'configs/deimv2/deimv2_dinov3_s_coco.yml',
        'input_size': [640, 640],
        'device': 'cuda:0',
    },
}

SEGMENTER_PRESETS = {
    'sam2.1_hiera_base_plus': {
        'backend': 'sam2',
        'variant': 'sam2.1_hiera_base_plus',
        'repo_envvar': REPO_ENV_VARS['sam2'],
        'config_relpath': 'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
        'training_template_relpath': 'sam2/configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml',
        'hf_model_id': 'facebook/sam2.1-hiera-base-plus',
        'device': 'cuda:0',
        'mask_threshold': 0.0,
    },
    'sam2.1_hiera_large': {
        'backend': 'sam2',
        'variant': 'sam2.1_hiera_large',
        'repo_envvar': REPO_ENV_VARS['sam2'],
        'config_relpath': 'sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
        'training_template_relpath': None,
        'hf_model_id': 'facebook/sam2.1-hiera-large',
        'device': 'cuda:0',
        'mask_threshold': 0.0,
    },
}

BASELINE_PRESETS = {
    'maskdino_r50': {
        'backend': 'maskdino',
        'variant': 'maskdino_r50',
        'repo_envvar': REPO_ENV_VARS['maskdino'],
        'config_relpath': 'configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml',
        'device': 'cuda:0',
        'num_classes': 1,
    },
}

BACKEND_DEFAULTS = {
    'deimv2_sam2': {
        'format_version': PACKAGE_FORMAT_VERSION,
        'backend': 'deimv2_sam2',
        'detector': {'preset': 'deimv2_m'},
        'segmenter': {'preset': 'sam2.1_hiera_base_plus'},
        'postprocess': deepcopy(DEFAULT_POSTPROCESS),
        'label_mapping': deepcopy(DEFAULT_LABEL_MAPPING),
        'metadata': {
            'name': 'deimv2_sam2_default',
            'family': 'foundation_detseg_v3',
        },
    },
    'maskdino': {
        'format_version': PACKAGE_FORMAT_VERSION,
        'backend': 'maskdino',
        'baseline': {'preset': 'maskdino_r50'},
        'postprocess': deepcopy(DEFAULT_POSTPROCESS),
        'label_mapping': deepcopy(DEFAULT_LABEL_MAPPING),
        'metadata': {
            'name': 'maskdino_r50_default',
            'family': 'foundation_detseg_v3',
        },
    },
}


def default_package_for_backend(backend):
    if backend not in BACKEND_DEFAULTS:
        raise KeyError(f'Unknown backend={backend!r}')
    return deepcopy(BACKEND_DEFAULTS[backend])


def resolve_detector_preset(name):
    return deepcopy(DETECTOR_PRESETS[name])


def resolve_segmenter_preset(name):
    return deepcopy(SEGMENTER_PRESETS[name])


def resolve_baseline_preset(name):
    return deepcopy(BASELINE_PRESETS[name])
