"""
Thin wrappers around the DEIMv2 repo for detector training and inference.
"""

import importlib
import subprocess
import sys
from pathlib import Path

import yaml

from shitspotter.algo_foundation_v3.config_utils import deep_update
from shitspotter.algo_foundation_v3.datasets import prepare_detector_training_data


def resolve_repo_dpath(detector_cfg):
    import os

    envvar = detector_cfg.get('repo_envvar', 'SHITSPOTTER_DEIMV2_REPO_DPATH')
    repo = detector_cfg.get('repo_dpath', os.environ.get(envvar, None))
    if repo is None:
        raise EnvironmentError(
            f'DEIMv2 repo path is not configured. Set {envvar} or detector.repo_dpath.'
        )
    return Path(repo).expanduser().resolve()


def resolve_model_config_fpath(detector_cfg):
    repo_dpath = resolve_repo_dpath(detector_cfg)
    config_fpath = detector_cfg.get('config_fpath', None)
    if config_fpath is not None:
        return Path(config_fpath).expanduser().resolve()
    config_relpath = detector_cfg.get('config_relpath', None)
    if config_relpath is None:
        raise KeyError('Detector package requires config_fpath or config_relpath')
    return (repo_dpath / config_relpath).resolve()


def _ensure_repo_on_path(repo_dpath):
    repo_str = str(repo_dpath)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _load_checkpoint_state(checkpoint_fpath):
    import torch

    checkpoint = torch.load(checkpoint_fpath, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint and 'module' in checkpoint['ema']:
            return checkpoint['ema']['module']
        if 'model' in checkpoint:
            return checkpoint['model']
    return checkpoint


def _infer_num_classes_from_state(state_dict):
    weight = state_dict.get('decoder.enc_score_head.weight', None)
    if weight is not None and getattr(weight, 'ndim', None) == 2:
        return int(weight.shape[0])

    weight = state_dict.get('decoder.denoising_class_embed.weight', None)
    if weight is not None and getattr(weight, 'ndim', None) == 2 and weight.shape[0] > 0:
        return int(weight.shape[0] - 1)

    return None


class DEIMv2Predictor:
    def __init__(self, detector_cfg):
        self.detector_cfg = detector_cfg
        self.model = None
        self.device = detector_cfg.get('device', 'cuda:0')
        self.input_size = tuple(detector_cfg.get('input_size', [640, 640]))

    def _lazy_init(self):
        if self.model is not None:
            return
        import torch
        import torch.nn as nn
        import torchvision.transforms as T

        repo_dpath = resolve_repo_dpath(self.detector_cfg)
        _ensure_repo_on_path(repo_dpath)
        YAMLConfig = importlib.import_module('engine.core').YAMLConfig

        config_fpath = resolve_model_config_fpath(self.detector_cfg)
        checkpoint_fpath = self.detector_cfg.get('checkpoint_fpath', None)
        if checkpoint_fpath is None:
            raise KeyError('Detector package requires detector.checkpoint_fpath')

        checkpoint_state = _load_checkpoint_state(checkpoint_fpath)
        num_classes = self.detector_cfg.get('num_classes', None)
        if num_classes is None:
            num_classes = _infer_num_classes_from_state(checkpoint_state)

        yaml_kwargs = {'resume': str(checkpoint_fpath)}
        if num_classes is not None:
            yaml_kwargs['num_classes'] = int(num_classes)

        cfg = YAMLConfig(str(config_fpath), **yaml_kwargs)
        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        cfg.model.load_state_dict(checkpoint_state)

        class Model(nn.Module):
            def __init__(self, cfg_):
                super().__init__()
                self.model = cfg_.model.deploy()
                self.postprocessor = cfg_.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
        ])
        self.model = Model(cfg).to(self.device).eval()

    def predict_image_records(self, image):
        import torch
        from PIL import Image

        self._lazy_init()

        image_pil = Image.fromarray(image)
        width, height = image_pil.size
        orig_size = torch.tensor([[width, height]], device=self.device)
        im_data = self.transforms(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            labels, boxes, scores = self.model(im_data, orig_size)
        labels = labels[0].detach().cpu().tolist()
        boxes = boxes[0].detach().cpu().tolist()
        scores = scores[0].detach().cpu().tolist()
        return [
            {'label': label, 'bbox_ltrb': box, 'score': score}
            for label, box, score in zip(labels, boxes, scores)
        ]


def build_training_config(detector_cfg, train_json_fpath, vali_json_fpath, output_dpath,
                          overrides=None):
    overrides = overrides or {}
    base_config_fpath = resolve_model_config_fpath(detector_cfg)
    output_dpath = Path(output_dpath)
    generated_dpath = output_dpath / 'generated_configs'
    generated_dpath.mkdir(parents=True, exist_ok=True)

    dataset_config = {
        'task': 'detection',
        'evaluator': {'type': 'CocoEvaluator', 'iou_types': ['bbox']},
        'num_classes': 1,
        'remap_mscoco_category': False,
        'train_dataloader': {
            'dataset': {
                'img_folder': '/',
                'ann_file': str(train_json_fpath),
                'return_masks': False,
            },
        },
        'val_dataloader': {
            'dataset': {
                'img_folder': '/',
                'ann_file': str(vali_json_fpath),
                'return_masks': False,
            },
        },
    }
    train_config = {
        '__include__': [str(base_config_fpath)],
        'output_dir': str(output_dpath),
        'summary_dir': str(output_dpath / 'summary'),
    }
    train_config = deep_update(train_config, dataset_config)
    train_config = deep_update(train_config, overrides)

    dataset_fpath = generated_dpath / 'dataset.yml'
    train_fpath = generated_dpath / 'train.yml'
    dataset_fpath.write_text(yaml.safe_dump(dataset_config, sort_keys=False))
    train_fpath.write_text(yaml.safe_dump(train_config, sort_keys=False))
    return train_fpath


def train_detector(train_kwcoco, vali_kwcoco, workdir, detector_cfg, test_kwcoco=None,
                   init_checkpoint_fpath=None, device=None, use_amp=False,
                   config_overrides=None):
    workdir = Path(workdir).expanduser().resolve()
    prepared = prepare_detector_training_data(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        test_kwcoco=test_kwcoco,
        output_dpath=workdir / 'prepared_data',
    )
    train_config_fpath = build_training_config(
        detector_cfg=detector_cfg,
        train_json_fpath=prepared.train_coco_fpath,
        vali_json_fpath=prepared.vali_coco_fpath,
        output_dpath=workdir,
        overrides=config_overrides,
    )

    repo_dpath = resolve_repo_dpath(detector_cfg)
    command = [sys.executable, str(repo_dpath / 'train.py'), '-c', str(train_config_fpath)]
    if init_checkpoint_fpath is not None:
        command += ['-t', str(Path(init_checkpoint_fpath).expanduser())]
    if device is not None:
        command += ['-d', device]
    if use_amp:
        command += ['--use-amp']
    subprocess.run(command, cwd=repo_dpath, check=True)
    return train_config_fpath
