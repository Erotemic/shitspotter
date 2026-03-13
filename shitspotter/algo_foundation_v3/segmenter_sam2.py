"""
Thin wrappers around SAM2 image prediction for box-prompted refinement.
"""

import importlib
import sys
from pathlib import Path


def resolve_repo_dpath(segmenter_cfg):
    import os

    envvar = segmenter_cfg.get('repo_envvar', 'SHITSPOTTER_SAM2_REPO_DPATH')
    repo = segmenter_cfg.get('repo_dpath', os.environ.get(envvar, None))
    if repo is None:
        return None
    return Path(repo).expanduser().resolve()


def _ensure_repo_on_path(repo_dpath):
    repo_str = str(repo_dpath)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


class SAM2Segmenter:
    def __init__(self, segmenter_cfg):
        self.segmenter_cfg = segmenter_cfg
        self.predictor = None

    def _lazy_init(self):
        if self.predictor is not None:
            return
        repo_dpath = resolve_repo_dpath(self.segmenter_cfg)
        if repo_dpath is not None:
            _ensure_repo_on_path(repo_dpath)
        try:
            SAM2ImagePredictor = importlib.import_module('sam2.sam2_image_predictor').SAM2ImagePredictor
        except Exception as ex:
            raise ImportError(
                'Unable to import SAM2. Install it or set SHITSPOTTER_SAM2_REPO_DPATH.'
            ) from ex

        checkpoint_fpath = self.segmenter_cfg.get('checkpoint_fpath', None)
        config_fpath = self.segmenter_cfg.get('config_fpath', None)
        if config_fpath is None and repo_dpath is not None and self.segmenter_cfg.get('config_relpath', None):
            config_fpath = repo_dpath / self.segmenter_cfg['config_relpath']

        if checkpoint_fpath and config_fpath:
            build_sam2 = importlib.import_module('sam2.build_sam').build_sam2
            predictor = SAM2ImagePredictor(
                build_sam2(
                    str(config_fpath),
                    str(checkpoint_fpath),
                    device=self.segmenter_cfg.get('device', 'cuda:0'),
                ),
                mask_threshold=float(self.segmenter_cfg.get('mask_threshold', 0.0)),
            )
        else:
            hf_model_id = self.segmenter_cfg.get('hf_model_id', None)
            if hf_model_id is None:
                raise KeyError('Segmenter package requires checkpoint+config or hf_model_id')
            predictor = SAM2ImagePredictor.from_pretrained(
                hf_model_id,
                device=self.segmenter_cfg.get('device', 'cuda:0'),
            )
        self.predictor = predictor

    def predict_masks_for_boxes(self, image, boxes_xyxy):
        self._lazy_init()
        self.predictor.set_image(image)
        mask_infos = []
        for box in boxes_xyxy:
            masks, scores, _ = self.predictor.predict(
                box=box,
                multimask_output=False,
                return_logits=False,
                normalize_coords=False,
            )
            best_idx = int(scores.argmax()) if len(scores) else 0
            mask_infos.append({
                'mask': masks[best_idx],
                'score': float(scores[best_idx]) if len(scores) else 0.0,
            })
        return mask_infos


def validate_segmenter_assets(segmenter_cfg):
    repo_dpath = resolve_repo_dpath(segmenter_cfg)
    if repo_dpath is not None and not repo_dpath.exists():
        raise FileNotFoundError(repo_dpath)
    checkpoint_fpath = segmenter_cfg.get('checkpoint_fpath', None)
    if checkpoint_fpath is not None and not Path(checkpoint_fpath).expanduser().exists():
        raise FileNotFoundError(checkpoint_fpath)
    config_fpath = segmenter_cfg.get('config_fpath', None)
    if config_fpath is not None and not Path(config_fpath).expanduser().exists():
        raise FileNotFoundError(config_fpath)
