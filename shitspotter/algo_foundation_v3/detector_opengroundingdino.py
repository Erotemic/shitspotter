"""
Adapter for the Open-GroundingDINO detector (tpl/Open-GroundingDino repo).

This module is analogous to detector_deimv2.py: it exposes a predictor class
with a predict_image_records() interface that returns
[{'label': str, 'bbox_ltrb': [x1,y1,x2,y2], 'score': float}, ...],
compatible with postprocess.detector_records_to_anns() and the SAM2 segmenter.

The tpl repo is not a proper Python package, so we add it to sys.path at
load time (same technique used in the existing pipeline scripts).
"""

import sys
import types
from pathlib import Path


def _resolve_repo_dpath(detector_cfg):
    import os
    envvar = detector_cfg.get('repo_envvar', 'SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH')
    repo = detector_cfg.get('repo_dpath', os.environ.get(envvar, None))
    if repo is None:
        raise EnvironmentError(
            f'Open-GroundingDINO repo path is not configured. '
            f'Set {envvar} or detector.repo_dpath.'
        )
    return Path(repo).expanduser().resolve()


def _ensure_repo_on_path(repo_dpath):
    repo_str = str(repo_dpath)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _build_inference_args(detector_cfg, repo_dpath):
    """Construct an args namespace suitable for build_model_main / PostProcess.

    The Open-GroundingDINO model builder reads all arch parameters from an
    SLConfig file and from the args namespace.  We load the SLConfig, merge
    its values into a plain namespace, then apply the inference-time overrides
    (device, label_list, use_coco_eval=False, distributed=False, amp=False).
    """
    from util.slconfig import SLConfig

    config_fpath = detector_cfg.get('config_fpath', None)
    if config_fpath is None:
        raise KeyError('detector_cfg requires config_fpath pointing to the SLConfig (.py) file')

    cfg = SLConfig.fromfile(str(config_fpath))
    cfg_dict = cfg._cfg_dict.to_dict()

    ns = types.SimpleNamespace(**cfg_dict)

    # Inference-time overrides
    ns.device = detector_cfg.get('device', 'cuda:0')
    ns.label_list = detector_cfg.get('label_list', ['poop'])
    ns.use_coco_eval = False
    ns.distributed = False
    ns.amp = False
    ns.rank = 0
    ns.debug = False
    ns.find_unused_params = False
    ns.freeze_keywords = getattr(ns, 'freeze_keywords', None)

    # These may be absent from older config files — supply safe defaults.
    if not hasattr(ns, 'no_interm_box_loss'):
        ns.no_interm_box_loss = False
    if not hasattr(ns, 'interm_loss_coef'):
        ns.interm_loss_coef = 1.0
    if not hasattr(ns, 'nms_iou_threshold'):
        ns.nms_iou_threshold = -1
    if not hasattr(ns, 'use_detached_boxes_dec_out'):
        ns.use_detached_boxes_dec_out = False
    if not hasattr(ns, 'coco_val_path'):
        ns.coco_val_path = ''
    if not hasattr(ns, 'modelname'):
        ns.modelname = 'groundingdino'

    return ns


def _load_checkpoint(checkpoint_fpath):
    import torch
    from groundingdino.util.utils import clean_state_dict
    # weights_only=False because OpenGroundingDINO checkpoints embed
    # argparse.Namespace objects alongside the model state dict.
    ckpt = torch.load(str(checkpoint_fpath), map_location='cpu', weights_only=False)
    return clean_state_dict(ckpt['model'])


def _make_transforms():
    """Return the standard val-time transforms used by Open-GroundingDINO."""
    import torchvision.transforms as T
    return T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _pil_resize_shortest(pil_img, size=800, max_size=1333):
    """Resize PIL image so the shorter side == size, capped by max_size."""
    from PIL import Image
    w, h = pil_img.size
    min_side = min(w, h)
    max_side = max(w, h)
    scale = size / min_side
    if max_side * scale > max_size:
        scale = max_size / max_side
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    return pil_img.resize((new_w, new_h), Image.BILINEAR)


class OpenGroundingDINOPredictor:
    """Detect objects with a fine-tuned Open-GroundingDINO checkpoint.

    Parameters in detector_cfg
    --------------------------
    repo_dpath / repo_envvar : path to the tpl/Open-GroundingDino repo
    config_fpath             : path to the SLConfig (.py) config used for training
    checkpoint_fpath         : path to the trained checkpoint (.pth)
    label_list               : list of category names to detect (default: ['poop'])
    device                   : torch device string (default: 'cuda:0')
    score_thresh             : minimum score to keep a detection (default: 0.0)
    """

    def __init__(self, detector_cfg):
        self.detector_cfg = detector_cfg
        self.model = None
        self.postprocessors = None
        self.transforms = None
        self.device = detector_cfg.get('device', 'cuda:0')
        self.label_list = detector_cfg.get('label_list', ['poop'])
        self.score_thresh = float(detector_cfg.get('score_thresh', 0.0))
        self._caption = ' . '.join(self.label_list) + ' .'

    def _lazy_init(self):
        if self.model is not None:
            return
        import warnings
        import torch

        # These warnings fire on every forward pass from OpenGroundingDINO's
        # internal use of torch.utils.checkpoint.  They are library-level issues
        # we cannot fix; show each once rather than flooding the output.
        warnings.filterwarnings('once', message='.*use_reentrant parameter should be passed.*')
        warnings.filterwarnings('once', message='.*None of the inputs have requires_grad.*')

        repo_dpath = _resolve_repo_dpath(self.detector_cfg)
        _ensure_repo_on_path(repo_dpath)

        from main import build_model_main

        checkpoint_fpath = self.detector_cfg.get('checkpoint_fpath', None)
        if checkpoint_fpath is None:
            raise KeyError('detector_cfg requires checkpoint_fpath')

        args = _build_inference_args(self.detector_cfg, repo_dpath)
        model, _criterion, postprocessors = build_model_main(args)

        state_dict = _load_checkpoint(checkpoint_fpath)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)

        self.model = model
        self.postprocessors = postprocessors
        self.transforms = _make_transforms()

    def predict_image_records(self, image):
        """Run detection on a single HxWxC uint8 numpy image.

        Returns
        -------
        list[dict]
            Each dict has keys 'label' (str), 'bbox_ltrb' ([x1,y1,x2,y2]),
            and 'score' (float).  All boxes are in original image coordinates.
        """
        import torch
        from PIL import Image

        self._lazy_init()

        image_pil = Image.fromarray(image)
        orig_w, orig_h = image_pil.size

        resized_pil = _pil_resize_shortest(image_pil)
        im_tensor = self.transforms(resized_pil).unsqueeze(0).to(self.device)

        # Build a NestedTensor with a zero mask (no padding).
        from util.misc import NestedTensor
        mask = torch.zeros(
            (1, im_tensor.shape[2], im_tensor.shape[3]),
            dtype=torch.bool,
            device=self.device,
        )
        nested = NestedTensor(im_tensor, mask)

        orig_target_sizes = torch.tensor(
            [[orig_h, orig_w]], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            outputs = self.model(nested, captions=[self._caption])

        results = self.postprocessors['bbox'](outputs, orig_target_sizes)
        result = results[0]

        scores = result['scores'].detach().cpu().tolist()
        labels = result['labels'].detach().cpu().tolist()
        boxes = result['boxes'].detach().cpu().tolist()

        records = []
        for score, label_idx, box in zip(scores, labels, boxes):
            if score < self.score_thresh:
                continue
            label_str = (
                self.label_list[label_idx]
                if 0 <= label_idx < len(self.label_list)
                else str(label_idx)
            )
            records.append({
                'label': label_str,
                'bbox_ltrb': box,
                'score': score,
            })
        return records
