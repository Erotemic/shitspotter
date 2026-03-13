"""
Thin wrappers around the MaskDINO repo for training and inference.
"""

import importlib
import sys
from pathlib import Path

from shitspotter.algo_foundation_v3.datasets import prepare_maskdino_training_data


def resolve_repo_dpath(baseline_cfg):
    import os

    envvar = baseline_cfg.get('repo_envvar', 'SHITSPOTTER_MASKDINO_REPO_DPATH')
    repo = baseline_cfg.get('repo_dpath', os.environ.get(envvar, None))
    if repo is None:
        raise EnvironmentError(
            f'MaskDINO repo path is not configured. Set {envvar} or baseline.repo_dpath.'
        )
    return Path(repo).expanduser().resolve()


def _ensure_repo_on_path(repo_dpath):
    repo_str = str(repo_dpath)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def resolve_config_fpath(baseline_cfg):
    repo_dpath = resolve_repo_dpath(baseline_cfg)
    config_fpath = baseline_cfg.get('config_fpath', None)
    if config_fpath is not None:
        return Path(config_fpath).expanduser().resolve()
    config_relpath = baseline_cfg.get('config_relpath', None)
    if config_relpath is None:
        raise KeyError('baseline config requires config_fpath or config_relpath')
    return (repo_dpath / config_relpath).resolve()


def _register_coco_dataset(name, json_fpath):
    from detectron2.data import DatasetCatalog
    from detectron2.data.datasets import register_coco_instances

    if name not in DatasetCatalog.list():
        register_coco_instances(name, {}, str(json_fpath), '/')
    return name


def train_maskdino(train_kwcoco, vali_kwcoco, workdir, baseline_cfg,
                   test_kwcoco=None, init_checkpoint_fpath=None,
                   ims_per_batch=None, base_lr=None, max_iter=None,
                   num_workers=None):
    repo_dpath = resolve_repo_dpath(baseline_cfg)
    _ensure_repo_on_path(repo_dpath)

    prepared = prepare_maskdino_training_data(
        train_kwcoco=train_kwcoco,
        vali_kwcoco=vali_kwcoco,
        test_kwcoco=test_kwcoco,
        output_dpath=Path(workdir) / 'prepared_data',
    )
    train_name = _register_coco_dataset(f'shitspotter_maskdino_train_{Path(workdir).name}', prepared.train_coco_fpath)
    vali_name = _register_coco_dataset(f'shitspotter_maskdino_vali_{Path(workdir).name}', prepared.vali_coco_fpath)

    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    add_maskdino_config = importlib.import_module('maskdino').add_maskdino_config
    train_net = importlib.import_module('train_net')

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    cfg.merge_from_file(str(resolve_config_fpath(baseline_cfg)))
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (vali_name,)
    cfg.OUTPUT_DIR = str(Path(workdir).expanduser().resolve())
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(baseline_cfg.get('num_classes', 1))
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    if init_checkpoint_fpath is not None:
        cfg.MODEL.WEIGHTS = str(Path(init_checkpoint_fpath).expanduser().resolve())
    if ims_per_batch is not None:
        cfg.SOLVER.IMS_PER_BATCH = int(ims_per_batch)
    if base_lr is not None:
        cfg.SOLVER.BASE_LR = float(base_lr)
    if max_iter is not None:
        cfg.SOLVER.MAX_ITER = int(max_iter)
    if num_workers is not None:
        cfg.DATALOADER.NUM_WORKERS = int(num_workers)

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer = train_net.Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return Path(cfg.OUTPUT_DIR)


class MaskDINOPredictor:
    def __init__(self, baseline_cfg):
        self.baseline_cfg = baseline_cfg
        self.predictor = None

    def _lazy_init(self):
        if self.predictor is not None:
            return
        repo_dpath = resolve_repo_dpath(self.baseline_cfg)
        _ensure_repo_on_path(repo_dpath)

        from detectron2.config import get_cfg
        from detectron2.engine.defaults import DefaultPredictor
        from detectron2.projects.deeplab import add_deeplab_config

        add_maskdino_config = importlib.import_module('maskdino').add_maskdino_config

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(str(resolve_config_fpath(self.baseline_cfg)))
        cfg.MODEL.WEIGHTS = str(Path(self.baseline_cfg['checkpoint_fpath']).expanduser().resolve())
        cfg.MODEL.DEVICE = self.baseline_cfg.get('device', 'cuda:0')
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = int(self.baseline_cfg.get('num_classes', 1))
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
        self.predictor = DefaultPredictor(cfg)

    def predict_image_records(self, image):
        self._lazy_init()
        instances = self.predictor(image[:, :, ::-1])['instances'].to('cpu')
        if not len(instances):
            return []
        boxes = instances.pred_boxes.tensor.numpy().tolist()
        scores = instances.scores.numpy().tolist()
        labels = instances.pred_classes.numpy().tolist()
        masks = instances.pred_masks.numpy()
        return [
            {'label': label, 'bbox_ltrb': box, 'score': score, 'mask': mask}
            for label, box, score, mask in zip(labels, boxes, scores, masks)
        ]
