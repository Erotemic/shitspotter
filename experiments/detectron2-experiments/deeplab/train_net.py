#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch
import itertools
import numpy as np
import json
from collections import OrderedDict

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import load_sem_seg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

opj = os.path.join


def build_sem_seg_train_aug(cfg):
    augs = [
        T.Resize((1024, 2048)),
        #T.Resize((1024, 2048)),
        T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        ),
        T.RandomRotation([-90, 90]),
        T.RandomBrightness(0.6, 1.4),
        T.RandomSaturation(0.6, 1.4),
        T.RandomLighting(0.3),
        T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                0.,
            )
    ]
    
    augs.append(T.RandomFlip())
    return augs

class SemSegAPEvaluator(SemSegEvaluator):
    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            conf_matrix_list = all_gather(self._conf_matrix)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self._conf_matrix = np.zeros_like(self._conf_matrix)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "sem_seg_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))

        acc = np.full(self._num_classes, np.nan, dtype=np.float)
        iou = np.full(self._num_classes, np.nan, dtype=np.float)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float)
        class_weights = pos_gt / np.sum(pos_gt)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        macc = np.sum(acc[acc_valid]) / np.sum(acc_valid)
        miou = np.sum(iou[acc_valid]) / np.sum(iou_valid)
        fiou = np.sum(iou[acc_valid] * class_weights[acc_valid])
        pacc = np.sum(tp) / np.sum(pos_gt)

        res = {}
        res["mIoU"] = 100 * miou
        res["fwIoU"] = 100 * fiou
        for i, name in enumerate(self._class_names):
            res["IoU-{}".format(name)] = 100 * iou[i]
        res["mACC"] = 100 * macc
        res["pACC"] = 100 * pacc
        for i, name in enumerate(self._class_names):
            res["ACC-{}".format(name)] = 100 * acc[i]

        prec = np.full(self._num_classes, np.nan, dtype=np.float)
        prec_valid = pos_pred > 0
        prec[prec_valid] = tp[prec_valid] / pos_pred[prec_valid]
        res["mPREC"] = 100 * np.mean(prec)
        for i, name in enumerate(self._class_names):
            res["PREC-{}".format(name)] = 100 * prec[i]

        for i, name_i in enumerate(self._class_names):
            for j, name_j in enumerate(self._class_names):
                res["CONF-{}-{}".format(name_i, name_j)] = self._conf_matrix[i, j]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "sem_seg_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"sem_seg": res})
        self._logger.info(results)
        return results


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        return SemSegAPEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        

    @classmethod
    def build_train_loader(cls, cfg):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
            mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize((1024, 2048)),])
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)


    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)
        

    # @classmethod
    # def build_optimizer(cls, cfg, model):
    #     """
    #     Returns:
    #         torch.optim.Optimizer:

    #     It now calls :func:`detectron2.solver.build_optimizer`.
    #     Overwrite it if you'd like a different optimizer.
    #     """
    #     return build_optimizer(cfg, model)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 5

    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def register_zero_waste_semseg(data_root):
    data_paths = {}
    for split in ["train", "val", "test"]:
        img_folder = opj(data_root, split, "data")
        ann_path = opj(data_root, split, "labels.json")
        sem_seg_path = opj(data_root, split, "sem_seg")
        data_paths[split] = (img_folder, ann_path, sem_seg_path)

    def get_train_dataloader():
        return load_sem_seg(gt_root=data_paths["train"][2], 
                            image_root=data_paths["train"][0], 
                            gt_ext='PNG', image_ext='PNG')
    
    def get_val_dataloader():
        return load_sem_seg(gt_root=data_paths["val"][2], 
                            image_root=data_paths["val"][0], 
                            gt_ext='PNG', image_ext='PNG')
    
    def get_test_dataloader():
        return load_sem_seg(gt_root=data_paths["test"][2], 
                            image_root=data_paths["test"][0], 
                            gt_ext='PNG', image_ext='PNG')

    print("Registering the zero-waste dataset splits")
    DatasetCatalog.register("zero-waste-semseg-train", get_train_dataloader)
    DatasetCatalog.register("zero-waste-semseg-val", get_val_dataloader)
    DatasetCatalog.register("zero-waste-semseg-test", get_test_dataloader)
    class_names = ["background", 'rigid_plastic', 'cardboard', 'metal', 'soft_plastic']
    class_colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (125, 0, 125)]
    # adding the metadata
    for split in ["train", "val", "test"]:
        #MetadataCatalog.get("zero-waste-semseg-%s" % split).thing_classes = class_names[1:]
        MetadataCatalog.get("zero-waste-semseg-%s" % split).stuff_classes = class_names
        MetadataCatalog.get("zero-waste-semseg-%s" % split).evaluator_type = "sem_seg"
        #MetadataCatalog.get("zero-waste-semseg-%s" % split).thing_colors = class_colors[1:]
        MetadataCatalog.get("zero-waste-semseg-%s" % split).stuff_colors = class_colors
        MetadataCatalog.get("zero-waste-semseg-%s" % split).ignore_label = 255




def main(args):
    register_zero_waste_semseg(args.dataroot)
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataroot", type=str, default="/scratch2/dinka/data/recycling/splits/",
                         help="root folder for the dataset on the disk")
    args = parser.parse_args()
    print("Command Line Args:", args)
    #main(args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )