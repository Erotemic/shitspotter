import os,  json, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as dset

import detectron2
import detectron2.data.transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
#from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
opj = os.path.join



def build_sem_seg_train_aug(cfg):
    augs = [
        #T.Resize((1024, 2048)),
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
                1.,
                255.,
            )
    ]
    
    augs.append(T.RandomFlip())
    return augs

# registers the zero-waste instance segmentation dataset to the catalog
def register_zero_waste_instances(data_root): 
    data_paths = {}
    for split in ["train", "val", "test"]:
        img_folder = opj(data_root, split, "data")
        ann_path = opj(data_root, split, "labels.json")
        data_paths[split] = (img_folder, ann_path)
    register_coco_instances("zero-waste-train", {}, data_paths["train"][1], data_paths["train"][0])
    register_coco_instances("zero-waste-val", {}, data_paths["val"][1], data_paths["val"][0])
    register_coco_instances("zero-waste-test", {}, data_paths["test"][1], data_paths["test"][0])    

# registers the TACO instance segmentation dataset to the catalog
def register_taco_instances(data_root): 
    data_paths = {}
    for split in ["train", "test"]:
        img_folder = opj(data_root, split, "data")
        ann_path = opj(data_root, split, "labels.json")
        data_paths[split] = (img_folder, ann_path)
    register_coco_instances("taco-zw-train", {}, data_paths["train"][1], data_paths["train"][0])
    register_coco_instances("taco-zw-test", {}, data_paths["test"][1], data_paths["test"][0])



# composes a config file describing the model and training logics
def setup_config(args):
    cfg = get_cfg()
    # the line below defines a model with a R_50_FPN_3x backbone; you can change it
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # the COCO checkpoint is loaded by default; redefine MODEL.WEIGHTS in options if need to load locally
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") 
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

# custom trainer (we can override class methods here like lr_schedule etc., if needed)
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder,
                             tasks=["segm"])

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize((1024, 2048)),])
        return build_detection_test_loader(cfg, dataset_name=dataset_name, mapper=mapper)


def main(args):
    if args.taco:
        register_taco_instances(args.dataroot)
    else:
        register_zero_waste_instances(args.dataroot)
    cfg = setup_config(args)

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
    parser.add_argument("--taco", type=bool, default=False,
                         help="whether to register a TACO dataset with ZeroWaste classes")
    args = parser.parse_args()
    print("Command Line Args:", args)
    main(args)
    # launch(
    #     main,
    #     args.num_gpus,
    #     num_machines=args.num_machines,
    #     machine_rank=args.machine_rank,
    #     dist_url=args.dist_url,
    #     args=(args,),
    # )
