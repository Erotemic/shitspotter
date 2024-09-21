"""
References:
    https://colab.research.google.com/drive/1DIk7bDpdZDkTTZyJbPADZklcbZKr1xkn#scrollTo=DvVulbjZcTdp

SeeAlso:
    ~/code/shitspotter/experiments/detectron2-experiments/setup_detectron.sh
"""


def main():
    import os
    import detectron2
    import ubelt as ub
    import shitspotter
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    full_fpath = ub.Path(shitspotter.util.find_shit_coco_fpath())
    bundle_dpath = full_fpath.parent

    dataset_paths = {
        'vali': bundle_dpath / 'vali_imgs691_99b22ad0.mscoco.json',
        'train': bundle_dpath / 'train_imgs5747_1e73d54f.mscoco.json',
    }

    dataset_infos = {}
    for key, fpath in dataset_paths.items():
        assert fpath.exists()
        row = {'path': fpath}
        row['name'] = fpath.name.split('.', 1)[0]
        dataset_infos[key] = row
    for key, row in dataset_infos.items():
        register_coco_instances(row['name'], {}, row['path'], row['path'].parent)

    # It would be nice if detectron had a resource path, but oh well...
    modpath = ub.Path(detectron2.__file__)
    detectron_repo_dpath = modpath.parent.parent

    cfg = get_cfg()
    cfg.merge_from_file(detectron_repo_dpath / 'configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    cfg.DATASETS.TRAIN = (dataset_infos['train']['name'],)
    cfg.DATASETS.TEST = (dataset_infos['vali']['name'],)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 60

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/experiments/detectron2-experiments/train_maskrcnn_v1.py
    """
    main()
