# References:
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
# https://github.com/facebookresearch/detectron2/issues/2442

cd ~/code/shitspotter/experiments/detectron2-experiments
rsync -avprPR ~/code/zerowaste/./deeplab .
rsync -avprPR ~/code/zerowaste/./maskrcnn .
#cat zerowaste_config.yaml
#gvim zerowaste_config.yaml

export CUDA_VISIBLE_DEVICES="1,"
python -c "if 1:
    import os
    import detectron2
    import ubelt as ub
    import platform
    import shitspotter
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    full_fpath = ub.Path(shitspotter.util.find_shit_coco_fpath())
    bundle_dpath = full_fpath.parent
    vali_fpath = bundle_dpath / 'train_imgs5747_1e73d54f.mscoco.json

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
    detectron_repo_dpath = ub.Path(detectron2.__file__).parent.parent
    shitspotter_repo_dpath = ub.Path(shitspotter.__file__).parent.parent

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.DATASETS.TRAIN = (dataset_infos['train']['name'],)
    # cfg.DATASETS.TEST = (dataset_infos['vali']['name'],)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real 'batch size' commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 120_000 # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The 'RoIHead batch size'. 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    train_prefix = ub.Path('~/data/dvc-repos/shitspotter_expt_dvc/training').expand()
    cfg.OUTPUT_DIR = None  # hack: null out for the initial
    hashid = ub.hash_data(cfg)[0:8]
    expt_name = f'{config_fpath.stem}_{config_fpath.parent.parent.name}_v3_{hashid}'
    output_dpath = (train_prefix / platform.node() / os.environ['USER'] / 'ShitSpotter' / 'runs' / expt_name)
    output_dpath.ensuredir()
    cfg.OUTPUT_DIR = os.fspath(output_dpath)
    print(ub.urepr(cfg, nl=-1))
    print(hashid)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
"



ls ~/data/dvc-repos/shitspotter_expt_dvc/training/*/*/*/runs/zerowaste_config_maskrcnn_75d01146
ls "$HOME"/data/dvc-repos/shitspotter_expt_dvc/training/*/*/*/runs/zerowaste_config_maskrcnn_75d01146/model_0119999.pth
ls /home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/zerowaste_config_maskrcnn_75d01146/model_0119999.pth

# Demo prediction


python -c "if 1:
    import shitspotter
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg

    checkpoint_fpath = '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/zerowaste_config_maskrcnn_75d01146/model_0119999.pth'

    shitspotter_repo_dpath = ub.Path(shitspotter.__file__).parent.parent
    config_fpath = shitspotter_repo_dpath / 'experiments/detectron2-experiments/maskrcnn/configs/zerowaste_config.yaml'

    cfg = get_cfg()
    print(config_fpath.read_text())
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_fpath)
    cfg.MODEL.WEIGHTS = checkpoint_fpath
    cfg = get_cfg()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set threshold for this model

    predictor = DefaultPredictor(cfg)

    import shitspotter
    full_fpath = ub.Path(shitspotter.util.find_shit_coco_fpath())
    bundle_dpath = full_fpath.parent
    vali_fpath = bundle_dpath / 'vali_imgs691_99b22ad0.mscoco.json'
    import kwcoco
    dset = kwcoco.CocoDataset(vali_fpath)


    import kwimage
    import kwarray
    torch_impl = kwarray.ArrayAPI.coerce('torch')

    images = dset.images()
    coco_img_iter = ub.ProgIter(images.coco_images_iter(), total=len(images), desc='predict')
    for coco_img in coco_img_iter:
        im = coco_img.imdelay(channels='blue|green|red').finalize()
        outputs = predictor(im)
        instances = outputs['instances']
        if len(instances):
            print()
            print(instances)
            print()

            boxes = instances.pred_boxes
            scores = instances.scores
            pred_classes = instances.pred_classes

            boxes = kwimage.Boxes(boxes.tensor, format='xywh').numpy()
            scores = torch_impl.numpy(instances.scores)
            pred_class_indexes = torch_impl.numpy(instances.pred_classes)

            raise Exception


"
