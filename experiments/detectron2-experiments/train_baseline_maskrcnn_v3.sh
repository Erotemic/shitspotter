#!/bin/bash
# References:
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0
# https://github.com/facebookresearch/detectron2/issues/2442
export CUDA_VISIBLE_DEVICES="1,"
#export CUDA_VISIBLE_DEVICES=0
DVC_DATA_DPATH=$HOME/data/dvc-repos/shitspotter_dvc
DVC_EXPT_DPATH=$HOME/data/dvc-repos/shitspotter_expt_dvc
WORKDIR=$DVC_EXPT_DPATH/training/$HOSTNAME/$USER

DATASET_CODE=ShitSpotter
EXPERIMENT_NAME="train_baseline_maskrcnn_v3"
KWCOCO_BUNDLE_DPATH=$DVC_DATA_DPATH
DEFAULT_ROOT_DIR=$WORKDIR/$DATASET_CODE/runs/$EXPERIMENT_NAME

TRAIN_FPATH=$KWCOCO_BUNDLE_DPATH/train_imgs5747_1e73d54f.mscoco.json
VALI_FPATH=$KWCOCO_BUNDLE_DPATH/vali_imgs691_99b22ad0.mscoco.json
echo "TRAIN_FPATH = $TRAIN_FPATH"
echo "VALI_FPATH = $VALI_FPATH"
echo "DEFAULT_ROOT_DIR = $DEFAULT_ROOT_DIR"

echo "
default_root_dir: $DEFAULT_ROOT_DIR
expt_name: train_baseline_maskrcnn_v3
train_fpath: $TRAIN_FPATH
vali_fpath: $VALI_FPATH
" > train_config_v3.yaml
cat train_config_v3.yaml
python -m shitspotter.detectron2.fit --config train_config_v3.yaml



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
