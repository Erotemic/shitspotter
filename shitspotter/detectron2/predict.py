#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class DetectronPredictCLI(scfg.DataConfig):
    # param1 = scfg.Value(None, help='param1')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from shitspotter.predict import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict()
            >>> cls = PredictCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

__cli__ = DetectronPredictCLI


def detectron_predict(config):
    # import shitspotter
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    import kwcoco
    import kwimage
    import kwarray

    # checkpoint_fpath = '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/zerowaste_config_maskrcnn_75d01146/model_0119999.pth'
    checkpoint_fpath = '/home/joncrall/data/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/train_baseline_maskrcnn_v3/v_966e49df/model_0029999.pth'
    vali_fpath = '/home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs691_99b22ad0.mscoco.json'

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
    # cfg.DATASETS.TRAIN = (dataset_infos['train']['name'],)
    # cfg.DATASETS.TEST = (dataset_infos['vali']['name'],)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2   # This is the real 'batch size' commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025   # pick a good LR
    cfg.SOLVER.MAX_ITER = 120_000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []          # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The 'RoIHead batch size'. 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1  # set threshold for this model
    cfg.MODEL.WEIGHTS = checkpoint_fpath

    predictor = DefaultPredictor(cfg)

    # full_fpath = ub.Path(shitspotter.util.find_shit_coco_fpath())
    # bundle_dpath = full_fpath.parent
    # vali_fpath = bundle_dpath / 'vali_imgs691_99b22ad0.mscoco.json'
    dset = kwcoco.CocoDataset(vali_fpath)
    dset.clear_annotations()

    torch_impl = kwarray.ArrayAPI.coerce('torch')()

    images = dset.images()
    coco_img_iter = ub.ProgIter(images.coco_images_iter(), total=len(images), desc='predict')
    for coco_img in coco_img_iter:
        image_id = coco_img['id']
        im = coco_img.imdelay(channels='blue|green|red').finalize()
        outputs = predictor(im)
        instances = outputs['instances']
        if len(instances):
            boxes = instances.pred_boxes
            scores = instances.scores
            boxes = kwimage.Boxes(boxes.tensor, format='ltrb').numpy()
            scores = torch_impl.numpy(instances.scores)
            pred_class_indexes = torch_impl.numpy(instances.pred_classes)
            dets = kwimage.Detections(
                boxes=boxes,
                scores=scores,
                class_idxs=pred_class_indexes
            )
            for ann in dets.to_coco():
                ann['image_id'] = image_id
                dset.add_annotation(**ann)

    dset.reroot(absolute=True)
    dset.fpath = 'pred.kwcoco.zip'
    dset.dump()


if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/predict.py
        python -m shitspotter.predict
    """
    __cli__.main()
