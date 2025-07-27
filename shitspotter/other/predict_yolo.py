r"""
This requires our custom fork of YOLO-v9

SeeAlso:
    ~/code/shitspotter/experiments/yolo-experiments/run_yolo_experiments_v1.sh
    /data/joncrall/dvc-repos/shitspotter_dvc/models/yolo-v9/demo-predict-torch.sh

git@github.com:Erotemic/YOLO.git

on the upgrades branch

CommandLine:
    python -m shitspotter.other.predict_yolo \
        --src path/to/input.kwcoco.zip \
        --dst path/to/output.kwcoco.zip \
        --checkpoint path/to/model.ckpt \
        --config path/to/train_config.yaml

Example:
    python -m shitspotter.other.predict_yolo \
        --src "$KWCOCO_BUNDLE_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip" \
        --dst "$EVAL_PATH/preds/yolo/test_imgs121_6cb3b6ff/pred.kwcoco.zip" \
        --checkpoint "$YOLO_MODEL_DPATH/best.ckpt" \
        --config "$YOLO_MODEL_DPATH/train_config.yaml"
"""
import ubelt as ub
import kwcoco
import numpy as np
import torch
import rich
import kwutil
import kwimage

from omegaconf.dictconfig import DictConfig
from yolo.utils.kwcoco_utils import tensor_to_kwimage
from yolo.utils.bounding_box_utils import create_converter
from yolo.utils.model_utils import PostProcess
from yolo.tools.solver import InferenceModel


def main(argv=True, **kwargs):
    """
    Ignore:
        argv = False
        kwargs = {
            'src': ub.Path('/home/joncrall/code/shitspotter/shitspotter_dvc/test_imgs30_d8988f8c.kwcoco.zip').expand(),
            'dst': ub.Path('~/data/dvc-repos/shitspotter_expt_dvc/_yolov9_evals/test/pred.kwcoco.zip').expand(),
            'checkpoint': ub.Path('~/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-epoch=0032-step=000132-trainlosstrain_loss=7.603.ckpt.ckpt').expand(),
            'model_config': ub.Path('~/code/shitspotter/shitspotter_dvc/models/yolo-v9/shitspotter-simple-v3-run-v06-train_config.yaml').expand(),
        }
    """
    import argparse

    parser = argparse.ArgumentParser(description='YOLO prediction to KWCoco')
    parser.add_argument('--src', type=str, required=True, help='Input kwcoco file')
    parser.add_argument('--dst', type=str, required=True, help='Output kwcoco file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to YOLO checkpoint')
    parser.add_argument('--model_config', type=str, required=True, help='Path to training config')
    parser.add_argument('--device', type=str, default='cuda', help='Torch device')
    parser.add_argument('--min_confidence', type=float, default=0.01, help='Minimum confidence threshold')
    parser.add_argument('--min_iou', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max_bbox', type=int, default=300, help='Max number of boxes per image')

    import scriptconfig as scfg
    PredictYoloCLI = scfg.DataConfig.cls_from_argparse(parser)
    args = PredictYoloCLI.cli(argv=argv, data=kwargs, verbose=True, special_options=False)

    device = torch.device(args.device)

    rich.print('[green]YOLO Prediction with args:[/green]')
    rich.print(vars(args))

    output_fpath = ub.Path(args.dst)
    output_fpath.parent.ensuredir()

    input_dset = kwcoco.CocoDataset.coerce(args.src)
    output_dset = input_dset.copy()
    output_dset.clear_annotations()
    output_dset.reroot(absolute=True)
    output_dset.fpath = output_fpath
    # output_dset._update_fpath(output_fpath) # broken

    proc_context = kwutil.ProcessContext(
        name='shitspotter.other.coco_yolo_predict',
        config=kwutil.Json.ensure_serializable(dict(args)),
        track_emissions=True,
    )
    proc_context.start()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')

    model_config = kwutil.Yaml.coerce(args.model_config, backend='pyyaml')
    checkpoint = args.checkpoint

    if isinstance(model_config, dict) and len(model_config) == 2:
        # hack to make coupling args easier with mlops
        if 'config' in model_config and 'checkpoint' in model_config:
            checkpoint = model_config['checkpoint']
            model_config = kwutil.Yaml.coerce(model_config['config'], backend='pyyaml')

    cfg = DictConfig(model_config)
    cfg.weight = checkpoint

    model = InferenceModel(cfg)
    model.to(device)
    model.eval()
    # dsize = tuple(cfg.image_size)

    vec2box = create_converter(
        cfg.model.name, model, cfg.model.anchor, cfg.image_size, device
    )

    vec2box = create_converter(
        cfg.model.name, model, cfg.model.anchor, cfg.image_size, device
    )

    nms_cfg = DictConfig({
        'min_confidence': args.min_confidence,
        'min_iou': args.min_iou,
        'max_bbox': args.max_bbox,
    })
    post_process = PostProcess(vec2box, nms_cfg)

    classes = cfg.dataset.class_list
    classes = kwcoco.CategoryTree.coerce([str(c) for c in classes])

    from yolo.tools.data_augmentation import PadAndResize
    image_size = model.cfg.image_size
    pad_resize = PadAndResize(image_size=image_size)

    for image_id in ub.ProgIter(input_dset.images(), desc='Predicting'):
        coco_img = output_dset.coco_image(image_id)
        imdata = coco_img.imdelay().finalize()

        # Ensure we are using the exact same data processing pipeline as
        # training
        from PIL import Image
        pil_img = Image.fromarray(imdata)
        pil_img = pil_img.convert('RGB')

        boxes = np.empty((0, 5))
        padded_image, _, rev_tensor = pad_resize(pil_img, boxes)

        import kwarray
        scale, pad_left, pad_top, pad_right, pad_bot = kwarray.ArrayAPI.numpy(rev_tensor)

        #input_hwc = dset.coco_image(2).imdelay().resize((640, 640)).finalize()
        #input_tensor = input_hwc.transpose(2, 0, 1)[None, ...]
        #input_tensor = input_tensor.astype(np.float32)
        #torch_inputs = torch.Tensor(input_tensor)

        from torchvision.transforms import functional as TF
        torch_inputs = TF.to_tensor(padded_image)[None, :]
        resize_info = {
            'scale': float(scale),
            'offset': tuple([float(pad_left), float(pad_top)]),
        }

        # resized, resize_info = kwimage.imresize(imdata, dsize=dsize, return_info=True)
        # input_tensor = resized.transpose(2, 0, 1)[None, ...].astype(np.float32)
        # input_tensor = torch.tensor(input_tensor, device=device)

        with torch.no_grad():
            torch_inputs = torch_inputs.to(device)
            raw_preds = model.forward(torch_inputs)
            preds = post_process(raw_preds)

        detections = [
            tensor_to_kwimage(yolo_tensor, classes=classes).numpy()
            for yolo_tensor in preds
        ]
        if len(detections) == 0:
            continue

        # Get the inverse transform to bring dets back to original size
        det_rescaler = kwimage.Affine.coerce(ub.udict(resize_info) - {'dsize'}).inv()
        detections = [dets.warp(det_rescaler) for dets in detections]

        for dets in detections:
            for ann in dets.to_coco(image_id=image_id):
                cat_id = input_dset.ensure_category(ann['category_name'])
                ann['category_id'] = cat_id
                output_dset.add_annotation(**ann)

    output_dset.dataset.setdefault('info', [])
    proc_context.stop()
    print(f'proc_context.obj = {ub.urepr(proc_context.obj, nl=3)}')
    output_dset.dataset['info'].append(proc_context.obj)
    output_dset.dump()
    bundle_dpath = ub.Path(output_dset.fpath).parent.ensuredir()
    rich.print(f'Wrote in: [link={bundle_dpath}]{bundle_dpath}[/link]')
    print(f'Wrote to: {output_dset.fpath}')


if __name__ == '__main__':
    main()
