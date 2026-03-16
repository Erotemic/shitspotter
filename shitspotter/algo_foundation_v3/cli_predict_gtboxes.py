"""
Predict masks using ground-truth boxes as prompts.
"""

import scriptconfig as scfg
import ubelt as ub

from shitspotter.algo_foundation_v3.config_utils import nonnull_overrides
from shitspotter.algo_foundation_v3.datasets import prepare_prediction_io
from shitspotter.algo_foundation_v3.kwcoco_adapter import (
    clone_dataset_for_predictions,
    coerce_input_kwcoco,
)
from shitspotter.algo_foundation_v3.packaging import dump_package, package_name, resolve_package
from shitspotter.algo_foundation_v3.polygon_utils import (
    expand_box_ltrb,
    mask_to_multi_polygon,
    segmentation_to_coco,
)
from shitspotter.algo_foundation_v3.segmenter_sam2 import SAM2Segmenter


class AlgoPredictGTBoxesCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path with truth boxes')
    dst = scfg.Value(None, help='output kwcoco path or directory')
    package_fpath = scfg.Value(None, help='path to packaged model config', required=True, alias=['model'])
    create_labelme = scfg.Value(False, help='if True create missing LabelMe sidecars')
    crop_padding = scfg.Value(None, help='override detector box padding')
    polygon_simplify = scfg.Value(None, help='override polygon simplification')
    min_component_area = scfg.Value(None, help='override min polygon area')
    keep_largest_component = scfg.Value(None, help='override keep-largest-component postprocess flag')
    device = scfg.Value(None, help='override device for segmenter')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        import kwutil

        from shitspotter.algo_foundation_v3.kwcoco_adapter import export_predictions_to_labelme

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        package_overrides = {
            'postprocess': nonnull_overrides(dict(config), [
                'crop_padding', 'polygon_simplify', 'min_component_area',
                'keep_largest_component',
            ]),
        }
        package_overrides = {k: v for k, v in package_overrides.items() if v not in [None, {}]}
        resolved_package = resolve_package(config.package_fpath, overrides=package_overrides)
        if resolved_package['backend'] != 'deimv2_sam2':
            raise ValueError('GT-box segmenter prediction requires a deimv2_sam2 package')

        if config.device is not None:
            resolved_package['segmenter']['device'] = config.device

        paths = prepare_prediction_io(config.src, config.dst, package_name(resolved_package) + '_gtboxes')
        src_fpath, _ = coerce_input_kwcoco(config.src, paths)
        pred_dset = clone_dataset_for_predictions(src_fpath, paths['pred_fpath'])
        src_dset = kwcoco.CocoDataset.coerce(src_fpath)

        proc_context = kwutil.ProcessContext(
            name='shitspotter.algo_foundation_v3.predict_gtboxes',
            config=kwutil.Json.ensure_serializable({
                'src': str(config.src),
                'dst': str(paths['pred_fpath']),
                'package_fpath': str(config.package_fpath),
                'resolved_package': resolved_package,
            }),
            track_emissions=True,
        )
        proc_context.start()

        segmenter = SAM2Segmenter(resolved_package['segmenter'])
        post_cfg = resolved_package['postprocess']
        for coco_img in ub.ProgIter(src_dset.images().coco_images, desc='predict sam2 from gt boxes'):
            image = coco_img.imdelay().finalize()
            image_shape = image.shape
            anns = coco_img.annots().objs
            gt_anns = [
                ann for ann in anns
                if ann.get('bbox', None) is not None and ann.get('category_id', None) is not None
            ]
            if not gt_anns:
                continue
            padded_boxes = []
            valid_gt_anns = []
            for ann in gt_anns:
                x, y, w, h = ann['bbox']
                box_ltrb = [x, y, x + w, y + h]
                padded_boxes.append(expand_box_ltrb(box_ltrb, post_cfg['crop_padding'], image_shape))
                valid_gt_anns.append(ann)
            mask_infos = segmenter.predict_masks_for_boxes(image, padded_boxes)
            for ann, prompt_box_ltrb, mask_info in zip(valid_gt_anns, padded_boxes, mask_infos):
                mpoly = mask_to_multi_polygon(
                    mask_info['mask'],
                    polygon_simplify=post_cfg['polygon_simplify'],
                    min_component_area=post_cfg['min_component_area'],
                    keep_largest_component=post_cfg['keep_largest_component'],
                )
                if not len(mpoly.data):
                    continue
                pred_ann = {
                    'image_id': coco_img.img['id'],
                    'category_id': ann['category_id'],
                    'bbox': mpoly.box().to_coco(),
                    'segmentation': segmentation_to_coco(mpoly),
                    'score': 1.0,
                    'role': 'prediction',
                    'foundation_backend': 'sam2_gtboxes',
                    'foundation_prompt_source': 'truth_box',
                    'source_gt_ann_id': ann.get('id', None),
                    'source_gt_bbox': ann['bbox'],
                    'prompt_bbox': [
                        float(prompt_box_ltrb[0]),
                        float(prompt_box_ltrb[1]),
                        float(prompt_box_ltrb[2] - prompt_box_ltrb[0]),
                        float(prompt_box_ltrb[3] - prompt_box_ltrb[1]),
                    ],
                }
                pred_dset.add_annotation(**pred_ann)

        proc_context.stop()
        proc_info = proc_context.obj or {}
        properties = proc_info.setdefault('properties', {})
        extra = properties.get('extra')
        if extra is None:
            extra = {}
            properties['extra'] = extra
        extra['resolved_package_fpath'] = str(paths['resolved_package_fpath'])
        pred_dset.dataset.setdefault('info', []).append(proc_info)
        dump_package(resolved_package, paths['resolved_package_fpath'])
        pred_dset.dump()

        if config.create_labelme:
            export_predictions_to_labelme(pred_dset, only_missing=True)

        print(f'Wrote GT-box prompted SAM2 predictions to {pred_dset.fpath}')


__cli__ = AlgoPredictGTBoxesCLI


if __name__ == '__main__':
    __cli__.main()
