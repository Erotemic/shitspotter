"""
Unified inference CLI for the foundation v3 pipeline.
"""

from pathlib import Path

import scriptconfig as scfg
import ubelt as ub

from shitspotter.algo_foundation_v3.baseline_maskdino import MaskDINOPredictor
from shitspotter.algo_foundation_v3.config_utils import nonnull_overrides
from shitspotter.algo_foundation_v3.detector_deimv2 import DEIMv2Predictor
from shitspotter.algo_foundation_v3.detector_opengroundingdino import OpenGroundingDINOPredictor
from shitspotter.algo_foundation_v3.kwcoco_adapter import (
    clone_dataset_for_predictions,
    coerce_input_kwcoco,
    export_predictions_to_labelme,
)
from shitspotter.algo_foundation_v3.datasets import prepare_prediction_io
from shitspotter.algo_foundation_v3.packaging import dump_package, package_name, resolve_package
from shitspotter.algo_foundation_v3.postprocess import (
    add_prediction_annotations,
    detector_records_to_anns,
    mask_records_to_anns,
)
from shitspotter.algo_foundation_v3.segmenter_sam2 import SAM2Segmenter


class AlgoPredictCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path or directory of images')
    dst = scfg.Value(None, help='output kwcoco path or output directory')
    package_fpath = scfg.Value(None, help='path to packaged model config', required=True, alias=['model'])
    backend = scfg.Value(None, help='optional explicit backend; must agree with package')
    create_labelme = scfg.Value(False, help='if True create missing LabelMe sidecars')
    score_thresh = scfg.Value(None, help='override score threshold')
    nms_thresh = scfg.Value(None, help='override NMS threshold')
    crop_padding = scfg.Value(None, help='override detector box padding')
    polygon_simplify = scfg.Value(None, help='override polygon simplification')
    min_component_area = scfg.Value(None, help='override min polygon area')
    keep_largest_component = scfg.Value(None, help='override keep-largest-component postprocess flag')
    device = scfg.Value(None, help='override device for detector / segmenter / baseline')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        import kwutil

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        package_overrides = {
            'backend': config.backend,
            'postprocess': nonnull_overrides(dict(config), [
                'score_thresh', 'nms_thresh', 'crop_padding',
                'polygon_simplify', 'min_component_area',
                'keep_largest_component',
            ]),
        }
        package_overrides = {k: v for k, v in package_overrides.items() if v not in [None, {}]}
        resolved_package = resolve_package(config.package_fpath, overrides=package_overrides)
        if config.backend is not None and config.backend != resolved_package['backend']:
            raise ValueError('CLI backend does not match package backend')

        if config.device is not None:
            if resolved_package['backend'] == 'deimv2_sam2':
                resolved_package['detector']['device'] = config.device
                resolved_package['segmenter']['device'] = config.device
            elif resolved_package['backend'] == 'maskdino':
                resolved_package['baseline']['device'] = config.device
            elif resolved_package['backend'] == 'opengroundingdino_sam2':
                resolved_package['detector']['device'] = config.device
                resolved_package['segmenter']['device'] = config.device

        paths = prepare_prediction_io(config.src, config.dst, package_name(resolved_package))
        src_fpath, _ = coerce_input_kwcoco(config.src, paths)
        pred_dset = clone_dataset_for_predictions(src_fpath, paths['pred_fpath'])
        src_dset = kwcoco.CocoDataset.coerce(src_fpath)

        proc_context = kwutil.ProcessContext(
            name='shitspotter.algo_foundation_v3.predict',
            config=kwutil.Json.ensure_serializable({
                'src': str(config.src),
                'dst': str(paths['pred_fpath']),
                'package_fpath': str(config.package_fpath),
                'resolved_package': resolved_package,
            }),
            track_emissions=True,
        )
        proc_context.start()

        backend = resolved_package['backend']
        if backend == 'deimv2_sam2':
            detector = DEIMv2Predictor(resolved_package['detector'])
            segmenter = SAM2Segmenter(resolved_package['segmenter'])
            for coco_img in ub.ProgIter(src_dset.images().coco_images, desc='predict deimv2_sam2'):
                image = coco_img.imdelay().finalize()
                detector_records = detector.predict_image_records(image)
                anns = detector_records_to_anns(
                    image=image,
                    detector_records=detector_records,
                    segmenter=segmenter,
                    label_mapping=resolved_package['label_mapping'],
                    post_cfg=resolved_package['postprocess'],
                )
                add_prediction_annotations(pred_dset, coco_img.img['id'], anns, backend_name=backend)
        elif backend == 'maskdino':
            predictor = MaskDINOPredictor(resolved_package['baseline'])
            for coco_img in ub.ProgIter(src_dset.images().coco_images, desc='predict maskdino'):
                image = coco_img.imdelay().finalize()
                mask_records = predictor.predict_image_records(image)
                anns = mask_records_to_anns(
                    mask_records=mask_records,
                    label_mapping=resolved_package['label_mapping'],
                    post_cfg=resolved_package['postprocess'],
                )
                add_prediction_annotations(pred_dset, coco_img.img['id'], anns, backend_name=backend)
        elif backend == 'opengroundingdino_sam2':
            detector = OpenGroundingDINOPredictor(resolved_package['detector'])
            segmenter = SAM2Segmenter(resolved_package['segmenter'])
            for coco_img in ub.ProgIter(src_dset.images().coco_images, desc='predict opengroundingdino_sam2'):
                image = coco_img.imdelay().finalize()
                detector_records = detector.predict_image_records(image)
                anns = detector_records_to_anns(
                    image=image,
                    detector_records=detector_records,
                    segmenter=segmenter,
                    label_mapping=resolved_package['label_mapping'],
                    post_cfg=resolved_package['postprocess'],
                )
                add_prediction_annotations(pred_dset, coco_img.img['id'], anns, backend_name=backend)
        else:  # pragma: no cover
            raise KeyError(backend)

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

        print(f'Wrote predictions to {pred_dset.fpath}')


__cli__ = AlgoPredictCLI


if __name__ == '__main__':
    __cli__.main()
