"""
Detector-only inference CLI for the foundation v3 pipeline.
"""

import scriptconfig as scfg
import ubelt as ub

from shitspotter.algo_foundation_v3.config_utils import nonnull_overrides
from shitspotter.algo_foundation_v3.datasets import prepare_prediction_io
from shitspotter.algo_foundation_v3.detector_deimv2 import DEIMv2Predictor
from shitspotter.algo_foundation_v3.kwcoco_adapter import clone_dataset_for_predictions, coerce_input_kwcoco
from shitspotter.algo_foundation_v3.packaging import dump_package, package_name, resolve_package
from shitspotter.algo_foundation_v3.postprocess import add_prediction_annotations, detector_records_to_bbox_anns


class AlgoPredictBoxesCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path or directory of images')
    dst = scfg.Value(None, help='output kwcoco path or output directory')
    package_fpath = scfg.Value(None, help='path to packaged model config', required=True, alias=['model'])
    score_thresh = scfg.Value(None, help='override score threshold')
    nms_thresh = scfg.Value(None, help='override NMS threshold')
    device = scfg.Value(None, help='override detector device')

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        import kwutil

        config = cls.cli(argv=argv, data=kwargs, strict=True)
        package_overrides = {
            'postprocess': nonnull_overrides(dict(config), [
                'score_thresh', 'nms_thresh',
            ]),
        }
        package_overrides = {k: v for k, v in package_overrides.items() if v not in [None, {}]}
        resolved_package = resolve_package(config.package_fpath, overrides=package_overrides)
        if resolved_package['backend'] != 'deimv2_sam2':
            raise ValueError('Detector-only box prediction currently requires a deimv2_sam2 package')

        if config.device is not None:
            resolved_package['detector']['device'] = config.device

        paths = prepare_prediction_io(config.src, config.dst, package_name(resolved_package) + '_boxes')
        src_fpath, _ = coerce_input_kwcoco(config.src, paths)
        pred_dset = clone_dataset_for_predictions(src_fpath, paths['pred_fpath'])
        src_dset = kwcoco.CocoDataset.coerce(src_fpath)

        proc_context = kwutil.ProcessContext(
            name='shitspotter.algo_foundation_v3.predict_boxes',
            config=kwutil.Json.ensure_serializable({
                'src': str(config.src),
                'dst': str(paths['pred_fpath']),
                'package_fpath': str(config.package_fpath),
                'resolved_package': resolved_package,
            }),
            track_emissions=True,
        )
        proc_context.start()

        detector = DEIMv2Predictor(resolved_package['detector'])
        for coco_img in ub.ProgIter(src_dset.images().coco_images, desc='predict deimv2 boxes'):
            image = coco_img.imdelay().finalize()
            detector_records = detector.predict_image_records(image)
            anns = detector_records_to_bbox_anns(
                detector_records=detector_records,
                label_mapping=resolved_package['label_mapping'],
                post_cfg=resolved_package['postprocess'],
            )
            add_prediction_annotations(pred_dset, coco_img.img['id'], anns, backend_name='deimv2_boxes')

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
        print(f'Wrote detector-only predictions to {pred_dset.fpath}')


__cli__ = AlgoPredictBoxesCLI


if __name__ == '__main__':
    __cli__.main()
