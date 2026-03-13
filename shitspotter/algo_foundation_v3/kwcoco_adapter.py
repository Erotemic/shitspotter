"""
Helpers for kwcoco IO, prediction bundle preparation, and LabelMe export.
"""

from pathlib import Path


def _image_extensions():
    import kwimage.im_io
    return kwimage.im_io.IMAGE_EXTENSIONS


def resolve_prediction_paths(src, dst=None, package_name='foundation_model'):
    import ubelt as ub

    src = ub.Path(src).expand()
    if dst is None:
        if src.is_dir():
            out_dpath = src / '_predictions' / 'foundation_detseg_v3' / package_name
        else:
            out_dpath = src.parent / (src.name + '-predict-output') / 'foundation_detseg_v3' / package_name
        pred_fpath = out_dpath / 'pred.kwcoco.zip'
    else:
        dst = ub.Path(dst).expand()
        if dst.exists() and dst.is_dir():
            out_dpath = dst
            pred_fpath = out_dpath / 'pred.kwcoco.zip'
        elif '.' not in dst.name:
            out_dpath = dst
            pred_fpath = out_dpath / 'pred.kwcoco.zip'
        else:
            pred_fpath = dst
            out_dpath = pred_fpath.parent

    out_dpath.ensuredir()
    return {
        'src': src,
        'out_dpath': out_dpath,
        'pred_fpath': pred_fpath,
        'input_fpath': out_dpath / 'input.kwcoco.zip',
        'resolved_package_fpath': out_dpath / 'resolved_package.yaml',
    }


def build_input_kwcoco_from_image_dir(image_dpath, input_fpath):
    import kwcoco
    import kwutil
    import ubelt as ub

    image_dpath = ub.Path(image_dpath)
    input_fpath = ub.Path(input_fpath)
    gpaths = sorted(kwutil.util_path.coerce_patterned_paths(image_dpath, expected_extension=_image_extensions()))
    dset = kwcoco.CocoDataset.from_image_paths(gpaths)
    dset.reroot(absolute=True)
    for img in dset.images().objs:
        img['sensor_coarse'] = 'phone'
        img['channels'] = 'red|green|blue'
        img['name'] = ub.Path(img['file_name']).stem
    dset._update_fpath(input_fpath)
    dset.dump()
    return input_fpath


def coerce_input_kwcoco(src, paths):
    src = paths['src']
    if src.is_dir():
        return build_input_kwcoco_from_image_dir(src, paths['input_fpath']), True
    return src, False


def clone_dataset_for_predictions(src_fpath, pred_fpath):
    import kwcoco
    import ubelt as ub

    pred_fpath = ub.Path(pred_fpath)
    pred_fpath.parent.ensuredir()
    src_dset = kwcoco.CocoDataset.coerce(src_fpath)
    pred_dset = src_dset.copy()
    pred_dset.clear_annotations()
    pred_dset.reroot(absolute=True)
    pred_dset._update_fpath(pred_fpath)
    pred_dset.dataset.setdefault('info', [])
    return pred_dset


def export_predictions_to_labelme(pred_dataset, only_missing=True, score_thresh=0.0):
    import kwcoco
    import kwimage
    from kwcoco.formats.labelme import LabelMeFile

    pred_dset = kwcoco.CocoDataset.coerce(pred_dataset)
    export_dset = pred_dset.copy()
    export_dset.clear_annotations()

    for coco_img in pred_dset.images().coco_images:
        for ann in coco_img.annots().objs:
            score = float(ann.get('score', 1.0))
            if score < score_thresh:
                continue
            segmentation = ann.get('segmentation', None)
            if segmentation is None:
                continue
            try:
                mpoly = kwimage.Segmentation.coerce(segmentation).to_multi_polygon()
                mpoly = mpoly.simplify(1.0)
            except Exception:
                continue
            if not len(mpoly.data):
                continue
            catname = pred_dset.cats[ann['category_id']]['name']
            export_ann = {
                'image_id': coco_img.img['id'],
                'category_id': export_dset.ensure_category(catname),
                'bbox': mpoly.box().to_coco(),
                'segmentation': mpoly.to_coco(style='new'),
                'score': score,
                'role': ann.get('role', 'prediction'),
            }
            export_dset.add_annotation(**export_ann)

    sidecars = list(LabelMeFile.multiple_from_coco(export_dset))
    written = []
    for sidecar in sidecars:
        sidecar.reroot(absolute=False)
        sidecar.fpath = sidecar.fpath.resolve()
        if not sidecar.data['shapes']:
            continue
        if only_missing and sidecar.fpath.exists():
            continue
        sidecar.dump()
        written.append(sidecar.fpath)
    return written
