"""
Export kwcoco datasets to COCO json for external training stacks.
"""

import json
from pathlib import Path


def _build_coco_export(src, dst, category_name='poop', include_segmentations=True, category_id=0):
    import kwcoco
    import kwimage

    src_dset = kwcoco.CocoDataset.coerce(src)
    export = {
        'info': {},
        'images': [],
        'annotations': [],
        'categories': [{
            'id': category_id,
            'name': category_name,
            'supercategory': category_name,
        }],
    }

    kept_gids = set()
    for img in src_dset.images().objs:
        img = img.copy()
        try:
            file_name = src_dset.get_image_fpath(img['id'])
        except Exception:
            bundle_dpath = src_dset.bundle_dpath or '.'
            file_name = Path(bundle_dpath) / img['file_name']
        img['file_name'] = str(Path(file_name).resolve())
        export['images'].append({
            'id': img['id'],
            'file_name': img['file_name'],
            'width': img.get('width', None),
            'height': img.get('height', None),
        })
        kept_gids.add(img['id'])

    ann_id = 1
    for ann in src_dset.annots().objs:
        gid = ann['image_id']
        if gid not in kept_gids:
            continue
        src_category_id = ann.get('category_id', None)
        if src_category_id is None:
            continue
        cat = src_dset.cats.get(src_category_id, None)
        if cat is None:
            continue
        catname = cat['name']
        if catname != category_name:
            continue
        new_ann = {
            'id': ann_id,
            'image_id': gid,
            'category_id': category_id,
            'iscrowd': int(ann.get('iscrowd', 0)),
            'bbox': ann.get('bbox', None),
            'area': float(ann.get('area', 0.0)),
        }
        if new_ann['bbox'] is None and ann.get('segmentation', None) is not None:
            seg = kwimage.Segmentation.coerce(ann['segmentation']).to_multi_polygon()
            new_ann['bbox'] = seg.box().to_coco()
            new_ann['area'] = float(seg.area)
        elif new_ann['bbox'] is not None and not new_ann['area']:
            new_ann['area'] = float(new_ann['bbox'][2] * new_ann['bbox'][3])
        if include_segmentations and ann.get('segmentation', None) is not None:
            new_ann['segmentation'] = ann['segmentation']
            if not new_ann['area']:
                seg = kwimage.Segmentation.coerce(ann['segmentation']).to_multi_polygon()
                new_ann['area'] = float(seg.area)
        export['annotations'].append(new_ann)
        ann_id += 1

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(export))
    return dst


def export_training_splits(train_kwcoco, vali_kwcoco, output_dpath, test_kwcoco=None,
                           category_name='poop', include_segmentations=True, category_id=0):
    output_dpath = Path(output_dpath)
    output_dpath.mkdir(parents=True, exist_ok=True)
    exports = {
        'train': _build_coco_export(
            train_kwcoco,
            output_dpath / 'train.mscoco.json',
            category_name=category_name,
            include_segmentations=include_segmentations,
            category_id=category_id,
        ),
        'vali': _build_coco_export(
            vali_kwcoco,
            output_dpath / 'vali.mscoco.json',
            category_name=category_name,
            include_segmentations=include_segmentations,
            category_id=category_id,
        ),
    }
    if test_kwcoco is not None:
        exports['test'] = _build_coco_export(
            test_kwcoco,
            output_dpath / 'test.mscoco.json',
            category_name=category_name,
            include_segmentations=include_segmentations,
            category_id=category_id,
        )
    return exports
