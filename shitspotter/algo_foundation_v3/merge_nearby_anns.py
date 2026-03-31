"""
Merge nearby/overlapping annotations within each image of a kwcoco dataset.

Uses the same ``find_low_overlap_covering_boxes`` algorithm as
``shitspotter.cli.simplify_kwcoco``, but without removing rare categories or
empty images — both of which would corrupt evaluation results.

Two annotations are considered "nearby" when a scaled version of their
bounding boxes overlap.  Annotations in the same connected component
(transitively nearby) are merged into a single annotation whose box is the
union of the merged polygons and whose segmentation is the shapely unary_union.

Usage (CLI):
    python -m shitspotter.algo_foundation_v3.merge_nearby_anns \\
        --src path/to/input.kwcoco.zip \\
        --dst path/to/merged.kwcoco.zip \\
        --scale 1.5

    scale controls how much each box is grown before testing overlap.
    1.0 = only merge boxes that already overlap.
    1.5 = grow each box by 50% (the default, matching simplify_kwcoco).
    2.0 = grow by 100%.
"""

import scriptconfig as scfg
import ubelt as ub


class MergeNearbyAnnsCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path', required=True)
    dst = scfg.Value(None, help='output kwcoco path', required=True)
    scale = scfg.Value(1.5, help=(
        'scale factor applied to each box before proximity test. '
        'Matches the default used in simplify_kwcoco (scale=1.5).'))

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        src_dset = kwcoco.CocoDataset.coerce(config.src)
        dst_dset = merge_nearby_annotations(src_dset, scale=config.scale)
        dst_dset.fpath = config.dst
        dst_dset.dump(config.dst)
        n_before = len(src_dset.anns)
        n_after = len(dst_dset.anns)
        print(f'Merged {n_before} → {n_after} annotations '
              f'({n_before - n_after} merged away) '
              f'across {src_dset.n_images} images')
        print(f'Wrote {config.dst}')


def merge_nearby_annotations(src_dset, scale=1.5):
    """Return a new CocoDataset with nearby annotations per image merged.

    Uses ``find_low_overlap_covering_boxes`` (same as simplify_kwcoco) to
    identify clusters, then merges each cluster's polygons with shapely
    unary_union.  All images are preserved — even ones with no annotations —
    so the result is safe to use as input to ``kwcoco eval``.

    Parameters
    ----------
    src_dset : kwcoco.CocoDataset
    scale : float
        Box scale factor for the proximity test (default 1.5, same as
        simplify_kwcoco).

    Returns
    -------
    kwcoco.CocoDataset
    """
    import kwimage
    import numpy as np
    from shapely.ops import unary_union
    from geowatch.utils.util_kwimage import find_low_overlap_covering_boxes

    dst_dset = src_dset.copy()
    dst_dset.remove_annotations(list(dst_dset.anns.keys()))

    to_add = []

    for coco_image in ub.ProgIter(
            src_dset.images().coco_images_iter(),
            total=src_dset.n_images,
            desc='merge nearby anns'):
        annots = coco_image.annots()
        if len(annots) == 0:
            continue

        dets = annots.detections
        if len(dets) == 0:
            continue

        min_dim = dets.boxes.to_xywh().data[..., 2:4].min()
        max_dim = dets.boxes.to_xywh().data[..., 2:4].max()

        result = find_low_overlap_covering_boxes(
            dets.boxes.to_polygons(),
            scale,
            min_box_dim=min_dim,
            max_box_dim=max_dim * 1.5,
            verbose=0,
        )
        _new_boxes, assignment = result

        for idxs in assignment:
            if len(idxs) == 1:
                # Single annotation — copy as-is
                ann = dict(annots.objs[idxs[0]])
                ann['image_id'] = coco_image.img['id']
                ann.pop('id', None)
                to_add.append(ann)
            else:
                # Merge cluster
                if not ub.allsame(
                        np.array(annots.category_id)[idxs].tolist()):
                    # Different categories — don't merge, copy individually
                    for idx in idxs:
                        ann = dict(annots.objs[idx])
                        ann['image_id'] = coco_image.img['id']
                        ann.pop('id', None)
                        to_add.append(ann)
                    continue

                polys = [p.to_shapely()
                         for p in ub.take(dets.data['segmentations'], idxs)]
                merged_shape = unary_union(polys)
                merged_poly = kwimage.MultiPolygon.coerce(merged_shape)

                base = ub.udict(annots.objs[idxs[0]]) - {'id', 'segmentation', 'bbox', 'area'}
                base['image_id'] = coco_image.img['id']
                base['bbox'] = merged_poly.box().to_coco()
                base['segmentation'] = merged_poly.to_coco(style='new')
                base['area'] = float(merged_poly.box().area)
                to_add.append(dict(base))

    for ann in to_add:
        dst_dset.add_annotation(**ann)

    return dst_dset


__cli__ = MergeNearbyAnnsCLI

if __name__ == '__main__':
    __cli__.main()
