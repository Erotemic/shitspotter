"""
Merge nearby/overlapping annotations within each image of a kwcoco dataset.

Two annotations are considered "nearby" when their padded bounding boxes
overlap (i.e. IoU of the padded boxes > 0).  Annotations in the same
connected component (transitively nearby) are merged into a single
annotation whose box is the union-hull and whose segmentation is the
polygon union.

This is used as a pre-processing step before evaluation so that a model
that detects and segments a group of nearby objects as one blob is not
penalised as heavily as one that misses them entirely.

Usage (CLI):
    python -m shitspotter.algo_foundation_v3.merge_nearby_anns \\
        --src path/to/input.kwcoco.zip \\
        --dst path/to/merged.kwcoco.zip \\
        --pad_factor 0.5

    pad_factor controls how much each box is grown before testing overlap.
    0.0 = only merge boxes that already overlap.
    0.5 = grow each box by 50% of its own size on each side before testing.
    1.0 = grow by 100% (box doubles in size for proximity test only).
"""

import scriptconfig as scfg
import ubelt as ub


class MergeNearbyAnnsCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path', required=True)
    dst = scfg.Value(None, help='output kwcoco path', required=True)
    pad_factor = scfg.Value(0.5, help=(
        'grow each box by this fraction of its own size before proximity test. '
        '0 = only merge overlapping boxes, 0.5 = also merge boxes within '
        '50%% of their size apart.'))

    @classmethod
    def main(cls, argv=1, **kwargs):
        import kwcoco
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        src_dset = kwcoco.CocoDataset.coerce(config.src)
        dst_dset = merge_nearby_annotations(src_dset, pad_factor=config.pad_factor)
        dst_dset.fpath = config.dst
        dst_dset.dump(config.dst)
        n_before = len(src_dset.anns)
        n_after = len(dst_dset.anns)
        print(f'Merged {n_before} → {n_after} annotations '
              f'({n_before - n_after} merged away) in {src_dset.n_images} images')
        print(f'Wrote {config.dst}')


def _padded_boxes(boxes, pad_factor):
    """Return boxes expanded by pad_factor * their own half-size on each side."""
    import kwimage
    import numpy as np
    ltrb = boxes.to_ltrb().data.copy().astype(float)
    w = ltrb[:, 2] - ltrb[:, 0]
    h = ltrb[:, 3] - ltrb[:, 1]
    pad_x = w * pad_factor
    pad_y = h * pad_factor
    ltrb[:, 0] -= pad_x
    ltrb[:, 1] -= pad_y
    ltrb[:, 2] += pad_x
    ltrb[:, 3] += pad_y
    return kwimage.Boxes(ltrb, 'ltrb')


def _cluster_nearby(boxes, pad_factor):
    """Return cluster labels for boxes via connected-components on padded IoU."""
    import numpy as np
    import scipy.sparse.csgraph

    if len(boxes) == 0:
        return np.array([], dtype=int)
    if len(boxes) == 1:
        return np.array([0], dtype=int)

    padded = _padded_boxes(boxes, pad_factor)
    ious = padded.ious(padded)
    # Two boxes are adjacent if their padded versions overlap at all (iou > 0)
    # Exclude self-loops
    np.fill_diagonal(ious, 0.0)
    adj = (ious > 0).astype(float)
    _, labels = scipy.sparse.csgraph.connected_components(adj, directed=False)
    return labels


def _merge_polygon_list(segmentations):
    """Merge a list of kwcoco-style segmentation dicts into one polygon."""
    import kwimage
    polys = []
    for seg in segmentations:
        try:
            p = kwimage.Polygon.coerce(seg)
            if p is not None:
                polys.append(p)
        except Exception:
            pass
    if not polys:
        return None
    result = polys[0]
    for p in polys[1:]:
        try:
            result = result.union(p)
            # union may return a MultiPolygon — take the largest component
            if hasattr(result, 'geoms'):
                # shapely MultiPolygon
                import kwimage
                largest = max(result.geoms, key=lambda g: g.area)
                result = kwimage.Polygon.from_shapely(largest)
        except Exception:
            pass
    return result


def merge_nearby_annotations(src_dset, pad_factor=0.5):
    """Return a new CocoDataset with nearby annotations per image merged.

    Parameters
    ----------
    src_dset : kwcoco.CocoDataset
    pad_factor : float
        How much to grow each box (as fraction of its own size) before
        testing proximity.

    Returns
    -------
    kwcoco.CocoDataset
        A shallow copy of src_dset with merged annotations.
    """
    import kwcoco
    import kwimage
    import numpy as np

    dst_dset = src_dset.copy()
    dst_dset.remove_annotations(list(dst_dset.anns.keys()))

    for img_id in ub.ProgIter(src_dset.images(), desc='merge nearby anns'):
        ann_ids = src_dset.index.gid_to_aids[img_id]
        if not ann_ids:
            continue

        anns = [src_dset.anns[aid] for aid in ann_ids]

        # Group by category first — only merge within same category
        cat_groups = {}
        for ann in anns:
            cid = ann.get('category_id', None)
            cat_groups.setdefault(cid, []).append(ann)

        for cid, cat_anns in cat_groups.items():
            if not cat_anns:
                continue

            bboxes_xywh = np.array([a['bbox'] for a in cat_anns], dtype=float)
            ltrb = np.column_stack([
                bboxes_xywh[:, 0],
                bboxes_xywh[:, 1],
                bboxes_xywh[:, 0] + bboxes_xywh[:, 2],
                bboxes_xywh[:, 1] + bboxes_xywh[:, 3],
            ])
            boxes = kwimage.Boxes(ltrb, 'ltrb')
            labels = _cluster_nearby(boxes, pad_factor)

            for cluster_id in np.unique(labels):
                mask = labels == cluster_id
                cluster_anns = [a for a, m in zip(cat_anns, mask) if m]

                if len(cluster_anns) == 1:
                    # No merging needed — copy annotation as-is
                    ann = dict(cluster_anns[0])
                    ann.pop('id', None)
                    dst_dset.add_annotation(image_id=img_id, **ann)
                    continue

                # Merge bboxes → union hull
                cluster_ltrb = ltrb[mask]
                merged_l = cluster_ltrb[:, 0].min()
                merged_t = cluster_ltrb[:, 1].min()
                merged_r = cluster_ltrb[:, 2].max()
                merged_b = cluster_ltrb[:, 3].max()
                merged_bbox = [
                    float(merged_l),
                    float(merged_t),
                    float(merged_r - merged_l),
                    float(merged_b - merged_t),
                ]

                # Merge polygons
                segs = [a.get('segmentation', None) for a in cluster_anns
                        if a.get('segmentation', None) is not None]
                merged_segm = None
                if segs:
                    poly = _merge_polygon_list(segs)
                    if poly is not None:
                        merged_segm = poly.to_coco(style='new')

                # Pick the highest-score annotation's metadata as the base
                scores = [float(a.get('score', 0.0)) for a in cluster_anns]
                base_ann = cluster_anns[int(np.argmax(scores))]

                new_ann = {
                    k: v for k, v in base_ann.items()
                    if k not in {'id', 'bbox', 'segmentation', 'area'}
                }
                new_ann['bbox'] = merged_bbox
                new_ann['area'] = float((merged_r - merged_l) * (merged_b - merged_t))
                if merged_segm is not None:
                    new_ann['segmentation'] = merged_segm

                dst_dset.add_annotation(image_id=img_id, **new_ann)

    return dst_dset


__cli__ = MergeNearbyAnnsCLI

if __name__ == '__main__':
    __cli__.main()
