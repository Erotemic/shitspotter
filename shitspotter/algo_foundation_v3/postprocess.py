"""
Shared postprocess for detector + segmenter and direct mask baselines.
"""

import numpy as np

from shitspotter.algo_foundation_v3.polygon_utils import (
    expand_box_ltrb,
    mask_to_multi_polygon,
    segmentation_to_coco,
)


def _normalize_label(raw_label, label_mapping):
    for key in [raw_label, str(raw_label)]:
        if key in label_mapping:
            return label_mapping[key]
    return None


def apply_box_filters(records, score_thresh, nms_thresh):
    import kwimage

    filtered = [record for record in records if float(record.get('score', 0.0)) >= score_thresh]
    if not filtered:
        return []
    boxes = kwimage.Boxes(np.array([record['bbox_ltrb'] for record in filtered], dtype=float), 'ltrb')
    scores = np.array([float(record['score']) for record in filtered], dtype=float)
    dets = kwimage.Detections(boxes=boxes, scores=scores, classes=['poop'])
    dets.data['record_idxs'] = np.arange(len(filtered))
    if nms_thresh is not None and nms_thresh > 0:
        dets = dets.non_max_supress(thresh=nms_thresh)
    keep_idxs = dets.data['record_idxs'].tolist()
    return [filtered[idx] for idx in keep_idxs]


def detector_records_to_bbox_anns(detector_records, label_mapping, post_cfg):
    anns = []
    records = apply_box_filters(
        detector_records,
        score_thresh=post_cfg['score_thresh'],
        nms_thresh=post_cfg['nms_thresh'],
    )
    for record in records:
        category_name = _normalize_label(record.get('label', 0), label_mapping)
        if category_name is None:
            continue
        x1, y1, x2, y2 = map(float, record['bbox_ltrb'])
        anns.append({
            'bbox': [x1, y1, x2 - x1, y2 - y1],
            'score': float(record['score']),
            'category_name': category_name,
        })
    return anns


def detector_records_to_anns(image, detector_records, segmenter, label_mapping, post_cfg):
    anns = []
    image_shape = image.shape
    records = apply_box_filters(
        detector_records,
        score_thresh=post_cfg['score_thresh'],
        nms_thresh=post_cfg['nms_thresh'],
    )
    if not records:
        return anns

    padded_boxes = [
        expand_box_ltrb(record['bbox_ltrb'], post_cfg['crop_padding'], image_shape)
        for record in records
    ]
    mask_infos = segmenter.predict_masks_for_boxes(image, padded_boxes)

    for record, prompt_box_ltrb, mask_info in zip(records, padded_boxes, mask_infos):
        category_name = _normalize_label(record.get('label', 0), label_mapping)
        if category_name is None:
            continue
        mpoly = mask_to_multi_polygon(
            mask_info['mask'],
            polygon_simplify=post_cfg['polygon_simplify'],
            min_component_area=post_cfg['min_component_area'],
            keep_largest_component=post_cfg['keep_largest_component'],
        )
        if not len(mpoly.data):
            continue
        anns.append({
            'bbox': mpoly.box().to_coco(),
            'segmentation': segmentation_to_coco(mpoly),
            'score': float(record['score']),
            'category_name': category_name,
            'foundation_prompt_source': 'detector_box',
            'detector_bbox': [
                float(record['bbox_ltrb'][0]),
                float(record['bbox_ltrb'][1]),
                float(record['bbox_ltrb'][2] - record['bbox_ltrb'][0]),
                float(record['bbox_ltrb'][3] - record['bbox_ltrb'][1]),
            ],
            'prompt_bbox': [
                float(prompt_box_ltrb[0]),
                float(prompt_box_ltrb[1]),
                float(prompt_box_ltrb[2] - prompt_box_ltrb[0]),
                float(prompt_box_ltrb[3] - prompt_box_ltrb[1]),
            ],
        })
    return anns


def mask_records_to_anns(mask_records, label_mapping, post_cfg):
    anns = []
    records = [record for record in mask_records if float(record.get('score', 0.0)) >= post_cfg['score_thresh']]
    if not records:
        return anns

    import kwimage
    boxes = kwimage.Boxes(np.array([record['bbox_ltrb'] for record in records], dtype=float), 'ltrb')
    scores = np.array([float(record['score']) for record in records], dtype=float)
    dets = kwimage.Detections(boxes=boxes, scores=scores, classes=['poop'])
    dets.data['record_idxs'] = np.arange(len(records))
    if post_cfg['nms_thresh'] is not None and post_cfg['nms_thresh'] > 0:
        dets = dets.non_max_supress(thresh=post_cfg['nms_thresh'])
    records = [records[idx] for idx in dets.data['record_idxs'].tolist()]

    for record in records:
        category_name = _normalize_label(record.get('label', 0), label_mapping)
        if category_name is None:
            continue
        mpoly = mask_to_multi_polygon(
            record['mask'],
            polygon_simplify=post_cfg['polygon_simplify'],
            min_component_area=post_cfg['min_component_area'],
            keep_largest_component=post_cfg['keep_largest_component'],
        )
        if not len(mpoly.data):
            continue
        anns.append({
            'bbox': mpoly.box().to_coco(),
            'segmentation': segmentation_to_coco(mpoly),
            'score': float(record['score']),
            'category_name': category_name,
        })
    return anns


def add_prediction_annotations(pred_dset, image_id, anns, backend_name):
    for ann in anns:
        ann = ann.copy()
        category_name = ann.pop('category_name')
        ann['image_id'] = image_id
        ann['category_id'] = pred_dset.ensure_category(category_name)
        ann['role'] = 'prediction'
        ann['foundation_backend'] = backend_name
        pred_dset.add_annotation(**ann)
