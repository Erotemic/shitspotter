"""
Polygon and mask helpers.
"""

import numpy as np


def expand_box_ltrb(box_ltrb, padding, image_shape):
    x1, y1, x2, y2 = map(float, box_ltrb)
    height, width = image_shape[0:2]
    x1 = max(0.0, x1 - padding)
    y1 = max(0.0, y1 - padding)
    x2 = min(float(width), x2 + padding)
    y2 = min(float(height), y2 + padding)
    return [x1, y1, x2, y2]


def _safe_polygon_area(poly):
    try:
        return float(poly.area)
    except Exception:
        return 0.0


def mask_to_multi_polygon(mask, polygon_simplify=0.0, min_component_area=0.0,
                          keep_largest_component=True):
    import kwimage

    bool_mask = np.asarray(mask).astype(bool)
    mpoly = kwimage.Mask.coerce(bool_mask).to_multi_polygon()
    polygons = list(mpoly.data)
    polygons = [
        poly for poly in polygons
        if _safe_polygon_area(poly) > 0 and _safe_polygon_area(poly) >= min_component_area
    ]
    if keep_largest_component and polygons:
        polygons = [max(polygons, key=_safe_polygon_area)]
    if polygon_simplify:
        polygons = [poly.simplify(polygon_simplify) for poly in polygons]
        polygons = [poly for poly in polygons if _safe_polygon_area(poly) > 0]
    return kwimage.MultiPolygon(polygons)


def segmentation_to_coco(segmentation):
    if segmentation is None:
        return None
    if hasattr(segmentation, 'to_coco'):
        return segmentation.to_coco(style='new')
    return segmentation
