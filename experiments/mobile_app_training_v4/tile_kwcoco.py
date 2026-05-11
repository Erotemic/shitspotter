#!/usr/bin/env python3
"""
Generate a tile-augmented kwcoco bundle for coarse-to-fine detector training.

Why
---
The phone app expects to run in several modes (see docs/000-app-detection-modes
in mobile_app_training_v4/README.md):

* ``FAST_FULL_FRAME``      — full preview frame, model input 320 or 416
* ``BALANCED_FULL_FRAME``  — full 1280x720 preview, model input 640
* ``ROI_HIGH_RES``         — crop a region from a higher-res capture and
                             feed the crop to the model
* ``TILED_SEARCH``         — split a higher-res frame into NxN overlapping
                             tiles and process one tile per frame
* ``FREEZE_FRAME_HIGH_RES`` — high-res still photo, possibly tiled

A detector trained only on the full-image downsampled view (the v3 default)
sees boxes that are quite small relative to the input grid. When the same
detector is later fed a high-res tile, the boxes are *much* larger relative
to the input grid, which is out-of-distribution.

This script materialises a training kwcoco that mixes:

1. The original image, resized so the long side is at most ``--full_dim``.
2. A 2x2 (or NxN) grid of overlapping tiles cut from the original
   high-resolution image, each tile resized so the long side is at most
   ``--tile_output_dim``.

Bounding boxes (and segmentation polygons when present) are warped into the
tile coordinate frame and clipped. Annotations whose intersection with the
tile drops below ``--min_keep_fraction`` of the original area are dropped to
avoid teaching the detector that a half-poop is a full poop.

Outputs
-------
Writes a new kwcoco bundle at ``--dst``. All asset files live under
``<dst-stem>_assets/`` so the bundle is self-contained and can be moved or
copied without breaking image paths.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import scriptconfig as scfg
import ubelt as ub


class TileKwcocoCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path (full-resolution annotations)')
    dst = scfg.Value(None, position=2, help='output kwcoco path (tile-augmented bundle)')

    full_dim = scfg.Value(1280, help='long-side cap for the kept full-frame image')
    tile_grid = scfg.Value(2, help='NxN grid of tiles per image')
    tile_overlap = scfg.Value(0.20, help='fractional overlap between adjacent tiles, 0..0.5')
    tile_output_dim = scfg.Value(640, help='long-side cap for each tile after resize')
    keep_full = scfg.Value(True, help='if True, also emit the resized full image')
    min_keep_fraction = scfg.Value(0.30, help='drop annotations whose visible fraction in a tile is below this')
    output_ext = scfg.Value('.jpg', help='asset extension')
    jpeg_quality = scfg.Value(90, help='JPEG encode quality if .jpg')
    category_name = scfg.Value('poop', help='category name to keep (others dropped)')
    workers = scfg.Value(0, help='reserved; current implementation is serial')
    progress = scfg.Value(True, help='show ProgIter progress')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _tile_extents(width, height, grid, overlap):
    """Yield (x0, y0, x1, y1) extents for an NxN grid of overlapping tiles.

    ``grid`` is the number of tiles per side. ``overlap`` is a fractional
    overlap with the adjacent tile, expressed in *tile* units. Concretely,
    if grid=2 and overlap=0.2 we want each tile to be roughly
    ``ceil(W * (1 + overlap) / 2)`` wide, with the second tile shifted
    so that it overlaps the first by ``overlap`` of its own width.
    """
    grid = max(int(grid), 1)
    overlap = max(min(float(overlap), 0.5), 0.0)
    # Tile dimension in source-pixel units. We want N tiles to cover the
    # full extent with `overlap` fractional overlap between neighbours:
    #   tile_size * N - tile_size * overlap * (N - 1) = full_extent
    # => tile_size = full_extent / (N - overlap * (N - 1))
    if grid == 1:
        return [(0, 0, int(width), int(height))]

    def _axis(extent):
        denom = grid - overlap * (grid - 1)
        tile = extent / denom
        starts = [int(round(i * tile * (1 - overlap))) for i in range(grid)]
        # Make sure the last tile reaches the edge:
        starts[-1] = max(starts[-1], int(extent - tile))
        ends = [min(int(round(s + tile)), int(extent)) for s in starts]
        starts = [max(0, s) for s in starts]
        return list(zip(starts, ends))

    xs = _axis(width)
    ys = _axis(height)
    return [(x0, y0, x1, y1) for (x0, x1) in xs for (y0, y1) in ys]


def _clip_bbox_xywh(bbox, x0, y0, x1, y1, min_keep_fraction):
    """Clip an xywh bbox to a tile and return the warped xywh, or None.

    Returns ``(new_xywh, keep_fraction)`` where ``new_xywh`` is in the
    tile's pixel coordinate frame (i.e. (0, 0) is the top-left of the
    tile). ``None`` is returned when the bbox does not survive the
    ``min_keep_fraction`` test.
    """
    bx, by, bw, bh = [float(v) for v in bbox]
    if bw <= 0 or bh <= 0:
        return None
    src_area = bw * bh
    nx0 = max(bx, x0)
    ny0 = max(by, y0)
    nx1 = min(bx + bw, x1)
    ny1 = min(by + bh, y1)
    new_w = nx1 - nx0
    new_h = ny1 - ny0
    if new_w <= 1 or new_h <= 1:
        return None
    keep = (new_w * new_h) / src_area
    if keep < min_keep_fraction:
        return None
    return [nx0 - x0, ny0 - y0, new_w, new_h], keep


def _resize_with_long_side(image, max_dim):
    import kwimage

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_dim:
        return image, 1.0
    scale = max_dim / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Pass dsize= explicitly. kwimage.imresize's positional API is
    # (img, scale=...), NOT (img, dsize=...) — passing the (W, H)
    # tuple positionally is interpreted as a scale factor, which on a
    # 4032x3024 phone image asks OpenCV to allocate a multi-TB buffer.
    resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation='area')
    return resized, scale


def _imwrite(fpath, image, ext, jpeg_quality):
    import kwimage
    kwargs = {}
    if ext.lower() in ('.jpg', '.jpeg'):
        kwargs['imwrite_params'] = [
            ('JPEG_QUALITY', int(jpeg_quality)),
        ]
    kwimage.imwrite(str(fpath), image, **kwargs)


def run(config):
    import kwcoco

    src_fpath = Path(str(config.src)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()
    if not src_fpath.exists():
        raise FileNotFoundError(src_fpath)

    src_dset = kwcoco.CocoDataset.coerce(src_fpath)

    asset_dname = dst_fpath.stem.replace('.kwcoco', '') + '_assets'
    asset_dpath = dst_fpath.parent / asset_dname
    asset_dpath.mkdir(parents=True, exist_ok=True)

    out = {
        'info': [{
            'name': 'mobile_app_training_v4.tile_kwcoco',
            'src': str(src_fpath),
            'config': {k: getattr(config, k) for k in [
                'full_dim', 'tile_grid', 'tile_overlap', 'tile_output_dim',
                'keep_full', 'min_keep_fraction', 'output_ext',
                'jpeg_quality', 'category_name',
            ]},
        }],
        'categories': [],
        'images': [],
        'annotations': [],
    }

    target_cat_name = str(config.category_name)
    src_cats_by_id = {c['id']: c for c in src_dset.dataset.get('categories', [])}
    keep_cat_ids = {cid for cid, cat in src_cats_by_id.items() if cat['name'] == target_cat_name}
    if not keep_cat_ids:
        raise RuntimeError(f'No category named {target_cat_name!r} in {src_fpath}')
    out['categories'].append({
        'id': 1,
        'name': target_cat_name,
        'supercategory': target_cat_name,
    })

    next_gid = 1
    next_ann_id = 1

    coco_imgs = list(src_dset.images().coco_images)
    iterator = ub.ProgIter(coco_imgs, desc='tile kwcoco', enabled=bool(config.progress))
    for coco_img in iterator:
        try:
            image = coco_img.imdelay().finalize()
        except Exception as ex:
            print(f'  warn: failed to read {coco_img.img.get("file_name")}: {ex}')
            continue

        h, w = image.shape[:2]
        gid = coco_img.img['id']
        anns = [
            ann for ann in src_dset.annots(gid=gid).objs
            if ann.get('category_id') in keep_cat_ids
            and ann.get('bbox') is not None
        ]

        # ---- emit downsized full frame (the BALANCED_FULL_FRAME mode) ----
        if bool(config.keep_full):
            full_resized, scale = _resize_with_long_side(image, int(config.full_dim))
            stem = f'gid{gid:08d}_full'
            asset_fname = stem + str(config.output_ext)
            asset_fpath = asset_dpath / asset_fname
            _imwrite(asset_fpath, full_resized, str(config.output_ext), int(config.jpeg_quality))
            out_h, out_w = full_resized.shape[:2]
            out['images'].append({
                'id': next_gid,
                'file_name': str(asset_fpath.relative_to(dst_fpath.parent)),
                'width': int(out_w),
                'height': int(out_h),
                'name': stem,
                'tile_role': 'full',
                'tile_source_gid': int(gid),
            })
            for ann in anns:
                bx, by, bw, bh = ann['bbox']
                new_bbox = [bx * scale, by * scale, bw * scale, bh * scale]
                out['annotations'].append({
                    'id': next_ann_id,
                    'image_id': next_gid,
                    'category_id': 1,
                    'bbox': new_bbox,
                    'area': float(new_bbox[2] * new_bbox[3]),
                    'iscrowd': int(ann.get('iscrowd', 0)),
                })
                next_ann_id += 1
            next_gid += 1

        # ---- emit overlapping tiles (the TILED_SEARCH / ROI mode) ----
        extents = _tile_extents(w, h, int(config.tile_grid), float(config.tile_overlap))
        for tile_idx, (x0, y0, x1, y1) in enumerate(extents):
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            tile_image = image[y0:y1, x0:x1]
            tile_resized, scale = _resize_with_long_side(
                tile_image, int(config.tile_output_dim))
            stem = f'gid{gid:08d}_tile{tile_idx:02d}_g{int(config.tile_grid)}'
            asset_fname = stem + str(config.output_ext)
            asset_fpath = asset_dpath / asset_fname
            _imwrite(asset_fpath, tile_resized, str(config.output_ext), int(config.jpeg_quality))
            out_h, out_w = tile_resized.shape[:2]
            out['images'].append({
                'id': next_gid,
                'file_name': str(asset_fpath.relative_to(dst_fpath.parent)),
                'width': int(out_w),
                'height': int(out_h),
                'name': stem,
                'tile_role': 'tile',
                'tile_source_gid': int(gid),
                'tile_extent_xyxy': [int(x0), int(y0), int(x1), int(y1)],
                'tile_resize_scale': float(scale),
            })

            kept = 0
            for ann in anns:
                clipped = _clip_bbox_xywh(
                    ann['bbox'], x0, y0, x1, y1,
                    float(config.min_keep_fraction))
                if clipped is None:
                    continue
                new_bbox, keep = clipped
                new_bbox = [v * scale for v in new_bbox]
                out['annotations'].append({
                    'id': next_ann_id,
                    'image_id': next_gid,
                    'category_id': 1,
                    'bbox': new_bbox,
                    'area': float(new_bbox[2] * new_bbox[3]),
                    'iscrowd': int(ann.get('iscrowd', 0)),
                    'tile_keep_fraction': float(keep),
                })
                next_ann_id += 1
                kept += 1
            # Track empty tiles in the image record so trainers can
            # decide whether to oversample non-empty tiles.
            out['images'][-1]['tile_num_kept_anns'] = kept
            next_gid += 1

    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    if dst_fpath.suffix == '.zip':
        # write json next to the zip then zip-bundle it
        json_fpath = dst_fpath.with_suffix('.json')
        json_fpath.write_text(json.dumps(out))
        # Use kwcoco's own bundling rather than rolling our own zip
        import kwcoco as _kw
        dset = _kw.CocoDataset.coerce(json_fpath)
        dset.fpath = str(dst_fpath)
        dset.dump()
        # Optional cleanup of intermediate json
        json_fpath.unlink(missing_ok=True)
    else:
        dst_fpath.write_text(json.dumps(out))

    print(f'wrote {len(out["images"])} images, {len(out["annotations"])} annotations '
          f'to {dst_fpath}')


__cli__ = TileKwcocoCLI


if __name__ == '__main__':
    __cli__.main()
