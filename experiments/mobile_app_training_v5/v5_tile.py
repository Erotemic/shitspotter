#!/usr/bin/env python3
"""
v5 multi-scale fixed-size tile extractor.

The key difference from v4's tile_kwcoco.py:

  v4: tile of variable resolution from each image — one tile is "the
      top-left quadrant of the source image". Tile aspect/area depends
      on the source dimensions. After tile extraction, every tile is
      resized to fit a long-side cap, but its content scale is fixed
      by where it was cut from.

  v5: tile at *fixed output resolution* (e.g. 320x320) from each of
      several pre-downscaled copies of the source image. One tile is
      "a 320x320 patch from the s=0.40 version of this source image".
      The same physical object can appear in multiple tiles at
      different apparent sizes, by virtue of being cropped from
      different source scales. This is the closest data-time mirror
      of multi-resolution inference at the detector.

Output kwcoco
-------------
Each tile becomes one image. The image dict carries:

  tile_source_gid                  source image's gid in --src
  tile_scale_name                  e.g. "s10", "s07", "s04", "s02"
  tile_scale_factor                float, e.g. 1.0, 0.66, 0.40, 0.25
  tile_extent_xyxy_in_source       (x0, y0, x1, y1) in ORIGINAL source pixels
  tile_role                        "positive" | "negative"
  tile_num_kept_anns               number of GT anns surviving in this tile

Annotations are warped through (source_scale * crop_offset) and
clipped to the tile. A tile is "positive" if the total surviving GT
area exceeds --min_gt_area_frac of the tile area; otherwise
"negative".
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import scriptconfig as scfg
import ubelt as ub


DEFAULT_SCALES = '1.0,0.66,0.40,0.25'


class V5TileCLI(scfg.DataConfig):
    src = scfg.Value(None, position=1, help='input kwcoco path (full-resolution annotations)')
    dst = scfg.Value(None, position=2, help='output kwcoco bundle (tile-augmented)')

    tile_size = scfg.Value(320, help='fixed output tile size in pixels (square)')
    source_scales = scfg.Value(
        DEFAULT_SCALES,
        help='comma-separated list of source-scale factors to extract tiles from')
    stride_frac = scfg.Value(0.5, help='sliding-window stride as a fraction of tile_size')
    min_gt_area_frac = scfg.Value(0.005,
        help='tile is "positive" iff surviving GT area / tile_area >= this')
    min_kept_box_frac = scfg.Value(0.30,
        help='annotations whose visible fraction in a tile is below this are dropped')
    min_source_scale_long_side = scfg.Value(64,
        help='skip a source scale whose long side falls below this; '
             'protects against downscaling so far the image has no useful content')
    keep_negative = scfg.Value(True,
        help='if True, also emit negative tiles for hard-neg mining; '
             'if False, only emit positive tiles (smaller bundle)')
    output_ext = scfg.Value('.jpg', help='asset extension')
    jpeg_quality = scfg.Value(90, help='JPEG quality if .jpg')
    category_name = scfg.Value('poop', help='category name to keep')
    seed = scfg.Value(0, help='RNG seed for any sampling')
    progress = scfg.Value(True, help='show ProgIter')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def _parse_scales(scales_str: str) -> List[Tuple[str, float]]:
    """'1.0,0.66,0.4' -> [('s10', 1.0), ('s07', 0.66), ('s04', 0.4)]."""
    out: List[Tuple[str, float]] = []
    for tok in str(scales_str).split(','):
        tok = tok.strip()
        if not tok:
            continue
        s = float(tok)
        if s <= 0 or s > 4.0:
            raise ValueError(f'scale {s} out of plausible range (0, 4]')
        # Compact name: s10, s07, s04, s02
        out.append((f's{int(round(s * 10)):02d}', s))
    if not out:
        raise ValueError('no scales parsed')
    return out


def _clip_bbox_xywh(bbox, x0, y0, x1, y1, min_keep_fraction):
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


def _resize_image_to_scale(image, scale: float):
    """Downscale image by `scale` using kwimage. Returns (resized, actual_scale)."""
    import kwimage

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image, 1.0
    try:
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation='area')
    except NotImplementedError:
        # skimage backend doesn't recognise 'area'
        resized = kwimage.imresize(image, dsize=(new_w, new_h), interpolation='linear')
    # actual_scale may differ from requested due to int rounding
    return resized, new_w / float(w)


def _grid_positions(extent: int, tile: int, stride: int) -> List[int]:
    """Sliding-window start positions covering [0, extent), inclusive of edges.

    Always emits 0 (left edge) and extent - tile (right edge if tile fits),
    plus regularly spaced intermediate positions. Avoids duplicate edge
    when stride happens to land on extent - tile.
    """
    if extent <= tile:
        return [0]  # one tile spanning the whole extent (will be letterboxed)
    positions = list(range(0, extent - tile + 1, max(stride, 1)))
    last = extent - tile
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _imwrite(fpath: Path, image, ext: str, jpeg_quality: int):
    import kwimage
    if ext.lower() in ('.jpg', '.jpeg'):
        import cv2
        kwimage.imwrite(
            str(fpath), image,
            params=[int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
    else:
        kwimage.imwrite(str(fpath), image)


def run(config):
    import kwcoco
    import kwimage
    import numpy as np

    src_fpath = Path(str(config.src)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()
    if not src_fpath.exists():
        raise FileNotFoundError(src_fpath)

    src_dset = kwcoco.CocoDataset.coerce(src_fpath)

    asset_dname = dst_fpath.stem.replace('.kwcoco', '') + '_assets'
    asset_dpath = dst_fpath.parent / asset_dname
    asset_dpath.mkdir(parents=True, exist_ok=True)

    scales = _parse_scales(str(config.source_scales))
    tile_size = int(config.tile_size)
    stride = max(1, int(round(tile_size * float(config.stride_frac))))
    min_long_side = int(config.min_source_scale_long_side)
    min_gt_area_abs = float(config.min_gt_area_frac) * (tile_size * tile_size)

    target_cat_name = str(config.category_name)
    src_cats_by_id = {c['id']: c for c in src_dset.dataset.get('categories', [])}
    keep_cat_ids = {cid for cid, cat in src_cats_by_id.items()
                    if cat['name'] == target_cat_name}
    if not keep_cat_ids:
        raise RuntimeError(f'No category named {target_cat_name!r} in {src_fpath}')

    out = {
        'info': [{
            'name': 'mobile_app_training_v5.v5_tile',
            'src': str(src_fpath),
            'config': {k: getattr(config, k) for k in [
                'tile_size', 'source_scales', 'stride_frac',
                'min_gt_area_frac', 'min_kept_box_frac',
                'min_source_scale_long_side', 'keep_negative',
                'output_ext', 'jpeg_quality', 'category_name',
            ]},
        }],
        'categories': [{'id': 1, 'name': target_cat_name,
                        'supercategory': target_cat_name}],
        'images': [],
        'annotations': [],
    }

    next_gid = 1
    next_ann_id = 1
    n_pos = 0
    n_neg = 0
    n_neg_kept = 0

    coco_imgs = list(src_dset.images().coco_images)
    iterator = ub.ProgIter(coco_imgs, desc='v5 multi-scale tile',
                           enabled=bool(config.progress))

    for coco_img in iterator:
        try:
            image_full = coco_img.imdelay().finalize()
        except Exception as ex:
            print(f'  warn: failed to read {coco_img.img.get("file_name")}: {ex}')
            continue
        if image_full.ndim == 2:
            image_full = np.repeat(image_full[..., None], 3, axis=-1)
        if image_full.shape[2] == 4:
            image_full = image_full[..., :3]

        H, W = image_full.shape[:2]
        gid = coco_img.img['id']
        anns_src = [
            ann for ann in src_dset.annots(gid=gid).objs
            if ann.get('category_id') in keep_cat_ids and ann.get('bbox') is not None
        ]

        for scale_name, scale_factor in scales:
            # Skip the scale if it would produce a too-tiny image.
            scaled_long = max(int(round(W * scale_factor)),
                              int(round(H * scale_factor)))
            if scaled_long < min_long_side:
                continue
            scaled_img, actual_scale = _resize_image_to_scale(image_full, scale_factor)
            sH, sW = scaled_img.shape[:2]

            # Build a list of (x0, y0) starts.
            xs = _grid_positions(sW, tile_size, stride)
            ys = _grid_positions(sH, tile_size, stride)

            # Pre-warp source annotations into scaled-image coords.
            anns_scaled = []
            for ann in anns_src:
                bx, by, bw, bh = ann['bbox']
                anns_scaled.append({
                    'bbox': [bx * actual_scale, by * actual_scale,
                             bw * actual_scale, bh * actual_scale],
                    'iscrowd': int(ann.get('iscrowd', 0)),
                    'src_ann_id': ann.get('id'),
                })

            for x0 in xs:
                for y0 in ys:
                    x1 = min(x0 + tile_size, sW)
                    y1 = min(y0 + tile_size, sH)
                    crop = scaled_img[y0:y1, x0:x1]

                    # Pad to (tile_size, tile_size) if at edge.
                    if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                        pad = np.zeros((tile_size, tile_size, 3), dtype=crop.dtype)
                        pad[:crop.shape[0], :crop.shape[1]] = crop
                        crop = pad

                    # Warp annotations into tile coords + classify pos/neg.
                    kept_anns = []
                    total_kept_area = 0.0
                    for ann in anns_scaled:
                        clipped = _clip_bbox_xywh(
                            ann['bbox'], x0, y0, x0 + tile_size, y0 + tile_size,
                            float(config.min_kept_box_frac),
                        )
                        if clipped is None:
                            continue
                        new_bbox, _keep = clipped
                        kept_anns.append({
                            'bbox': new_bbox,
                            'iscrowd': ann['iscrowd'],
                            'src_ann_id': ann['src_ann_id'],
                        })
                        total_kept_area += new_bbox[2] * new_bbox[3]

                    is_positive = total_kept_area >= min_gt_area_abs
                    if not is_positive and not bool(config.keep_negative):
                        continue

                    role = 'positive' if is_positive else 'negative'

                    # Map tile-extent back into ORIGINAL source pixels for
                    # downstream debugging.
                    src_x0 = int(round(x0 / max(actual_scale, 1e-6)))
                    src_y0 = int(round(y0 / max(actual_scale, 1e-6)))
                    src_x1 = int(round((x0 + tile_size) / max(actual_scale, 1e-6)))
                    src_y1 = int(round((y0 + tile_size) / max(actual_scale, 1e-6)))

                    stem = (f'gid{gid:08d}_{scale_name}'
                            f'_x{x0:05d}_y{y0:05d}_{role}')
                    asset_fname = stem + str(config.output_ext)
                    asset_fpath = asset_dpath / asset_fname
                    _imwrite(asset_fpath, crop, str(config.output_ext),
                             int(config.jpeg_quality))

                    out['images'].append({
                        'id': next_gid,
                        'file_name': str(asset_fpath.relative_to(dst_fpath.parent)),
                        'width': tile_size,
                        'height': tile_size,
                        'name': stem,
                        'tile_source_gid': int(gid),
                        'tile_scale_name': scale_name,
                        'tile_scale_factor': float(scale_factor),
                        'tile_actual_scale': float(actual_scale),
                        'tile_extent_xyxy_in_source': [src_x0, src_y0, src_x1, src_y1],
                        'tile_role': role,
                        'tile_num_kept_anns': len(kept_anns),
                    })
                    for ann in kept_anns:
                        out['annotations'].append({
                            'id': next_ann_id,
                            'image_id': next_gid,
                            'category_id': 1,
                            'bbox': ann['bbox'],
                            'area': float(ann['bbox'][2] * ann['bbox'][3]),
                            'iscrowd': ann['iscrowd'],
                            'src_ann_id': ann['src_ann_id'],
                        })
                        next_ann_id += 1
                    next_gid += 1
                    if is_positive:
                        n_pos += 1
                    else:
                        n_neg += 1
                        n_neg_kept += 1

    # Write the bundle.
    dst_fpath.parent.mkdir(parents=True, exist_ok=True)
    if dst_fpath.suffix == '.zip':
        json_fpath = dst_fpath.with_suffix('.json')
        json_fpath.write_text(json.dumps(out))
        import kwcoco as _kw
        dset = _kw.CocoDataset.coerce(json_fpath)
        dset.fpath = str(dst_fpath)
        dset.dump()
        json_fpath.unlink(missing_ok=True)
    else:
        dst_fpath.write_text(json.dumps(out))

    print(f'wrote {len(out["images"])} tiles '
          f'(positive={n_pos}, negative={n_neg_kept}, dropped_neg={n_neg - n_neg_kept})')
    print(f'  annotations: {len(out["annotations"])}')
    print(f'  scales: {", ".join(f"{n}={s}" for n, s in scales)}')
    print(f'  -> {dst_fpath}')


__cli__ = V5TileCLI


if __name__ == '__main__':
    __cli__.main()
