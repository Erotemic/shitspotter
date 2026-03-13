#!/usr/bin/env python3
"""
Resize images in a kwcoco bundle and warp annotations to match.

The main use case is offline preprocessing of very large source imagery into a
smaller training bundle, so downstream trainers do not need to decode full-size
phone images only to immediately resize them in-memory.
"""
from pathlib import Path

import scriptconfig as scfg
import ubelt as ub


def _coerce_scale_and_size(old_w, old_h, max_dim=None, scale=None, min_dim=None,
                           allow_upscale=False):
    if scale is not None:
        sx = sy = float(scale)
    else:
        candidates = []
        if max_dim is not None:
            long_side = max(old_w, old_h)
            if long_side > 0:
                candidates.append(float(max_dim) / float(long_side))
        if min_dim is not None:
            short_side = min(old_w, old_h)
            if short_side > 0:
                candidates.append(float(min_dim) / float(short_side))
        if not candidates:
            raise ValueError('Specify scale, max_dim, or min_dim')
        sx = sy = min(candidates)
    if not allow_upscale:
        sx = min(sx, 1.0)
        sy = min(sy, 1.0)
    new_w = max(int(round(old_w * sx)), 1)
    new_h = max(int(round(old_h * sy)), 1)
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)
    return sx, sy, new_w, new_h


def _safe_rel_image_name(img, gid, suffix, asset_dname):
    file_name = img.get('file_name', '')
    path = Path(file_name)
    if file_name and not path.is_absolute() and '..' not in path.parts:
        return (Path(asset_dname) / path).as_posix()
    stem = img.get('name', None) or f'image_{gid:08d}'
    return (Path(asset_dname) / f'{stem}{suffix}').as_posix()


def _warp_annotation(ann, transform, input_dims, output_dims):
    import kwimage

    bbox = ann.get('bbox', None)
    if bbox is not None:
        ann['bbox'] = list(kwimage.Boxes([bbox], 'xywh').warp(
            transform,
            input_dims=input_dims,
            output_dims=output_dims,
        ).to_coco())[0]

    segmentation = ann.get('segmentation', None)
    if segmentation is not None:
        seg = kwimage.Segmentation.coerce(segmentation, dims=input_dims)
        ann['segmentation'] = seg.warp(
            transform,
            input_dims=input_dims,
            output_dims=output_dims,
        ).to_coco(style='new')

    keypoints = ann.get('keypoints', None)
    if keypoints is not None:
        pts = kwimage.Points.from_coco(keypoints)
        ann['keypoints'] = pts.warp(
            transform,
            input_dims=input_dims,
            output_dims=output_dims,
        ).to_coco(style='orig')

    area = ann.get('area', None)
    if area is not None:
        ann['area'] = float(area) * float(transform.matrix[0, 0]) * float(transform.matrix[1, 1])
    return ann


def resize_kwcoco(src, dst, max_dim=None, scale=None, min_dim=None,
                  asset_dname='resized_assets', output_ext=None,
                  interpolation='linear', allow_upscale=False):
    """
    Resize a kwcoco dataset into a new bundle.

    Example:
        >>> from shitspotter.cli.resize_kwcoco import resize_kwcoco
        >>> import kwcoco
        >>> import ubelt as ub
        >>> dpath = ub.Path.appdir('shitspotter/tests/resize_kwcoco').ensuredir()
        >>> src_dset = kwcoco.CocoDataset.demo('vidshapes8', dpath=dpath / 'src_demo')
        >>> dst = dpath / 'resized.kwcoco.json'
        >>> new_dset = resize_kwcoco(src_dset, dst=dst, max_dim=128, asset_dname='images_128')
        >>> assert new_dset.fpath == dst
        >>> assert new_dset.n_images == src_dset.n_images
        >>> gid = new_dset.images()[0]
        >>> old = src_dset.index.imgs[gid]
        >>> new = new_dset.index.imgs[gid]
        >>> assert max(new['width'], new['height']) <= 128
        >>> assert new['width'] <= old['width']
        >>> assert new['height'] <= old['height']
        >>> ann = new_dset.annots(gid=gid).objs[0]
        >>> assert ann['bbox'][2] <= old['width']
        >>> assert (dst.parent / new['file_name']).exists()
    """
    import kwcoco
    import kwimage

    src_dset = kwcoco.CocoDataset.coerce(src)
    dst = Path(dst).expanduser().resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    new_dset = src_dset.copy()
    new_dset.fpath = dst
    new_dset.bundle_dpath = str(dst.parent)

    for gid, img in ub.ProgIter(list(new_dset.index.imgs.items()), desc='resize kwcoco'):
        src_img = src_dset.index.imgs[gid]
        old_w = int(src_img.get('width', 0) or 0)
        old_h = int(src_img.get('height', 0) or 0)
        if old_w <= 0 or old_h <= 0:
            shape = kwimage.load_image_shape(src_dset.get_image_fpath(gid))
            old_h, old_w = map(int, shape[0:2])
        sx, sy, new_w, new_h = _coerce_scale_and_size(
            old_w=old_w,
            old_h=old_h,
            max_dim=max_dim,
            min_dim=min_dim,
            scale=scale,
            allow_upscale=allow_upscale,
        )
        transform = kwimage.Affine.scale((sx, sy))
        old_ext = Path(src_img['file_name']).suffix or '.png'
        suffix = output_ext or old_ext
        rel_fpath = _safe_rel_image_name(src_img, gid, suffix, asset_dname)
        out_fpath = dst.parent / rel_fpath
        out_fpath.parent.mkdir(parents=True, exist_ok=True)

        delayed = src_dset.coco_image(gid).imdelay().prepare().resize(
            (new_w, new_h),
            interpolation=interpolation,
        )
        resized = delayed.finalize()
        kwimage.imwrite(out_fpath, resized)

        img['file_name'] = rel_fpath
        img['width'] = new_w
        img['height'] = new_h

        for aid in list(new_dset.index.gid_to_aids.get(gid, [])):
            ann = new_dset.index.anns[aid]
            _warp_annotation(
                ann,
                transform=transform,
                input_dims=(old_h, old_w),
                output_dims=(new_h, new_w),
            )

    new_dset.dump(newlines=True)
    return new_dset


class ResizeKwcocoCLI(scfg.DataConfig):
    src = scfg.Value(None, help='input kwcoco path', required=True)
    dst = scfg.Value(None, help='output kwcoco path')
    inplace = scfg.Flag(False, help='overwrite input kwcoco metadata path')
    max_dim = scfg.Value(None, type=float, help='resize so the longest side is at most this value')
    min_dim = scfg.Value(None, type=float, help='resize so the shortest side is at most this value')
    scale = scfg.Value(None, type=float, help='explicit scale factor')
    asset_dname = scfg.Value('resized_assets', help='relative asset directory inside the destination bundle')
    output_ext = scfg.Value(None, help='optional image extension override, e.g. .jpg')
    interpolation = scfg.Value('linear', help='interpolation mode for delayed image resize')
    allow_upscale = scfg.Flag(False, help='if true, allow the resize to enlarge smaller images')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> from shitspotter.cli.resize_kwcoco import ResizeKwcocoCLI
            >>> import kwcoco
            >>> import ubelt as ub
            >>> dpath = ub.Path.appdir('shitspotter/tests/resize_kwcoco_cli').ensuredir()
            >>> src = kwcoco.CocoDataset.demo('vidshapes8', dpath=dpath / 'src_demo').fpath
            >>> dst = dpath / 'cli_resized.kwcoco.json'
            >>> ResizeKwcocoCLI.main(argv=0, src=src, dst=dst, max_dim=96)
            >>> assert dst.exists()
        """
        config = cls.cli(argv=argv, data=kwargs, strict=True, verbose='auto')
        if config.inplace:
            if config.dst not in [None, config.src]:
                raise ValueError('inplace=True is incompatible with dst != src')
            config.dst = config.src
        if config.dst is None:
            raise ValueError('Must specify output path')
        resize_kwcoco(
            src=config.src,
            dst=config.dst,
            max_dim=config.max_dim,
            min_dim=config.min_dim,
            scale=config.scale,
            asset_dname=config.asset_dname,
            output_ext=config.output_ext,
            interpolation=config.interpolation,
            allow_upscale=config.allow_upscale,
        )


__cli__ = ResizeKwcocoCLI


if __name__ == '__main__':
    __cli__.main()
