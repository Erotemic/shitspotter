#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class ExtractPolygonsCLI(scfg.DataConfig):
    """
    CommandLine:
        python -m shitspotter.cli.extract_polygons ExtractPolygonsCLI \
            --src /data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_train_evals/eval/flat/heatmap_eval/heatmap_eval_id_a6da2301/.pred/heatmap_pred/heatmap_pred_id_690643f5/pred.kwcoco.zip \
            --dst test-polys3.kwcoco.json

        kwcoco stats /data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_0a1c1fa4/.pred/heatmap_pred/heatmap_pred_id_5ffadae5/pred.kwcoco.zip

        python -m shitspotter.cli.extract_polygons ExtractPolygonsCLI \
            --src /data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_0a1c1fa4/.pred/heatmap_pred/heatmap_pred_id_5ffadae5/pred.kwcoco.zip \
            --dst test-polys3.kwcoco.json

        kwcoco eval /home/joncrall/data/dvc-repos/shitspotter_dvc/vali_imgs228_20928c8c.kwcoco.zip test-polys3.kwcoco.json --iou_thresh=0.1

    Ignore:
        dst = 'test-polys.kwcoco.zip'
    """
    src = scfg.Value(None, help='input kwcoco', position=1)
    dst = scfg.Value(None, help='output kwcoco', position=2)
    thresh = scfg.Value(0.5, help='detection threshold')
    scale = scfg.Value(0.5, help='The scale at which to run polygon thresholding')
    workers = scfg.Value('auto', help='number of background workers')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        CommandLine:
            xdoctest -m shitspotter.cli.extract_polygons ExtractPolygonsCLI

        Ignore:
            kwargs = {}
            kwargs['dst'] = 'tmp.kwcoco.zip'
            kwargs['src'] = '/data/joncrall/dvc-repos/shitspotter_expt_dvc/_shitspotter_evals/eval/flat/heatmap_eval/heatmap_eval_id_0a1c1fa4/.pred/heatmap_pred/heatmap_pred_id_5ffadae5/pred.kwcoco.zip'

        Example:
            >>> # xdoctest: +REQUIRES(module:geowatch)
            >>> from shitspotter.cli.extract_polygons import *  # NOQA
            >>> import geowatch
            >>> import ubelt as ub
            >>> from shitspotter.cli.extract_polygons import *  # NOQA
            >>> dpath = ub.Path.appdir('shitspotter', 'tests', 'test_extract')
            >>> dset = geowatch.coerce_kwcoco('geowatch', heatmap=True)
            >>> cmdline = 0
            >>> kwargs = dict(dst=dpath / 'out.kwcoco.zip')
            >>> kwargs['src'] = dset.fpath
            >>> cls = ExtractPolygonsCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        import kwcoco
        import kwutil
        import kwimage
        src = config.src
        dst = config.dst
        thresh = config.thresh
        scale = config.scale
        workers = kwutil.coerce_num_workers(config.workers)

        dset = kwcoco.CocoDataset(src)

        try:
            name = ub.modpath_to_modname(__file__),
        except NameError:
            # for ipython
            name = 'shitspotter.cli.extract_polygons'

        proc_context = kwutil.ProcessContext(
            name=name,
            type='process',
            config=kwutil.Json.ensure_serializable(dict(config)),
        )
        proc_context.start()

        heatmap_channel = 'salient'  # fixme: hard coded!
        catname = 'poop'  # fixme: hard coded!

        dset = kwcoco.CocoDataset(src)
        category_id = dset.ensure_category(catname)

        # We will do thresholding in a downsampled version of the data for a
        # bit more speed, and because thats the scale we predicted at anyway.
        warp_threshspace_from_imgspace = kwimage.Affine.scale(scale)

        pman = kwutil.util_progress.ProgressManager()
        jobs = ub.JobPool(
            mode='process',
            max_workers=workers,
        )
        with pman, jobs:
            gid_iter = pman.ProgIter(dset.images(), desc='Submit Detect Jobs')

            # Send each image to a worker, which will extract the polygons
            for image_id in gid_iter:
                coco_img = dset.coco_image(image_id)
                jobs.submit(extract_from_single_img, coco_img, thresh,
                            heatmap_channel, warp_threshspace_from_imgspace)

            # Collect the worker jobs and add results to the dataset
            for job in pman.ProgIter(jobs.as_completed(), total=len(jobs), desc='Collect Detect Jobs'):
                image_id, new_anns = job.result()
                for new_ann in new_anns:
                    new_ann['image_id'] = image_id
                    new_ann['category_id'] = category_id
                    dset.add_annotation(**new_ann)

        if proc_context is not None:
            proc_context.stop()
            dset.dataset['info'].append(proc_context.obj)

        dset.fpath = dst
        ub.Path(dset.fpath).parent.ensuredir()
        print(f'Write detections to {dset.fpath}')
        dset.dump()


def extract_from_single_img(coco_img, thresh, heatmap_channel, warp_threshspace_from_imgspace):
    import kwimage
    import numpy as np
    warp_imgspace_from_threshspace = warp_threshspace_from_imgspace.inv()
    image_id = coco_img.img['id']
    delayed = coco_img.imdelay(channels=heatmap_channel)
    delayed = delayed.warp(warp_threshspace_from_imgspace)

    probs = delayed.finalize()[:, :, 0]
    probs = np.nan_to_num(probs)
    # TODO: parameterize
    probs = kwimage.gaussian_blur(probs, kernel=3)
    probs = kwimage.morphology(probs, mode='dilate', element='ellipse',
                               kernel=3)
    probs = kwimage.morphology(probs, mode='close', element='ellipse',
                               kernel=5)
    probs = kwimage.morphology(probs, mode='dilate', element='ellipse',
                               kernel=3)

    c_mask = probs > thresh
    mask = kwimage.Mask.coerce(c_mask)
    mpolys = mask.to_multi_polygon()
    polys = mpolys.data

    # if 0:
    #     mpolys = mpolys.simplify(1)
    #     kwplot.imshow(probs, docla=True)
    #     mpolys.draw()

    new_anns = []
    for poly in polys:
        try:
            # poly = poly.simplify(5)
            threshold = [thresh]
            score = float(score_poly(poly, probs, threshold)[0])

            # warp back to the original space.
            new_poly = poly.warp(warp_imgspace_from_threshspace)
            new_ann = {
                'segmentation': new_poly,
                'score': score,
            }
            bbox = new_poly.to_box().to_coco()
            new_ann['bbox'] = bbox
            new_anns.append(new_ann)
        except ValueError as ex:
            print(f'ex={ex}')
            # skip shapely errors
            ...
    return image_id, new_anns


def score_poly(poly, probs, threshold=-1, use_rasterio=True):
    """
    Compute the average heatmap response of a heatmap inside of a polygon.

    Args:
        poly (kwimage.Polygon | MultiPolygon):
            in pixel coords

        probs (ndarray):
            heatmap to compare poly against in [..., c, h, w] format.
            The last two dimensions should be height, and width.
            Any leading batch dimensions will be preserved in output,
            e.g. (gid, chan, h, w) -> (gid, chan)

        use_rasterio (bool):
            use rasterio.features module instead of kwimage

        threshold (float | List[float | str]):
            Return fraction of poly with probs > threshold.  If -1, return
            average value of probs in poly. Can be a list of values, in which
            case returns all of them.

    Returns:
        List[ndarray] | ndarray:

            When thresholds is a list, returns a corresponding list of ndarrays
            with an entry keeping the leading dimensions of probs and
            marginalizing over the last two.
    """
    import kwimage
    import numpy as np
    if not isinstance(poly, (kwimage.Polygon, kwimage.MultiPolygon)):
        poly = kwimage.MultiPolygon.from_shapely(poly)  # 2.4% of runtime

    _return_list = ub.iterable(threshold)
    if not _return_list:
        threshold = [threshold]

    # First compute the valid bounds of the polygon
    # And create a mask for only the valid region of the polygon

    box = poly.box().quantize().to_xywh()

    # Ensure box is inside probs
    ymax, xmax = probs.shape[-2:]
    box = box.clip(0, 0, xmax, ymax).to_xywh()
    if box.area == 0:
        import warnings
        warnings.warn(
            'warning: scoring a polygon against an img with no overlap!')
        zeros = np.zeros(probs.shape[:-2])
        return [zeros] * len(threshold) if _return_list else zeros
    # sl_y, sl_x = box.to_slice()
    x, y, w, h = box.data
    pixels_are = 'areas' if use_rasterio else 'points'
    # kwimage inverse
    # 95% of runtime... would batch be faster?
    rel_poly = poly.translate((-x, -y))
    # rel_mask = rel_poly.to_mask((h, w), pixels_are=pixels_are).data
    # shapely polys hash correctly (based on shape, not memory location)
    # kwimage polys don't
    import kwimage
    rel_poly_ = kwimage.MultiPolygon.from_shapely(rel_poly.to_shapely())
    rel_mask = rel_poly_.to_mask((h, w), pixels_are=pixels_are).data
    # Slice out the corresponding region of probabilities
    rel_probs = probs[..., y:y + h, x:x + w]

    result = []

    # handle nans
    msk = (np.isfinite(rel_probs) * rel_mask).astype(bool)
    all_non_finite = not msk.any()

    mskd_rel_probs = np.ma.array(rel_probs, mask=~msk)

    for t in threshold:
        if all_non_finite:
            stat = np.full(rel_probs.shape[:-2], fill_value=np.nan)
        elif t == 'max':
            stat = mskd_rel_probs.max(axis=(-2, -1)).filled(0)
        elif t == 'min':
            stat = mskd_rel_probs.min(axis=(-2, -1)).filled(0)
        elif t == 'mean':
            stat = mskd_rel_probs.mean(axis=(-2, -1)).filled(0)
        elif t == 'median':
            stat = np.ma.median(mskd_rel_probs, axis=(-2, -1)).filled(0)
        elif t == -1:
            # Alias for mean, todo: deprecate
            stat = mskd_rel_probs.mean(axis=(-2, -1)).filled(0)
        else:
            # Real threshold case
            hard_prob = rel_probs > t
            mskd_hard_prob = np.ma.array(rel_probs, mask=~(msk | hard_prob))
            stat = mskd_hard_prob.mean(axis=(-2, -1))
        result.append(stat)

    return result if _return_list else result[0]


__cli__ = ExtractPolygonsCLI

if __name__ == '__main__':
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/cli/extract_polygons.py
        python -m shitspotter.cli.extract_polygons
    """
    __cli__.main()
