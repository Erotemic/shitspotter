"""
Main prediction API
"""

#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class PredictCLI(scfg.DataConfig):
    src = scfg.Value(None, help='Path to input kwcoco or path to directory of images', position=1)
    dst = scfg.Value(None, help='Path to output kwcoco file or directory to write results to')
    package_fpath = scfg.Value(None, help='Path to the packaged model.', alias=['model'])
    create_labelme = scfg.Value(False, help='if True, update original paths with labelme sidecars if they dont already exist')

    @classmethod
    def main(cls, argv=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +SKIP
            >>> from shitspotter.cli.predict import *  # NOQA
            >>> argv = 0
            >>> kwargs = dict()
            >>> cls = PredictCLI
            >>> config = cls(**kwargs)
            >>> cls.main(argv=argv, **config)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))

        if config.package_fpath is None:
            # default to current best model
            config.package_fpath = ub.Path('~/code/shitspotter/shitspotter_dvc/models/train_baseline_maskrcnn_v3_v_966e49df_model_0014999.pth').expand()

        if config.src is None:
            raise Exception('give a source')
            config.src = ub.Path('/home/joncrall/code/shitspotter/shitspotter_dvc/assets/_contributions/mikael-simburg-2024-12-26')
            config.src = ub.Path('/data/joncrall/dvc-repos/shitspotter_dvc/assets/poop-2024-11-22-T195205')
        else:
            config.src = ub.Path(config.src)

        if config.dst is None:
            # config.dst = ub.Path('./predict_output')
            config.dst = config.src.parent / (config.src.name + '-predict-output')

        config.dst = ub.Path(config.dst)
        config.src = ub.Path(config.src)

        if not config.dst.exists():
            if '.' not in config.dst.name:
                dpath = config.dst.resolve().ensuredir()
                out_fpath = dpath / 'output.kwcoco.zip'
            else:
                out_fpath = config.dst
        else:
            if config.dst.is_dir():
                dpath = config.dst.resolve().ensuredir()
                out_fpath = dpath / 'output.kwcoco.zip'
            else:
                out_fpath = config.dst

        out_dpath = out_fpath.parent

        if config.src.is_dir():
            import kwcoco
            import kwutil
            import kwimage
            import kwimage.im_io
            kwimage.im_io
            config.src.glob
            gpaths = sorted(kwutil.util_path.coerce_patterned_paths(config.src, expected_extension=kwimage.im_io.IMAGE_EXTENSIONS))
            dset = kwcoco.CocoDataset.from_image_paths(gpaths)
            dset.reroot(absolute=True)
            for img in dset.images().objs:
                img['sensor_coarse'] = 'phone'
                img['channels'] = 'red|green|blue'
            input_fpath = out_dpath / 'input.kwcoco.zip'
            dset._update_fpath(input_fpath)
            dset.dump()
            src_fpath = dset.fpath
        else:
            src_fpath = config.src

        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        from geowatch.mlops import schedule_evaluation
        import shitspotter
        import shitspotter.pipelines
        pipeline = shitspotter.pipelines.detectron_evaluation_pipeline()
        pipeline  # TODO: do something with this directly

        package_fpath = config.package_fpath

        params = ub.codeblock(
                f'''
                pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
                matrix:
                    detectron_pred.checkpoint_fpath:
                         - {package_fpath}
                    detectron_pred.src_fpath:
                        - {src_fpath}
                    detectron_pred.workers: 4
                    detectron_pred.write_heatmap: true
                    detectron_pred.nms_thresh: 0.5
                    detection_evaluation.__enabled__: 0
                    heatmap_eval.__enabled__: 0
                ''')

        schedule_evaluation.main(
            cmdline=0,
            params=params,
            root_dpath=out_dpath,
            devices="0,",
            backend='tmux',
            run=1
        )

        # Maybe this doesn't work and we need to use an output bundle path.
        candidates = list((out_dpath / 'pred').glob('*/*/*/pred.kwcoco.zip'))
        real_out_fpath = candidates[0]
        ub.symlink(real_path=real_out_fpath, link_path=out_fpath, overwrite=True)

        if config.create_labelme:
            # TODO: finishme
            coco_dset = kwcoco.CocoDataset(real_out_fpath)
            seed_labels(coco_dset)
            # import xdev
            # xdev.embed()

        # python -m geowatch.mlops.schedule_evaluation \
        #     --params="
        #         pipeline: 'shitspotter.pipelines.detectron_evaluation_pipeline()'
        #         matrix:
        #             detectron_pred.checkpoint_fpath:
        #                  - $HOME/code/shitspotter/experiments/detectron_models.yaml
        #             detectron_pred.src_fpath:
        #                 - $VALI_FPATH
        #             detectron_pred.workers: 4
        #             detectron_pred.write_heatmap: true
        #             detectron_pred.nms_thresh: 0.5
        #             detection_eval.__enabled__: 1
        #             heatmap_eval.__enabled__: 1
        #     " \
        #     --root_dpath="$EVAL_PATH" \
        #     --devices="0," --tmux_workers=1 \
        #     --backend=tmux --skip_existing=1 \
        #     --run=1


def seed_labels(coco_dset):
    """
    Not sure if this goes here, but we want to seed any unannotated images
    with predictions as labelme files.

    Ignore:
        coco_dset = kwcoco.CocoDataset('/data/joncrall/dvc-repos/shitspotter_dvc/models/predict_output/output_bundle/pred.kwcoco.zip')
        coco_dset = kwcoco.CocoDataset('/home/joncrall/predict_output/pred/flat/detectron_pred/detectron_pred_id_864dc4c9/pred.kwcoco.zip')

    """
    # xdoctest: +REQUIRES(module:kwutil)
    from kwcoco.formats.labelme import LabelMeFile
    import kwimage
    import numpy as np
    new_anns = []
    for coco_img in coco_dset.images().coco_images:
        annots = coco_img.annots()
        if len(annots) == 0:
            print('skip')
            continue
        dets = annots.detections
        dets.data['aids'] = np.array(list(annots))
        dets = dets.compress(dets.data['scores'] > 0.90)
        dets = dets.non_max_supress()
        ssegs = dets.data['segmentations']
        fixed_ssegs = ssegs.__class__([])
        flags = []
        for sseg in ssegs:
            try:
                shp = sseg.to_multi_polygon().to_shapely()
                shp = shp.simplify(2)
                new_sseg = kwimage.MultiPolygon.from_shapely(shp)
                # new_sseg = kwimage.Polygon.circle((shp.centroid.x, shp.centroid.y), 50)
                fixed_ssegs.append(new_sseg)
                flags.append(True)
            except Exception as ex:
                print(f'ex={ex}')
                flags.append(False)
        print(f'flags={flags}')
        dets = dets.compress(flags)
        dets.data['segmentations'] = fixed_ssegs

        base_anns = coco_dset.annots(dets.data['aids']).objs
        modified_anns = dets.to_coco(dset=coco_dset, image_id=coco_img.img['id'], style='new')
        final_anns = []
        for base_ann, mod_ann in zip(base_anns, modified_anns):
            final_anns.append(base_ann | mod_ann)
        new_anns.extend(final_anns)

    new_dset = coco_dset.copy()
    new_dset.clear_annotations()
    for ann in new_anns:
        new_dset.add_annotation(**ann)

    sidecars = LabelMeFile.multiple_from_coco(new_dset)
    sidecars = list(sidecars)

    for sidecar in sidecars:
        sidecar.reroot(absolute=False)
        sidecar.fpath = sidecar.fpath.resolve()
        print(f'sidecar.fpath={sidecar.fpath}')
        if sidecar.data['shapes']:
            if not sidecar.fpath.exists():
                sidecar.dump()


__cli__ = PredictCLI

if __name__ == '__main__':
    """

    CommandLine:
        python -m shitspotter.cli.predict
        python -m shitspotter.cli.predict /home/joncrall/data/dvc-repos/shitspotter_staging/assets/poop-2024-12-30-T212347 --create_labelme=True
    """
    __cli__.main()
