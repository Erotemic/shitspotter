datasets = []
datasets.extend([
    {
        'name': r'ImageNet-LSVRC2017',
        'coco_fpath': '/data/joncrall/dvc-repos/ImageNet/ILSVRC/ilsvrc2017_full.kwcoco.zip',
    },
    {
        'name': r'MSCOCO2017',
        'coco_fpath': '/data/joncrall/dvc-repos/mscoco2017/coco2017/annotations/instances_trainval2017.json',
    },
    {
        'name': r'CityScapes',
        'coco_fpath': '/data/joncrall/dvc-repos/Cityscapes/cityscapes_all.kwcoco.zip',
    },
    {
        'name': r'ZeroWaste',
        'coco_fpath': '/data/joncrall/dvc-repos/ZeroWaste/zerowaste-f/splits_final_deblurred/all.kwcoco.zip',
    },
    {
        'name': r'TACO',
        'coco_fpath': '/home/joncrall/data/dvc-repos/TACO/data/annotations.json',
    },
    {
        'name': r'TrashCanV1',
        'coco_fpath': '/home/joncrall/data/dvc-repos/TrashCan-v1/trashcan_instance_trainval.kwcoco.zip',
    },

    {
        'name': r'SpotGarbage-GINI',
        'coco_fpath': '/data/joncrall/dvc-repos/SpotGarbage-GINI/all.kwcoco.zip',
    },
    {
        'name': r'UAVVaste',
        'coco_fpath': '/data/joncrall/dvc-repos/UAVVaste/annotations/annotations.json',
    },
    {
        'name': r'MSHIT',
        'coco_fpath': '/data/joncrall/dvc-repos/MSHIT/all.kwcoco.zip',
    },
    {
        'name': '``ScatSpotter\'\'',
        'coco_fpath': '/home/joncrall/code/shitspotter/shitspotter_dvc/data.kwcoco.json',
    },
])

# datasets = datasets[::-1]


def main():
    import kwcoco
    from kwcoco.cli import coco_plot_stats
    import ubelt as ub
    import itertools as it
    fpath_to_row = {r['coco_fpath']: r for r in datasets}
    for row in ub.ProgIter(fpath_to_row.values(), verbose=3, freq=1):
        name = row['name'].split(' ')[0]
        row['name'] = name

    if 0:
        print(list(fpath_to_row.keys()))
        dsets = kwcoco.CocoDataset.load_multiple(fpath_to_row.keys(), workers=8, ordered=False, verbose=3)

        # Do a first pass where we build the tables which are relatively quick
        fastpath, slowpath = it.tee(dsets)
        fpath_to_stats = {}
        for dset in fastpath:
            print(f'dset.fpath={dset.fpath}')
            cache_dpath = ub.Path(dset.fpath).absolute().parent / '_cache'
            cacher = ub.Cacher(dset._cached_hashid(), dpath=cache_dpath)
            scalar_stats = cacher.tryload()
            if scalar_stats is None:
                info = coco_plot_stats.build_stats_data(dset)
                scalar_stats, tables_data, nonsaved_data, dataframes = info
                cacher.save(scalar_stats)
            fpath_to_stats[dset.fpath] = scalar_stats
        dsets = list(slowpath)

        jobs = ub.JobPool(mode='process', max_workers=10)

        # Do a second pass where we do the plots, which are expensive.
        for dset in ub.ProgIter(dsets, verbose=3, freq=1):
            row = fpath_to_row[dset.fpath]
            name = row['name'].split(' ')[0]
            hashid = dset._cached_hashid()
            print(f'dset.fpath={dset.fpath}, {name}, {hashid}')
            bundle_dpath = ub.Path(dset.bundle_dpath).absolute()
            cache_dpath = bundle_dpath / '_cache'
            stamp = ub.CacheStamp('plots', depends=hashid, dpath=cache_dpath)
            # stamp.clear()
            dpath = bundle_dpath / ('_visual_stats_' + name + hashid[0:8])
            scalar_stats = fpath_to_stats[dset.fpath]
            if stamp.expired() or 1:
                coco_plot_stats.PlotStatsCLI.main
                kwargs = dict(
                    cmdline=0, src=dset.fpath, dst_dpath=dpath,
                    # plots=['polygon_area_vs_num_verts_jointplot_logscale']
                    # plots=['all_polygons']
                )
                job = jobs.submit(coco_plot_stats.PlotStatsCLI.main, **kwargs)
                job.stamp = stamp

        for job in jobs.as_completed():
            job.result()
            job.stamp.renew()
            # stamp.renew()

    if True:
        accum = ub.ddict(list)
        for fpath, row in fpath_to_row.items():
            bundle_dpath = ub.Path(fpath).parent
            found = list(bundle_dpath.glob('_visual_stats*'))
            found2 = list(bundle_dpath.glob('_cache*/_visual_stats*'))
            found = (found or found2)[0]
            if found2:
                assert not found2
                found.move(found.parent.parent / found.name)
            image_fpaths = (found / 'annot_stat_plots').ls()
            for img_fpath in image_fpaths:
                accum[img_fpath.name].append((row, img_fpath))

        for name, paths in accum.items():
            import kwimage
            stack = []
            for row, p in paths:
                canvas = kwimage.imread(p)
                canvas = kwimage.draw_header_text(canvas, row['name'], fontScale=3, thickness=3, color='black', bg_color='white')
                stack.append(canvas)
            canvas = kwimage.stack_images_grid(stack, chunksize=5, axis=0)
            dpath = ub.Path('/home/joncrall/code/shitspotter/papers/application-2024/plots/appendix/dataset_compare')
            canvas_fpath = (dpath / ('combo_' + name + '.png'))
            # import kwplot
            # kwplot.plt
            # kwplot.figure()
            # kwplot.imshow(canvas)
            import numpy as np
            canvas = np.ascontiguousarray(canvas)
            if canvas_fpath.is_dir():
                canvas_fpath.delete()
            kwimage.imwrite(canvas_fpath, canvas, backend='auto')
            dpath.ensuredir()
            import rich
            rich.print(f'Dpath: [link={dpath}]{dpath}[/link]')

if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/papers/application-2024/scripts/related_dataset_comparison.py
    """
    main()
