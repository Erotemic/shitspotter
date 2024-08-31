#!/usr/bin/env python3
"""
DEPRECATED
Use kwcoco plot_stats instead
"""
import scriptconfig as scfg
import ubelt as ub
import os

try:
    from line_profiler import profile
except ImportError:
    profile = ub.identity


class CocoAnnotationStatsCLI(scfg.DataConfig):
    """
    Inspect properties of annotations write stdout and programatic reports.
    """
    src = scfg.Value(None, help='path to kwcoco file', position=1)
    dst_fpath = scfg.Value('auto', help='manifest of results. If unspecfied defaults to dst_dpath / "stats.json"')
    dst_dpath = scfg.Value('./coco_annot_stats', help='directory to dump results')

    dpi = scfg.Value(300, help='dpi for figures')

    @classmethod
    def main(cls, cmdline=1, **kwargs):
        """
        Example:
            >>> # xdoctest: +REQUIRES(env:HAS_SHITSPOTTER_DATA)
            >>> import shitspotter
            >>> coco_fpath = shitspotter.util.find_shit_coco_fpath()
            >>> from shitspotter.cli.coco_annotation_stats import *  # NOQA
            >>> cmdline = 0
            >>> kwargs = dict(src=coco_fpath)
            >>> cls = CocoAnnotationStatsCLI
            >>> cls.main(cmdline=cmdline, **kwargs)
        """
        import rich
        from rich.markup import escape
        config = cls.cli(cmdline=cmdline, data=kwargs, strict=True)
        rich.print('config = ' + escape(ub.urepr(config, nl=1)))
        run(config)


@profile
def run(config):
    import json
    import kwutil

    dst_dpath = ub.Path(config['dst_dpath'])
    if config['dst_fpath'] == 'auto':
        dst_fpath = dst_dpath / 'stats.json'
    else:
        dst_fpath = ub.Path(config['dst_fpath'])
    dst_dpath.ensuredir()

    import rich
    rich.print(f'Destination Path: [link={dst_dpath}]{dst_dpath}[/link]')

    if True:
        from kwutil.process_context import ProcessContext
        proc_context = ProcessContext(
            name='shitspotter.cli.coco_annotation_stats',
            config=kwutil.Json.ensure_serializable(config.to_dict())
        )
        proc_context.start()
    else:
        proc_context = None

    import safer

    plots_dpath = dst_dpath / 'annot_stat_plots'
    tables_fpath = dst_dpath / 'stats_tables.json'

    tables_data, nonsaved_data = build_stats_data(config)
    rich.print(f'Write {tables_fpath}')
    with safer.open(tables_fpath, 'w', temp_file=not ub.WIN32) as fp:
        json.dump(tables_data, fp)

    print('Preparing plots')
    rich.print(f'Will write plots to: [link={plots_dpath}]{plots_dpath}[/link]')
    plot_functions = _define_plot_functions(
        plots_dpath, tables_data, nonsaved_data, dpi=config.dpi)

    pman = kwutil.util_progress.ProgressManager()
    with pman:
        plot_func_keys = list(plot_functions.keys())
        # plot_func_keys = [
        #     # 'images_over_time',
        #     'images_timeofday_distribution',
        # ]
        for key in pman.ProgIter(plot_func_keys, desc='plot'):
            func = plot_functions[key]
            func()
    rich.print(f'Finished writing plots to: [link={plots_dpath}]{plots_dpath}[/link]')

    # Write manifest of all data written to disk
    summary_data = {}
    if proc_context is not None:
        proc_context.stop()
        obj = proc_context.stop()
        obj = kwutil.Json.ensure_serializable(obj)
        summary_data['info'] = [obj]

    print('Finalizing manifest')
    summary_data['src'] = str(config['src'])
    summary_data['plots_dpath'] = os.fspath(plots_dpath)
    summary_data['tables_fpath'] = os.fspath(tables_fpath)
    # Write file to indicate the process has completed correctly
    # TODO: Use safer
    rich.print(f'Write {dst_fpath}')
    with safer.open(dst_fpath, 'w', temp_file=not ub.WIN32) as fp:
        json.dump(summary_data, fp, indent='    ')


@profile
def build_stats_data(config):
    import kwcoco
    import kwimage
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import json

    print('Loading kwcoco file')
    dset = kwcoco.CocoDataset.coerce(config['src'])

    annots = dset.annots()
    print('Building stats')
    detections : kwimage.Detections = annots.detections
    boxes : kwimage.Boxes = detections.boxes
    polys : kwimage.PolygonList = detections.data['segmentations']

    box_width =  boxes.width.ravel()
    box_height = boxes.height.ravel()

    box_canvas_width = np.array(annots.images.get('width'))
    box_canvas_height = np.array(annots.images.get('height'))

    images = dset.images()
    image_widths = images.get('width')
    image_heights = images.get('height')
    max_width = max(image_widths)  # NOQA
    max_height = max(image_heights)  # NOQA

    anns_per_image = np.array(images.n_annots)
    images_with_eq0_anns = (anns_per_image == 0).sum()
    images_with_ge1_anns = (anns_per_image >= 1).sum()
    scalar_stats = {
        **dset.basic_stats(),
        'images_with_eq0_anns': images_with_eq0_anns,
        'images_with_ge1_anns': images_with_ge1_anns,
        'frac_images_with_ge1_anns': images_with_ge1_anns / len(images),
        'frac_images_with_eq0_anns': images_with_eq0_anns / len(images),
    }
    print(f'scalar_stats = {ub.urepr(scalar_stats, nl=1)}')

    images.lookup('geos_point')

    perimage_data = pd.DataFrame({
        'anns_per_image': anns_per_image,
        'width': image_widths,
        'height': image_heights,
        'datetime': images.get('datetime', None),
    })

    ESTIMATE_SUNLIGHT = 1
    if ESTIMATE_SUNLIGHT:
        from shitspotter.util.util_gis import coco_estimate_sunlight
        try:
            sunlight_values = coco_estimate_sunlight(dset, image_ids=images)
            perimage_data['sunlight'] = sunlight_values
        except ImportError:
            print('Unable to estimate sunlight')

    perannot_data = gpd.GeoDataFrame({
        'geometry': [p.to_shapely() for p in polys],
        'annot_id': annots.ids,
        'image_id': annots.image_id,
        'box_rt_area': np.sqrt(boxes.area.ravel()),
        'box_width': box_height,
        'box_height': box_height,
        'rel_box_width': box_width / box_canvas_width,
        'rel_box_height': box_height / box_canvas_height,
    })
    perannot_data['num_vertices'] = perannot_data.geometry.apply(geometry_length)
    perannot_data = polygon_shape_stats(perannot_data)
    perannot_data['centroid_x'] = perannot_data.geometry.centroid.x
    perannot_data['centroid_y'] = perannot_data.geometry.centroid.y
    perannot_data['rel_centroid_x'] = perannot_data.geometry.centroid.x / box_canvas_width
    perannot_data['rel_centroid_y'] = perannot_data.geometry.centroid.y / box_canvas_height

    _summary_data = ub.udict(perannot_data.to_dict()) - {'geometry'}
    _summary_df = pd.DataFrame(_summary_data)
    tables_data = {}
    tables_data['perannot_data'] = json.loads(_summary_df.to_json(orient='table'))
    tables_data['perimage_data'] = json.loads(perimage_data.to_json(orient='table'))

    # Data that we don't serialize, but some plots depend on.
    nonsaved_data = {
        'boxes': boxes,
        'polys': polys,
    }
    return tables_data, nonsaved_data


@profile
def _define_plot_functions(plots_dpath, tables_data, nonsaved_data, dpi=300):
    """
    Defines plot functions as closures, to make it easier to share common
    data and write the functions in a more concise manner. This also
    makes it easier to enable / disable plots.

    Unfortunately we do lose some static analysis due to use of closures.
    """
    import kwplot
    import pandas as pd
    import json
    sns = kwplot.autosns(verbose=3)

    polys = nonsaved_data['polys']
    boxes = nonsaved_data['boxes']

    perannot_data = pd.read_json(json.dumps(tables_data['perannot_data']), orient='table')
    perimage_data = pd.read_json(json.dumps(tables_data['perimage_data']), orient='table')

    plot_functions = {}
    def register(func):
        key = func.__name__
        if key in plot_functions:
            raise AssertionError('duplicate plot name')
        plot_functions[key] = func

    annot_max_x = boxes.br_x.max()
    annot_max_y = boxes.br_y.max()

    max_anns_per_image = perimage_data['anns_per_image'].max()

    figman = kwplot.FigureManager(
        dpath=plots_dpath,
        dpi=dpi,
        verbose=1
    )
    figman.labels.add_mapping({
        'num_vertices': 'Num Polygon Vertices',
        'centroid_x': 'Polygon Centroid X',
        'centroid_y': 'Polygon Centroid Y',
        'obox_major': 'OBox Major Axes Length',
        'obox_minor': 'OBox Minor Axes Length',
        'rt_area': 'Polygon sqrt(Area)'
    })

    def _split_histplot(data, x):
        """
        TODO: generalize, if there is not a huge delta between histogram
        values, then fallback to just a single histogram plot.
        Need to figure out:
        is it possible to pass this an existing figure, so we don't always
        create a new one with plt.subplots?

        References:
            https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
            https://stackoverflow.com/questions/63726234/how-to-draw-a-broken-y-axis-catplot-graphes-with-seaborn
        """
        plt = kwplot.plt
        fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace': 0.05})

        split_point = 'auto'
        if split_point == 'auto':
            histogram = data[x].value_counts()
            small_values = histogram[histogram < histogram.mean()]
            split_point = int(small_values.max() * 1.5)

        sns.histplot(data=data, x=x, ax=ax_top, binwidth=1, discrete=True)
        sns.histplot(data=data, x=x, ax=ax_bottom, binwidth=1, discrete=True)

        sns.despine(ax=ax_bottom)
        sns.despine(ax=ax_top, bottom=True)
        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        #remove one of the legend
        if ax_bottom.legend_ is not None:
            ax_bottom.legend_.remove()

        ax_top.set_ylabel('')
        ax_top.set_ylim(bottom=split_point)   # those limits are fake
        ax_bottom.set_ylim(0, split_point)
        return ax_top, ax_bottom

    @register
    def polygon_centroid_distribution():
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.kdeplot(data=perannot_data, x='centroid_x', y='centroid_y', ax=ax)
        sns.scatterplot(data=perannot_data, x='centroid_x', y='centroid_y', ax=ax, hue='rt_area', alpha=0.8)
        ax.set_aspect('equal')
        ax.set_title('Polygon Absolute Centroid Positions')
        #ax.set_xlim(0, max_width)
        #ax.set_ylim(0, max_height)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        figman.labels.relabel(ax)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        figman.finalize('centroid_absolute_distribution.png')

    @register
    def image_size_distribution():
        # ax = figman.figure(fnum=1, doclf=True).gca()
        img_dsizes = [f'{w}✕{h}' for w, h in zip(perimage_data['width'], perimage_data['height'])]
        perimage_data['img_dsizes'] = img_dsizes
        # sns.histplot(data=perimage_data, x='img_dsizes', ax=ax)
        data = perimage_data
        x = 'img_dsizes'
        ax_top, ax_bottom = _split_histplot(data=data, x=x)
        ax_bottom.set_xlabel('Image Width ✕ Height')
        ax_bottom.set_ylabel('Number of Images')
        # ax.set_aspect('equal')
        ax_top.set_title('Image Size Histogram')
        figman.finalize('image_size_distribution.png')

    @register
    def obox_size_distribution():
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.kdeplot(data=perannot_data, x='obox_major', y='obox_minor', ax=ax)
        sns.scatterplot(data=perannot_data, x='obox_major', y='obox_minor', ax=ax)
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_title('Oriented Bounding Box Sizes')
        ax.set_aspect('equal')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        figman.labels.relabel(ax)
        figman.finalize('obox_size_distribution.png')

    @register
    def polygon_area_vs_num_verts():
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.kdeplot(data=perannot_data, x='rt_area', y='num_vertices', ax=ax)
        sns.scatterplot(data=perannot_data, x='rt_area', y='num_vertices', ax=ax)
        figman.labels.relabel(ax)
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylim(0, ax.get_ylim()[1])
        ax.set_title('Polygon Area vs Num Vertices')
        figman.finalize('polygon_area_vs_num_verts.png')

    @register
    def polygon_area_histogram():
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.histplot(data=perannot_data, x='rt_area', ax=ax, kde=True)
        figman.labels.relabel(ax)
        ax.set_title('Polygon sqrt(Area) Histogram')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_ylabel('Number of Annotations')
        ax.set_yscale('symlog')
        figman.finalize('polygon_area_histogram.png')

    @register
    def polygon_num_vertices_histogram():
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.histplot(data=perannot_data, x='num_vertices', ax=ax)
        ax.set_title('Polygon Number of Vertices Histogram')
        ax.set_ylabel('Number of Annotations')
        ax.set_xlim(0, ax.get_xlim()[1])
        ax.set_yscale('linear')
        figman.labels.relabel(ax)
        figman.finalize('polygon_num_vertices_histogram.png')

    @register
    def anns_per_image_histogram():
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.histplot(data=perimage_data, x='anns_per_image', ax=ax, binwidth=1)
        ax.set_yscale('linear')
        ax.set_xlabel('Number of Annotations')
        ax.set_ylabel('Number of Images')
        ax.set_title('Number of Annotations per Image')
        ax.set_xlim(0, max_anns_per_image)
        figman.labels.relabel(ax)
        # ax.set_yscale('symlog', linthresh=10)
        figman.labels.force_integer_xticks()
        figman.finalize('anns_per_image_histogram.png')

    @register
    def anns_per_image_histogram_splity():
        # References:
        # https://stackoverflow.com/questions/32185411/break-in-x-axis-of-matplotlib
        # https://stackoverflow.com/questions/63726234/how-to-draw-a-broken-y-axis-catplot-graphes-with-seaborn

        # ax = figman.figure(fnum=1, doclf=True).gca()
        plt = kwplot.plt
        fig, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace': 0.05})

        # ax_top, ax_bottom = _split_histplot(data=data, x=x)

        sns.histplot(data=perimage_data, x='anns_per_image', ax=ax_top, binwidth=1, discrete=True)
        sns.histplot(data=perimage_data, x='anns_per_image', ax=ax_bottom, binwidth=1, discrete=True)

        sns.despine(ax=ax_bottom)
        sns.despine(ax=ax_top, bottom=True)
        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        split_point = 20

        #remove one of the legend
        if ax_bottom.legend_ is not None:
            ax_bottom.legend_.remove()

        ax.set_yscale('linear')
        ax_bottom.set_xlabel('Number of Annotations')
        ax_bottom.set_ylabel('Number of Images')
        ax_top.set_ylabel('')
        ax_top.set_title('Number of Annotations per Image')
        ax_bottom.set_xlim(0 - 0.5, max_anns_per_image + 0.5)
        # figman.labels.relabel(ax)

        ax_top.set_ylim(bottom=split_point)   # those limits are fake
        ax_bottom.set_ylim(0, split_point)

        # figman.labels.force_integer_ticks(axis='x', method='ticker', ax=ax_bottom)
        figman.labels.force_integer_ticks(axis='x', method='maxn', ax=ax_bottom)
        figman.finalize('anns_per_image_histogram_splity.png', fig=fig)

    @register
    def anns_per_image_histogram_ge1():
        ax = figman.figure(fnum=1, doclf=True).gca()
        perimage_ge1_data = perimage_data[perimage_data['anns_per_image'] >= 1]
        sns.histplot(data=perimage_ge1_data, x='anns_per_image', ax=ax, binwidth=1, discrete=True)
        ax.set_yscale('linear')
        ax.set_xlabel('Number of Annotations')
        ax.set_ylabel('Number of Images')
        ax.set_title('Number of Annotations per Image\n(with at least 1 annotation)')
        figman.labels.relabel(ax)
        ax.set_xlim(1 - 0.5, max_anns_per_image + 0.5)
        # figman.labels.force_integer_ticks(axis='x', method='maxn', ax=ax)
        figman.labels.force_integer_ticks(axis='x', method='ticker', ax=ax, hack_labels=0)
        # ax.set_xticks(ax.get_xticks().astype(int))

        # ax.set_yscale('symlog', linthresh=10)
        figman.finalize('anns_per_image_histogram_ge1.png')

    @register
    def images_over_time():
        import pandas as pd
        import numpy as np
        img_df = perimage_data.sort_values('datetime')
        img_df['pd_datetime'] = pd.to_datetime(img_df.datetime)
        img_df['collection_size'] = np.arange(1, len(img_df) + 1)
        ax = figman.figure(fnum=1, doclf=True).gca()
        sns.histplot(data=img_df, x='pd_datetime', ax=ax, cumulative=True)
        sns.lineplot(data=img_df, x='pd_datetime', y='collection_size')
        ax.set_title('Images collected over time')
        ax.set_xlabel('Datetime')
        ax.set_ylabel('Number of Images Collected')
        figman.finalize('images_over_time.png')

    @register
    def images_timeofday_distribution():
        import pandas as pd
        import numpy as np
        import kwutil

        img_df = perimage_data.sort_values('datetime')
        img_df['pd_datetime'] = pd.to_datetime(img_df.datetime)
        img_df['collection_size'] = np.arange(1, len(img_df) + 1)
        datetimes = [kwutil.datetime.coerce(x) for x in img_df['datetime']]
        img_df['timestamp'] = [x.timestamp() for x in datetimes]
        img_df['date'] = [x.date() for x in datetimes]
        img_df['time'] = [x.time() for x in datetimes]
        img_df['year_month'] = [x.strftime('%Y-%m') for x in datetimes]
        img_df['day_of_year'] = [x.timetuple().tm_yday for x in datetimes]
        img_df['month'] = [x.strftime('%m') for x in datetimes]
        img_df['hour_of_day'] = [z.hour + z.minute / 60 + z.second / 3600 for z in img_df['time']]
        ax = figman.figure(fnum=1, doclf=True).gca()
        # sns.histplot(data=img_df, x='month', ax=ax)
        # sns.kdeplot(data=img_df, x='day_of_year', y='hour_of_day')
        # sns.scatterplot(data=img_df, x='day_of_year', y='hour_of_day', hue='sunlight_values')
        palette = sns.color_palette("flare", n_colors=4, as_cmap=True).reversed()
        sns.scatterplot(data=img_df, x='day_of_year', y='hour_of_day')
        sns.scatterplot(data=img_df, x='day_of_year', y='hour_of_day', hue='sunlight', palette=palette, legend=False)
        # sns.kdeplot(data=img_df, x='hour_of_day', y='day_of_year')
        import kwimage
        kwplot.phantom_legend({
            'Night': kwimage.Color.coerce(palette.colors[0]).as255(),
            'Day': kwimage.Color.coerce(palette.colors[-1]).as255(),
            'nan': 'blue',
        }, mode='circle')
        ax.set_title('Time Captured')
        ax.set_xlabel('Day of Year')
        ax.set_ylabel('Hour of Day')
        figman.finalize('images_timeofday_distribution.png')

    @register
    def all_polygons():
        ax = figman.figure(fnum=1, doclf=True).gca()
        edgecolor = 'black'
        facecolor = 'baby shit brown'  # its a real color!
        #edgecolor = 'darkblue'
        #facecolor = 'lawngreen'
        #edgecolor = kwimage.Color.coerce('kitware_darkblue').as01()
        #facecolor = kwimage.Color.coerce('kitware_green').as01()
        polys.draw(alpha=0.5, edgecolor=edgecolor, facecolor=facecolor)
        ax.set_xlabel('Image X Coordinate')
        ax.set_ylabel('Image Y Coordinate')
        ax.set_title(f'All {len(polys)} Polygons')
        ax.set_aspect('equal')
        ax.set_xlim(0, annot_max_x)
        ax.set_ylim(0, annot_max_y)
        figman.labels.relabel(ax)
        ax.set_ylim(0, annot_max_y)  # not sure why this needs to be after the relabel, should ideally fix that.
        ax.invert_yaxis()
        figman.finalize('all_polygons.png', tight_layout=0)  # tight layout seems to cause issues here

    return plot_functions


def polygon_shape_stats(df):
    """
    Compute shape statistics about a geopandas dataframe (assume UTM CRS)
    """
    import numpy as np
    import kwimage
    df['hull_rt_area'] = np.sqrt(df.geometry.convex_hull.area)
    df['rt_area'] = np.sqrt(df.geometry.area)

    obox_whs = [kwimage.MultiPolygon.from_shapely(s).oriented_bounding_box().extent
                for s in df.geometry]

    df['obox_major'] = [max(e) for e in obox_whs]
    df['obox_minor'] = [min(e) for e in obox_whs]
    df['major_obox_ratio'] = df['obox_major'] / df['obox_minor']

    # df['ch_aspect_ratio'] =
    # df['isoperimetric_quotient'] = df.geometry.apply(shapestats.ipq)
    # df['boundary_amplitude'] = df.geometry.apply(shapestats.compactness.boundary_amplitude)
    # df['eig_seitzinger'] = df.geometry.apply(shapestats.compactness.eig_seitzinger)
    return df


def geometry_flatten(geom):
    """
    References:
        https://gis.stackexchange.com/questions/119453/count-the-number-of-points-in-a-multipolygon-in-shapely
    """
    if hasattr(geom, 'geoms'):  # Multi<Type> / GeometryCollection
        for g in geom.geoms:
            yield from geometry_flatten(g)
    elif hasattr(geom, 'interiors'):  # Polygon
        yield geom.exterior
        yield from geom.interiors
    else:  # Point / LineString
        yield geom


def geometry_length(geom):
    return sum(len(g.coords) for g in geometry_flatten(geom))


__cli__ = CocoAnnotationStatsCLI

if __name__ == '__main__':
    r"""

    CommandLine:
        python ~/code/shitspotter/shitspotter/cli/coco_annotation_stats.py

        LINE_PROFILE=1 python -m shitspotter.cli.coco_annotation_stats $HOME/data/dvc-repos/shitspotter_dvc/data.kwcoco.json \
            --dst_fpath $HOME/code/shitspotter/coco_annot_stats/stats.json \
            --dst_dpath $HOME/code/shitspotter/coco_annot_stats
    """
    __cli__.main()
