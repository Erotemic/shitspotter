#!/usr/bin/env python3
import scriptconfig as scfg
import ubelt as ub


class CocoAnnotationStatsCLI(scfg.DataConfig):
    """
    Inspect properties of annotations write stdout and programatic reports.
    """
    src = scfg.Value(None, help='path to kwcoco file', position=1)
    dst_fpath = scfg.Value('auto', help='manifest of results. If unspecfied defaults to dst_dpath / "stats.json"')
    dst_dpath = scfg.Value('./coco_annot_stats', help='directory to dump results')

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

        dst_dpath = ub.Path(config['dst_dpath'])
        if config['dst_fpath'] == 'auto':
            dst_fpath = dst_dpath / 'stats.json'
        else:
            dst_fpath = ub.Path(config['dst_fpath'])
        dst_dpath.ensuredir()

        plots_dpath = dst_dpath / 'annot_stat_plots'
        tables_fpath = dst_dpath / 'stats_tables.json'

        import kwcoco
        import kwimage

        import numpy as np
        import pandas as pd
        dset = kwcoco.CocoDataset.coerce(config['src'])

        annots = dset.annots()
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

        perimage_data = pd.DataFrame({
            'anns_per_image': images.n_annots,
            'width': image_widths,
            'height': image_heights,
        })

        import geopandas as gpd
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

        import json
        import os

        _summary_data = ub.udict(perannot_data.to_dict()) - {'geometry'}
        _summary_df = pd.DataFrame(_summary_data)
        tables_data = {}
        tables_data['perannot_data'] = json.loads(_summary_df.to_json(orient='table'))
        tables_data['perimage_data'] = json.loads(perimage_data.to_json(orient='table'))
        tables_fpath.write_text(json.dumps(tables_data))

        draw_plots(plots_dpath, perannot_data, perimage_data, polys, boxes)

        # TODO: add config info
        summary_data = {}
        summary_data['src'] = str(config['src'])
        summary_data['plots_dpath'] = os.fspath(plots_dpath)
        summary_data['tables_fpath'] = os.fspath(tables_fpath)
        # Write file to indicate the process has completed correctly
        # TODO: Use safer
        dst_fpath.write_text(json.dumps(summary_data, indent='    '))


def draw_plots(plots_dpath, perannot_data, perimage_data, polys, boxes):
    import kwplot
    sns = kwplot.autosns()

    annot_max_x = boxes.br_x.max()
    annot_max_y = boxes.br_y.max()

    figman = kwplot.FigureManager(
        dpath=plots_dpath,
    )
    figman.labels.add_mapping({
        'num_vertices': 'Num Polygon Vertices',
        'centroid_x': 'Polygon Centroid X',
        'centroid_y': 'Polygon Centroid Y',
        'obox_major': 'Oriented Bounding Box Major Axes Length',
        'obox_minor': 'Oriented Bounding Box Minor Axes Length',
        'rt_area': 'Polygon sqrt(Area)'
    })

    # --- Polygon Centroid Distribution
    ax = figman.figure(fnum=1, doclf=True).gca()
    sns.kdeplot(data=perannot_data, x='centroid_x', y='centroid_y', ax=ax)
    sns.scatterplot(data=perannot_data, x='centroid_x', y='centroid_y', ax=ax, hue='rt_area', alpha=0.9)
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
    # ---

    # --- Relative Polygon Centroid Distribution
    ax = figman.figure(fnum=1, doclf=True).gca()
    sns.kdeplot(data=perannot_data, x='rel_centroid_x', y='rel_centroid_y', ax=ax)
    sns.scatterplot(data=perannot_data, x='rel_centroid_x', y='rel_centroid_y', ax=ax, hue='rt_area', alpha=0.9)
    ax.set_aspect('equal')
    ax.set_title('Polygon Relative Centroid Positions')
    figman.labels.relabel(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    figman.labels.relabel(ax)
    ax.invert_yaxis()
    figman.finalize('centroid_relative_distribution.png')
    # ---

    # ---
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
    # ---

    # ---
    ax = figman.figure(fnum=1, doclf=True).gca()
    sns.kdeplot(data=perannot_data, x='rt_area', y='num_vertices', ax=ax)
    sns.scatterplot(data=perannot_data, x='rt_area', y='num_vertices', ax=ax)
    figman.labels.relabel(ax)
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_title('Polygon Area vs Num Vertices')
    figman.finalize('polygon_area_vs_num_verts.png')
    # ---

    # ---
    ax = figman.figure(fnum=1, doclf=True).gca()
    sns.histplot(data=perannot_data, x='rt_area', ax=ax, kde=True)
    figman.labels.relabel(ax)
    ax.set_title('Polygon sqrt(Area) Histogram')
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylabel('Number of Annotations')
    ax.set_yscale('symlog')
    figman.finalize('polygon_area_histogram.png')
    # ---

    # ---
    ax = figman.figure(fnum=1, doclf=True).gca()
    sns.histplot(data=perannot_data, x='num_vertices', ax=ax)
    ax.set_title('Polygon Number of Vertices Histogram')
    ax.set_ylabel('Number of Annotations')
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_yscale('linear')
    figman.labels.relabel(ax)
    figman.finalize('polygon_num_vertices_histogram.png')
    # ---

    # ---
    ax = figman.figure(fnum=1, doclf=True).gca()
    sns.histplot(data=perimage_data, x='anns_per_image', ax=ax, binwidth=1)
    ax.set_yscale('linear')
    ax.set_xlabel('Number of Annotations')
    ax.set_ylabel('Number of Images')
    ax.set_title('Number of Annotations per Image')
    ax.set_xlim(0, ax.get_xlim()[1])
    figman.labels.relabel(ax)
    figman.finalize('anns_per_image_histogram.png')
    # ---

    # ---
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
    ax.set_title('All Polygons')
    ax.set_aspect('equal')
    ax.set_xlim(0, annot_max_x)
    ax.set_ylim(0, annot_max_y)
    ax.invert_yaxis()
    figman.labels.relabel(ax)
    ax.set_ylim(0, annot_max_y)  # not sure why this needs to be after the relabel, should ideally fix that.
    figman.finalize('all_polygons.png', tight_layout=0)  # tight layout seems to cause issues here
    # ---


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
    """

    CommandLine:
        python ~/code/shitspotter/shitspotter/cli/coco_annotation_stats.py
        python -m shitspotter.cli.coco_annotation_stats $HOME/data/dvc-repos/shitspotter_dvc/data.kwcoco.json
    """
    __cli__.main()
