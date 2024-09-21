# from shitspotter.util.util_math import Rational
from kwutil.util_math import Rational
from dateutil.parser import parse as parse_datetime
import pandas as pd
import ubelt as ub
import numpy as np


def update_analysis_plots():
    """
    Update all of the dataset analytics
    """
    import shitspotter
    import kwplot
    kwplot.autoplt()

    coco_dset = shitspotter.open_shit_coco()

    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    dump_dpath = (ub.Path(coco_dset.bundle_dpath) / 'analysis').ensuredir()
    print('coco_dset = {!r}'.format(coco_dset))

    fig = data_over_time(coco_dset, fnum=1)
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.5)
    fig.tight_layout()
    fig.savefig(dump_dpath / 'images_over_time.png')

    fig = spacetime_scatterplot(coco_dset)
    # fig.set_size_inches(np.array([6.4, 4.8]) * 1.5)
    fig.set_size_inches(np.array([6.4, 4.8]) * 1.5)
    fig.tight_layout()
    fig.savefig(dump_dpath / 'scat_scatterplot.png')

    doggos(dump_dpath / 'doggos.jpg')

    fig = show_3_images(coco_dset, dump_dpath)

    dump_demo_warp_img(coco_dset, dump_dpath)


def data_over_time(coco_dset, fnum=1):
    """
    import shitspotter
    coco_dset = shitspotter.open_shit_coco()
    """
    rows = []
    for gid, img in coco_dset.index.imgs.items():
        row = img.copy()
        rows.append(row)
    img_df = pd.DataFrame(rows)

    img_df = img_df.sort_values('datetime')
    img_df['collection_size'] = np.arange(1, len(img_df) + 1)

    img_df['pd_datetime'] = pd.to_datetime(img_df.datetime)

    import kwplot
    sns = kwplot.autosns()
    fig = kwplot.figure(fnum=3, doclf=True)
    ax = fig.gca()
    sns.histplot(data=img_df, x='pd_datetime', ax=ax, cumulative=True)
    sns.lineplot(data=img_df, x='pd_datetime', y='collection_size')
    ax.set_title('Images collected over time')
    return fig


def spacetime_scatterplot(coco_dset):
    """
    Ignore:
        import shitspotter
        import kwplot
        kwplot.autoplt()
        coco_dset = shitspotter.open_shit_coco()

    References:
        https://catherineh.github.io/programming/2017/10/24/emoji-data-markers-in-matplotlib

    Ignore:
        import matplotlib.pyplot as plt

        pip install mplcairo

        import matplotlib
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import mplcairo
        matplotlib.use("module://mplcairo.qt")
        print(matplotlib.get_backend())

        symbola = ub.grabdata('https://github.com/gearit/ttf-symbola/raw/master/Symbola.ttf')

        # fm.findSystemFonts()
        # '../fonts/Twitter Colour Emoji.ttf')
        fm.fontManager.addfont(symbola)

        # fm.findSystemFonts('../fonts/Twitter Colour Emoji.ttf')
        # fm.fontManager.addfont('../fonts/Twitter Colour Emoji.ttf')

        fig, ax = plt.subplots()
        t = ax.text(.5, .5, b'\xF0\x9F\x98\x85\xf0\x9f\x98\x8d\xF0\x9F\x98\x85'.decode('utf-8'), fontname='symbola', fontsize=30, ha='center')
        t = ax.text(.5, .25, '😅💩😍😅', fontname='symbola', fontsize=30, ha='center')

        import unicodedata
        unicodedata.name('💩')
    """
    import geopandas as gpd
    from shapely import geometry
    from pyproj import CRS
    import kwplot
    sns = kwplot.autosns()

    # image_locs
    rows = []
    for gid, img in coco_dset.index.imgs.items():
        row = img.copy()
        if 'geos_point' in img:
            geos_point = img['geos_point']
            if isinstance(geos_point, dict):
                coords = geos_point['coordinates']
                point = [Rational.coerce(x) for x in coords]
                row['geometry'] = geometry.Point(point)
                rows.append(row)

    img_locs = gpd.GeoDataFrame(rows, crs='crs84')

    # ip_pt = ip_loc.iloc[0].geometry
    utm_crs = CRS.from_epsg(utm_epsg_from_latlon(img_locs.iloc[0].geometry.y, img_locs.iloc[0].geometry.x))  # NOQA
    img_utm_loc = img_locs.to_crs(utm_crs)
    img_utm_xy = np.array([(p.x, p.y) for p in img_utm_loc.geometry.values])

    # import requests
    # location = requests.get('http://ipinfo.io')
    # ip_lat, ip_lon = list(map(Rational.from_float, map(float, location.json()['loc'].split(','))))
    # ip_loc = gpd.GeoDataFrame({'geometry': [geometry.Point(ip_lat, ip_lon)]}, crs='crs84')
    # ip_utm_loc = ip_loc.to_crs(utm_crs)
    # ip_utm_xy = np.array([(p.x, p.y) for p in ip_utm_loc.geometry.values])

    center_utm_xy = np.median(img_utm_xy, axis=0)
    distances = ((img_utm_xy - center_utm_xy) ** 2).sum(axis=1) ** 0.5
    distances = np.array(distances)
    img_locs['distance'] = distances
    datetimes = [parse_datetime(x) for x in img_locs['datetime']]
    img_locs['datetime_obj'] = datetimes
    img_locs['timestamp'] = [x.timestamp() for x in datetimes]
    img_locs['date'] = [x.date() for x in datetimes]
    img_locs['time'] = [x.time() for x in datetimes]
    img_locs['year_month'] = [x.strftime('%Y-%m') for x in datetimes]
    img_locs['hour_of_day'] = [z.hour + z.minute / 60 + z.second / 3600 for z in img_locs['time']]
    # (img_locs['timestamp'] / (60 * 60)) % (24)
    hue_labels = sorted(img_locs['year_month'].unique())
    # dt = img_locs['datetime'].iloc[0]
    # date = dt.date()
    # time = dt.time()

    # img_locs['only_paired'] = img_locs['datetime'] < change_point
    # num_pair_images = img_locs['only_paired'].sum()
    # num_triple_images = (~img_locs['only_paired']).sum()
    # num_trip_groups = num_triple_images // 3
    # pair_groups = num_pair_images // 2
    # print('num_trip_groups = {!r}'.format(num_trip_groups))
    # print('pair_groups = {!r}'.format(pair_groups))

    fig = kwplot.figure(fnum=1, doclf=1)
    ax = sns.scatterplot(data=img_locs, x='distance', y='hour_of_day', facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, 0))
    # hue='year_month')
    ax.set_xscale('log')
    ax.set_xlabel('Distance from home (meters)')
    ax.set_ylabel('Time of day (hours)')
    ax.set_title('Distribution of Images (n={})'.format(len(img_locs)))

    text = '💩'

    label_to_color = ub.dzip(hue_labels, kwplot.Color.distinct(len(hue_labels)))

    chunked = list(ub.chunks(label_to_color.items(), nchunks=24))
    selected = [c[0] for c in chunked[:-1]] + [chunked[-1][-1]]
    selected = dict(selected)

    emoji_plot_pil(ax, img_locs, text, label_to_color)
    # emoji_plot_font(ax, img_locs, text, label_to_color)

    # idx = img_locs['distance'].argmin()
    # Answer jasons question
    # cand = img_locs[img_locs['distance'] < 10]
    # pt = cand.iloc[cand['distance'].argmin()]
    # name = pt['name']
    # name = img_locs.iloc[idx]['name']

    kwplot.phantom_legend(selected)
    return fig


def _configure_osm():
    """
    Configure open street map
    """
    import osmnx as ox
    import ubelt as ub
    import os
    osm_settings_dirs_varnames = [
        'data_folder',
        'logs_folder',
        'imgs_folder',
        'cache_folder',
    ]
    # Make osm dirs point at a standardized location
    osm_cache_root = ub.Path.appdir('osm')
    for varname in osm_settings_dirs_varnames:
        val = ub.Path(getattr(ox.settings, varname))
        if not val.is_absolute():
            new_val = os.fspath(osm_cache_root / os.fspath(val))
            setattr(ox.settings, varname, new_val)

    ox.settings.log_console = True
    ox.settings.log_console = True
    return ox


def plot_on_map(coco_dset):
    """
    Ignore:
        import shitspotter
        import kwplot
        kwplot.autoplt()
        coco_dset = shitspotter.open_shit_coco()
    """
    import geopandas as gpd
    from shapely import geometry
    import kwplot
    # from pyproj import CRS
    sns = kwplot.autosns()  # NOQA
    rows = []
    for gid, img in coco_dset.index.imgs.items():
        row = img.copy()
        if 'geos_point' in img:
            geos_point = img['geos_point']
            if isinstance(geos_point, dict):
                coords = geos_point['coordinates']
                point = [Rational.coerce(x) for x in coords]
                row['geometry'] = geometry.Point(point)
                rows.append(row)
    img_locs = gpd.GeoDataFrame(rows, crs='crs84')
    utm_zones = [utm_epsg_from_latlon(geom.y, geom.x) for geom in img_locs.geometry]
    img_locs['utm_zones'] = utm_zones

    # TODO: cluster into UTM zones?
    # import geodatasets
    # wld_map_gdf = gpd.read_file(geodatasets.get_path('naturalearth.land'))
    ox = _configure_osm()

    for utm_zone, group in img_locs.groupby('utm_zones'):
        fig = kwplot.figure(doclf=True, fnum=utm_zone)
        ax = fig.gca()

        import kwimage
        box = kwimage.Box.coerce([group.bounds.minx.min(), group.bounds.maxx.max(),
                                  group.bounds.miny.min(), group.bounds.maxy.max()], format='xxyy')
        region_geom = box.to_polygon().to_shapely()

        osm_graph = ox.graph_from_polygon(region_geom)

        ax = kwplot.figure(fnum=1, docla=True).gca()
        fig, ax = ox.plot_graph(osm_graph, bgcolor='lawngreen', node_color='dodgerblue', edge_color='skyblue', ax=ax)
        # group.plot(ax=ax, color='orange')
        emoji_plot_pil2(data=group, x='lon', y='lat', text='💩', hue=None, ax=ax)

        # utm_crs = CRS.from_epsg(utm_zone)
        # img_utm_loc = group.to_crs(utm_crs)
        # img_utm_xy = np.array([(p.x, p.y) for p in img_utm_loc.geometry.values])

        # wld_map_gdf.plot(ax=ax)
        # img_locs.plot(ax=ax, kind='kde')
        # img_utm_loc.plot(ax=ax)
        # img_utm_loc.plot(ax=ax)
        if 0:
            img_locs['lon'] = img_locs.geometry.x
            img_locs['lat'] = img_locs.geometry.y
            fig = kwplot.figure(doclf=True)
            ax = fig.gca()
            sns.scatterplot(data=img_locs, x='lon', y='lat', ax=ax)


def emoji_plot_font(ax, img_locs, text, label_to_color):
    for x, y in zip(img_locs.geometry.x, img_locs.geometry.y):
        ax.annotate(text, xy=(x, y), fontname='symbola', color='brown', fontsize=20)


def emoji_plot_pil(ax, img_locs, text, label_to_color):
    """
    # import kwimage
    # kwplot.imshow(kwimage.stack_images([g.image._A for g in label_to_img.values()]), fnum=2, doclf=True)
    """
    import kwimage
    from PIL import Image, ImageFont, ImageDraw
    # hue_labels
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from kwplot.mpl_make import crop_border_by_color
    # text = 'P'
    # sudo apt install ttf-ancient-fonts-symbola
    # font = ImageFont.truetype('/usr/share/fonts/truetype/Sarai/Sarai.ttf', 60, encoding='unic')
    # font = ImageFont.truetype("/data/Steam/steamapps/common/Proton 6.3/dist/share/wine/fonts/symbol.ttf", 60, encoding='unic')
    label_to_img = {}

    # symbola = ub.grabdata('https://github.com/gearit/ttf-symbola/raw/master/Symbola.ttf')
    # QmQY15hiCfFLXCeFxie1iLhqjzY8fEgGxkT8i3uvrWN4me
    symbola = ub.grabdata(
        'https://github.com/taylor/fonts/raw/master/Symbola.ttf',
        hash_prefix='65d634649ab3c4e718b376db0d2e7566d8cfccfff12c70fb3ae2e29a',
        hasher='sha512',
    )

    font = ImageFont.truetype(symbola, 32, encoding='unic')
    for label, color in label_to_color.items():
        col = kwimage.Color(list(color) + [1]).as255()
        pil_img = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        pil_draw = ImageDraw.Draw(pil_img)
        pil_draw.text((0, 0), text, col, font=font)
        img = np.asarray(pil_img)
        img = crop_border_by_color(img, (255, 255, 255, 0))
        image_box = OffsetImage(img, zoom=0.5)
        label_to_img[label] = image_box

    image_box = OffsetImage(img, zoom=0.5)
    for _, row in img_locs.iterrows():
        x = row.distance
        y = row.hour_of_day
        label = row.year_month
        image_box = label_to_img[label]
        ab = AnnotationBbox(image_box, (x, y), frameon=False)
        ax.add_artist(ab)


def emoji_plot_pil2(data, x, y, text, hue=None, ax=None):
    """
    # import kwimage
    # kwplot.imshow(kwimage.stack_images([g.image._A for g in label_to_img.values()]), fnum=2, doclf=True)
    """
    import kwimage
    from PIL import Image, ImageFont, ImageDraw
    # hue_labels
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from kwplot.mpl_make import crop_border_by_color
    # text = 'P'
    # sudo apt install ttf-ancient-fonts-symbola
    # font = ImageFont.truetype('/usr/share/fonts/truetype/Sarai/Sarai.ttf', 60, encoding='unic')
    # font = ImageFont.truetype("/data/Steam/steamapps/common/Proton 6.3/dist/share/wine/fonts/symbol.ttf", 60, encoding='unic')
    label_to_img = {}

    # symbola = ub.grabdata('https://github.com/gearit/ttf-symbola/raw/master/Symbola.ttf')
    # QmQY15hiCfFLXCeFxie1iLhqjzY8fEgGxkT8i3uvrWN4me
    symbola = ub.grabdata(
        'https://github.com/taylor/fonts/raw/master/Symbola.ttf',
        hash_prefix='65d634649ab3c4e718b376db0d2e7566d8cfccfff12c70fb3ae2e29a',
        hasher='sha512',
    )

    font = ImageFont.truetype(symbola, 32, encoding='unic')

    if hue is None:
        col = kwimage.Color.coerce('orange', alpha=1).as255()
        pil_img = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
        pil_draw = ImageDraw.Draw(pil_img)
        pil_draw.text((0, 0), text, col, font=font)
        img = np.asarray(pil_img)
        img = crop_border_by_color(img, (255, 255, 255, 0))
        default_image_box = OffsetImage(img, zoom=0.5)
    else:
        raise NotImplementedError
        label_to_color = None
        for label, color in label_to_color.items():
            col = kwimage.Color(list(color) + [1]).as255()
            pil_img = Image.new("RGBA", (64, 64), (255, 255, 255, 0))
            pil_draw = ImageDraw.Draw(pil_img)
            pil_draw.text((0, 0), text, col, font=font)
            img = np.asarray(pil_img)
            img = crop_border_by_color(img, (255, 255, 255, 0))
            image_box = OffsetImage(img, zoom=0.5)
            label_to_img[label] = image_box

    xy = data[[x, y]]
    image_box = OffsetImage(img, zoom=0.5)
    for _, row in xy.iterrows():
        row = row.to_dict()
        x_pos = row[x]
        y_pos = row[y]
        if hue is None:
            image_box = default_image_box
        else:
            # label = row.year_month
            image_box = label_to_img[label]
        ab = AnnotationBbox(image_box, (x_pos, y_pos), frameon=False)
        ax.add_artist(ab)


def show_data_around_name(coco_dset, name):
    base_img = coco_dset.index.name_to_img[name]
    gid1 = base_img['id']

    chosen_gids = [gid1, gid1 + 1, gid1 + 2]
    images = []
    import kwimage
    import kwplot
    kwplot.autompl()
    import numpy as np
    for coco_img in coco_dset.images(chosen_gids).coco_images:
        imdata = coco_img.delay().finalize()
        rchip, sf_info = kwimage.imresize(imdata, max_dim=800, return_info=True)
        rchip = np.rot90(rchip, k=3)
        images.append(rchip)

    images[0] = kwimage.draw_header_text(images[0], 'Before')
    images[1] = kwimage.draw_header_text(images[1], 'After')
    images[2] = kwimage.draw_header_text(images[2], 'Negative')

    canvas = kwimage.stack_images(images, pad=10, axis=1)
    kwplot.imshow(canvas, fnum=2)


def doggos(dump_fpath):
    import kwimage
    from skimage import exposure  # NOQA
    # from skimage.exposure import match_histograms
    raw_images = [
        kwimage.imread('/home/joncrall/Pictures/PXL_20210721_131129724.jpg'),
        kwimage.imread('/home/joncrall/Pictures/PXL_20210724_160822858.jpg'),
        kwimage.imread('/home/joncrall/Pictures/dogos-with-cookies.jpg'),
        kwimage.imread('/home/joncrall/Pictures/IMG_20201112_111841954.jpg'),

        kwimage.imread('/home/joncrall/Pictures/IMG_20190804_172127271_HDR.jpg'),
        kwimage.imread('/home/joncrall/Pictures/PXL_20210622_172920627.jpg'),
        kwimage.imread('/home/joncrall/Pictures/PXL_20210223_022651672.jpg'),
        kwimage.imread('/home/joncrall/Pictures/PXL_20210301_011213537.jpg'),
    ]

    images2 = [kwimage.imresize(kwimage.ensure_float01(img), max_dim=1024) for img in raw_images]
    # ref = images2[1]
    # ref = kwimage.stack_images_grid(images2[0:1], chunksize=4)
    # images2 = [match_histograms(img, ref, multichannel=False) for img in images2]
    canvas = kwimage.stack_images_grid(images2, chunksize=4)
    # fpath = str(dpath / 'dogos.jpg')
    fpath = dump_fpath
    kwimage.imwrite(dump_fpath, kwimage.ensure_uint255(canvas))
    return fpath


def show_data_diversity(coco_dset):
    names = [
        'PXL_20211106_172807114.jpg',
        'PXL_20210603_215310039.jpg',
        'IMG_20201112_112429442.jpg',
        'PXL_20210228_040756340.jpg',
        'PXL_20210209_150443866.jpg',

        'PXL_20210321_201238147.jpg',
        'PXL_20210125_140106674.jpg',
        'PXL_20201116_141610597.jpg',
        'PXL_20210823_041957725.jpg',
    ]

    images = []
    import kwimage
    import numpy as np
    for name in names:
        img = coco_dset.index.name_to_img[name]
        gpath = coco_dset.get_image_fpath(img['id'])
        imdata1 = kwimage.imread(gpath)
        rchip1, sf_info1 = kwimage.imresize(imdata1, max_dim=800, return_info=True)
        if rchip1.shape[0] > rchip1.shape[1]:
            rchip1 = np.rot90(rchip1)
        # coco_img = coco_dset.coco_image(img['id'])
        # imdata1 = coco_img.delay().finalize()
        images.append(rchip1)

    canvas = kwimage.stack_images_grid(images, pad=10)
    kwimage.imwrite('viz_shit_dataset_sample.jpg', canvas)
    import kwplot
    kwplot.autompl()
    kwplot.imshow(canvas, fnum=1)


def show_3_images(coco_dset, dump_dpath):
    """
    import shitspotter
    coco_dset = shitspotter.open_shit_coco()
    dump_dpath = (ub.Path(coco_dset.bundle_dpath) / 'analysis').ensuredir()
    """
    import kwimage
    import kwplot
    kwplot.autompl()

    # all_gids = list(coco_dset.images())
    # n = 0
    # n += 3
    # gids = all_gids[910 + n:]
    # chosen_gids = gids[0:3]

    names = [
        'PXL_20210525_190443057',
        'PXL_20210525_190539541',
        'PXL_20210525_190547010.MP']
    chosen_gids = list(coco_dset.images(names=names))

    images = []
    import numpy as np
    coco_images = coco_dset.images(chosen_gids).coco_images
    for coco_img in coco_images:
        imdata = coco_img.delay().finalize()
        if 1:
            truth = coco_img.annots().detections
            if len(truth):
                enlarged = truth.boxes.scale(2.0, about='centroid').astype(int)
                imdata = enlarged.draw_on(imdata, color='kitware_orange', thickness=32)
        rchip, sf_info = kwimage.imresize(imdata, max_dim=800, return_info=True)
        rchip = np.rot90(rchip, k=3)
        images.append(rchip)

    images[0] = kwimage.draw_header_text(images[0], 'Before')
    images[1] = kwimage.draw_header_text(images[1], 'After')
    images[2] = kwimage.draw_header_text(images[2], 'Negative')

    canvas = kwimage.stack_images(images, pad=10, axis=1)
    fpath = dump_dpath / 'viz_three_images.jpg'
    kwimage.imwrite(fpath, canvas)

    # figman = kwplot.FigureManager()
    # fig = kwplot.figure(fnum=1, doclf=True)
    # kwplot.imshow(canvas, fnum=1)

    # fig.set_size_inches(np.array([6.4, 4.8]) * 1.5)
    # fig.tight_layout()
    # figman.finalize(fpath, fig=fig)
    # return fig


def dump_demo_warp_img(coco_dset, dump_dpath):
    import kwplot
    import kwimage
    kwplot.autompl()
    # gid1, gid2 = (3, 4)
    # gid1, gid2 = (30, 31)
    # gid1, gid2 = (34, 35)
    # gid1, gid2 = (99, 100)

    gid1, gid2 = list(coco_dset.images(
        names=['PXL_20210525_190443057', 'PXL_20210525_190539541']))

    # fig, fig2 = demo_warp(coco_dset, gid1, gid2)
    figs = demo_warp(coco_dset, gid1, gid2)

    fig1, fig2, fig3, fig4 = figs
    # for fig in figs:
    #     fig.canvas.draw()

    ##
    ####
    # Plotting

    # fig2.set_size_inches(25.6 , 13.37)
    fig1.tight_layout()

    fig1.set_size_inches(14.4 / 2, 24.84 / 2)
    fig1.tight_layout()
    toshow_im1 = kwplot.render_figure_to_image(fig1)
    tosave_im1 = kwplot.mpl_make.crop_border_by_color(toshow_im1)
    kwimage.imwrite(dump_dpath / 'viz_align_process.png', tosave_im1)

    fig2.set_size_inches(8 * 0.8, 13 * 0.8)
    fig2.tight_layout()
    toshow_im2 = kwplot.render_figure_to_image(fig2)
    tosave_im2 = kwplot.mpl_make.crop_border_by_color(toshow_im2)
    kwimage.imwrite(dump_dpath / 'viz_candidate_ann.png', tosave_im2)

    fig3.set_size_inches(16 * 0.8, 13 * 0.8)
    fig3.tight_layout()
    toshow_im3 = kwplot.render_figure_to_image(fig3)
    tosave_im3 = kwplot.mpl_make.crop_border_by_color(toshow_im3)
    kwimage.imwrite(dump_dpath / 'viz_diff_to_binthresh.png', tosave_im3)

    fig4.set_size_inches(24 * 1.0, 13 * 1.0)
    fig4.tight_layout()
    toshow_im3 = kwplot.render_figure_to_image(fig3)
    tosave_im3 = kwplot.mpl_make.crop_border_by_color(toshow_im3)
    kwimage.imwrite(dump_dpath / 'viz_align_upclose.png', tosave_im3)

    # insepctor = match.ishow()
    # print('insepctor = {!r}'.format(insepctor))

    # display_cfg = {
    #     'overlay': False,
    #     'with_homog': True,
    # }
    # insepctor = MatchInspector(match=match)
    # insepctor.initialize(match=match, cfgdict=display_cfg)
    # insepctor.show()
    # match.ishow()


def iter_warp(coco_dset):
    import xdev
    pairs = list(ub.iter_window(list(coco_dset.images()), 2))
    for gid1, gid2 in xdev.InteractiveIter(pairs):
        print('gid1, gid2 = {!r}, {}'.format(gid1, gid2))
        figs = demo_warp(coco_dset, gid1, gid2)
        for fig in figs:
            fig.canvas.draw()
        pass


def demo_warp(coco_dset, gid1, gid2, overview=2, max_sidelen=512,
              ratio_thresh=0.625):
    """
    import shitspotter
    coco_dset = shitspotter.open_shit_coco()

    from kwutil import util_time

    # name = 'PXL_20201122_163121561'
    name = 'PXL_20211125_175720394'
    print([n for n in coco_dset.index.name_to_img if name in n])

    img1 = coco_dset.index.name_to_img[name]

    coco_img1 = coco_dset.coco_image(img1['id'])
    coco_images = coco_dset.images().coco_images
    after = [c for c in coco_images if c.datetime > coco_img1.datetime]
    after = sorted(after, key=lambda c: c.datetime)
    coco_img2 =after[0]

    overview = 1
    max_sidelen = None
    ratio_thresh = 0.625
    gid1, gid2 = coco_img1.img['id'], coco_img2.img['id']

    # gid1, gid2 = 24, 25
    # gid1, gid2 = 339, 340  # paralax
    """
    import numpy as np
    import kwimage
    import kwplot
    from vtool_ibeis import PairwiseMatch
    import cv2
    kwplot.autompl()

    # gid1, gid2 = (3, 4)
    # gid1, gid2 = (30, 31)
    # gid1, gid2 = (34, 35)
    # gid1, gid2 = (99, jk100)
    # gid1, gid2 = (101, 102)

    # img1 = coco_dset.coco_image(gid1)
    # img2 = coco_dset.coco_image(gid2)
    # imdata1 = img1.delay().finalize()
    # imdata2 = img2.delay().finalize()

    # imdata1 = np.rot90(imdata1)
    # imdata2 = np.rot90(imdata2)
    import shitspotter

    fpath1 = coco_dset.get_image_fpath(gid1)
    fpath2 = coco_dset.get_image_fpath(gid2)
    imdata1 = shitspotter.util.imread_with_exif(fpath1, overview=overview)
    imdata2 = shitspotter.util.imread_with_exif(fpath2, overview=overview)

    if 0:
        imdata2 = np.rot90(imdata2, -1)

    maxdim = max(max(imdata1.shape[0:2]), max(imdata2.shape[0:2]))
    if max_sidelen is not None:
        maxdim = max(512, maxdim)

    # rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=512, return_info=True)
    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=512, return_info=True)
    # rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=800, return_info=True)
    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=800, return_info=True)
    # rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=maxdim, return_info=True)
    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=maxdim, return_info=True)

    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=512, return_info=True)
    # rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=800, return_info=True)
    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=800, return_info=True)
    rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=maxdim, return_info=True)
    rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=maxdim, return_info=True)

    undo_scale = kwimage.Affine.coerce(ub.udict(sf_info1) - {'dsize'}).inv()
    # undo_scale = kwimage.Affine.coerce(sf_info1).inv()

    # import kwplot
    # kwplot.autompl()
    # kwplot.imshow(imdata1, fnum=1, pnum=(1, 2, 1))
    # kwplot.imshow(imdata2, fnum=1, pnum=(1, 2, 2))

    # from vtool_ibeis.inspect_matches import MatchInspector

    annot1 = {'rchip': np.ascontiguousarray(rchip1_rgb[:, :, ::-1])}
    annot2 = {'rchip': np.ascontiguousarray(rchip2_rgb[:, :, ::-1])}

    match_cfg = {
        'symetric': False,
        'K': 1,
        # 'ratio_thresh': 0.625,
        'ratio_thresh': ratio_thresh,
        'refine_method': 'homog',
        'rotation_invariance': True,
        'affine_invariance': True,
    }
    match = PairwiseMatch(annot1, annot2)
    match.verbose = True
    match.apply_all(cfgdict=match_cfg)

    if 0:
        match.ishow()

    rchip1 = np.ascontiguousarray(kwimage.ensure_float01(match.annot1['rchip'][:, :, ::-1]))
    rchip2 = np.ascontiguousarray(kwimage.ensure_float01(match.annot2['rchip'][:, :, ::-1]))
    score = match.fs.sum()
    print('score = {!r}'.format(score))

    rchip2_dims = rchip2.shape[0:2]
    rchip2_dsize = rchip2_dims[::-1]

    rchip1_dims = rchip1.shape[0:2]
    rchip1_dsize = rchip1_dims[::-1]
    M1 = match.H_12
    rchip1_align = cv2.warpPerspective(rchip1, M1, rchip2_dsize)
    rchip1_bounds = kwimage.Box.coerce([0, 0, rchip1_dsize[0] - 1, rchip1_dsize[1] - 1], format='xywh').to_polygon()
    rchip2_bounds = kwimage.Box.coerce([0, 0, rchip2_dsize[0] - 1, rchip2_dsize[1] - 1], format='xywh').to_polygon()
    warp_bounds1 = rchip1_bounds.warp(M1)
    shpb1 = warp_bounds1.to_shapely()
    # Only keep valid regions
    warp_bounds1 = kwimage.Polygon.from_shapely(shpb1.intersection(rchip2_bounds.to_shapely()))
    # warp_bounds.draw()
    valid_rchip2_mask1 = warp_bounds1.to_mask(dims=rchip2_dims).data
    rchip2_align = rchip2 * valid_rchip2_mask1[:, :, None]

    if 0:
        _alternate_matching_experimental(rchip1, rchip1_align, rchip2_align,
                                         rchip2_dims, rchip2_dsize,
                                         rchip1_bounds, warp_bounds1, M1)
    else:
        raw1 = kwimage.gaussian_blur(rchip1_align, kernel=7)
        raw2 = kwimage.gaussian_blur(rchip2_align, kernel=7)
        rchip1_refine = rchip1_align
        rchip2_refine = rchip2_align
        M = M1.copy()
    # raw1 = raw1.mean(axis=2, keepdims=True)
    # raw2 = raw2.mean(axis=2, keepdims=True)
    diff_img_RAW = np.linalg.norm(np.abs(raw1 - raw2), axis=2)

    diff_img = diff_img_RAW
    mask = diff_img.copy()
    mask = kwimage.morphology(mask, 'close', kernel=3)
    mask = kwimage.gaussian_blur(mask, sigma=2.0)
    mask = kwimage.morphology(mask, 'close', kernel=7)
    print(sorted(mask.ravel())[-100:])
    mask = kwimage.morphology((mask > 0.4).astype(np.float32), 'dilate', kernel=10)
    mask = kwimage.gaussian_blur(mask, sigma=3.0)
    mask = (mask > 0.2).astype(np.float32)
    mask = kwimage.morphology(mask, 'close', kernel=30)
    mask = kwimage.morphology((mask).astype(np.float32), 'dilate', kernel=30)

    # Warp mask back onto original image
    tf_orig_from_align = np.asarray(undo_scale @ np.linalg.inv(M))
    orig_dsize = imdata1.shape[0:2][::-1]
    mask1_orig = cv2.warpPerspective(mask, tf_orig_from_align, orig_dsize)

    overlay = kwimage.ensure_alpha_channel(mask1_orig[..., None])
    overlay[mask1_orig > 0.5, 3] = 0.0
    overlay[mask1_orig < 0.5, 3] = 0.8
    orig = kwimage.ensure_alpha_channel(kwimage.ensure_float01(imdata1))
    attention_imdata1 = kwimage.overlay_alpha_images(overlay, orig)
    # attention_imdata1 = imdata1 * mask1_orig[..., None]

    rstack = kwimage.stack_images([rchip1, rchip2], pad=10, axis=1)

    rchip1_align_a = kwimage.ensure_alpha_channel(rchip1_refine, 0.5)
    rchip2_align_a = kwimage.ensure_alpha_channel(rchip2_refine, 1.0)
    overlay_align = kwimage.overlay_alpha_layers([rchip1_align_a, rchip2_align_a])
    # kwplot.imshow(overlay_align, fnum=6, title='Aligned')
    align_stack1 = kwimage.stack_images([rchip1_refine, rchip2_refine, overlay_align], pad=10, axis=1)
    # align_stack2 = kwimage.stack_images([rchip1_align_final, rchip2_align_final], pad=10, axis=1)
    diff_stack = kwimage.stack_images([diff_img, mask], pad=10, axis=1)

    fig1 = kwplot.figure(fnum=1, doclf=True)
    pnum_a = kwplot.PlotNums(nRows=3, nCols=1)

    kwplot.imshow(rstack, fnum=1, pnum=pnum_a[0], title='Raw Before / After Image Pair')
    ax = kwplot.figure(fnum=1, pnum=pnum_a[1]).gca()
    ax.set_title('SIFT Features Matches (used to align the images)')
    match.show(ax=ax, show_ell=1, show_lines=False, ell_alpha=0.2, vert=False)

    kwplot.imshow(align_stack1, fnum=1, pnum=pnum_a[2], title='Aligned Images')
    # kwplot.imshow(align_stack2, fnum=3, pnum=pnum_a[3], title='Refined Alignment')

    fig3 = kwplot.figure(fnum=3, doclf=True)
    kwplot.imshow(diff_stack, fnum=3, title='Difference Image -> Binary Mask')
    # kwplot.imshow(attention_rchip1, fnum=3, pnum=pnum_a[4], title='Candidate Annotation Regions')

    fig2 = kwplot.figure(fnum=2, doclf=True)
    kwplot.imshow(attention_imdata1, fnum=2, title='Candidate Annotation Regions')

    fig4 = kwplot.figure(fnum=4, doclf=True)
    kwplot.imshow(align_stack1, fnum=4, title='Aligned')
    return fig1, fig2, fig3, fig4


def _alternate_matching_experimental(rchip1, rchip1_align, rchip2_align,
                                     rchip2_dims, rchip2_dsize, rchip1_bounds,
                                     warp_bounds1, M1):
    import kwimage
    import kwplot
    import numpy as np
    from vtool_ibeis import PairwiseMatch
    import cv2
    if 0:
        """
        pip install itk-elastix
        https://github.com/InsightSoftwareConsortium/ITKElastix/issues/159
        pip install itk==v5.3rc04.post1
        pip install itk==v5.3rc4

        python -c "import itk; itk.elastix_registration_method"

        pip install itk==v5.3rc3

        https://github.com/Erotemic/local/blob/main/tools/supported_python_versions_pip.py

        pip install itk-elastix==[0.14.0]
        pip install itk-filtering==5.3rc2
        pip install itk-filtering==5.3rc2
        pip install itk==v5.3rc04.post1
        """
        # import itk
        # fixed_image:
        # registered_image, params = itk.elastix_registration_method(fixed_image, moving_image)
    if 0:
        # TODO: diffeomorphism - Diffeomorphic Demons
        # https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/66_Registration_Demons.html
        # https://www.sci.utah.edu/~wolters/LiteraturZurVorlesung/Literatur/F:_Anisotropy/Vercauteren_DiffeomorphicDemons-Paper_2007.pdf
        # https://discourse.itk.org/t/python-code-for-diffeomorphicdemonsregistration-in-simpleitk/1937
        # https://www.cs.ucf.edu/~bagci/teaching/mic16/lec17.pdf
        import SimpleITK as sitk
        fixed = sitk.GetImageFromArray(rchip1_align.mean(axis=2))
        moving = sitk.GetImageFromArray(rchip2_align.mean(axis=2))

        numberOfBins = 36
        samplingPercentage = 0.10
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsMattesMutualInformation(numberOfBins)
        R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
        R.SetMetricSamplingStrategy(R.RANDOM)
        R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 200)
        R.SetInitialTransform(sitk.TranslationTransform(fixed.GetDimension()))
        R.SetInterpolator(sitk.sitkBSpline)
        outTx = R.Execute(fixed, moving)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed)
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(outTx)
        out = resampler.Execute(moving)
        arr1 = kwimage.atleast_3channels(sitk.GetArrayFromImage(fixed))
        arr2 = kwimage.atleast_3channels(sitk.GetArrayFromImage(out))
        # arr1 = kwimage.gaussian_blur(arr1, sigma=2.0)
        # arr2 = kwimage.gaussian_blur(arr2, sigma=2.0)
        kwplot.imshow(arr1, fnum=1)
        kwplot.imshow(arr2, fnum=2)
        diff_img_MI = np.abs(arr1 - arr2)
        kwplot.imshow(diff_img_MI, fnum=6)
        raw1 = arr1
        raw2 = arr2
        rchip1_refine = arr1
        rchip2_refine = arr2
        M = M1.copy()
        # TODO: need to be able to invert transform before we can use thi
    else:
        if 0:
            # Refine?
            # TODO: an iterative matching approach seems to work well
            annot1_refine = {'rchip': np.ascontiguousarray(kwimage.ensure_uint255(rchip1_align[:, :, ::-1]))}
            annot2_refine = {'rchip': np.ascontiguousarray(kwimage.ensure_uint255(rchip2_align[:, :, ::-1]))}
            match_cfg2 = {
                'symetric': True,
                'K': 1,
                'ratio_thresh': 0.625,
                'refine_method': 'homog',
                'rotation_invariance': False,
                'affine_invariance': True,
            }
            match2 = PairwiseMatch(annot1_refine, annot2_refine)
            match2.apply_all(cfgdict=match_cfg2)

            M2 = match2.H_12
            # rchip1_refine = cv2.warpPerspective(rchip1_align, M2, rchip2_dsize)
            rchip1_refine = cv2.warpPerspective(rchip1, M2 @ M1, rchip2_dsize)
            warp_bounds2 = warp_bounds1.warp(M2)
            if 0:

                rchip1_bounds.draw(alpha=0.5, color='red')
                warp_bounds1.draw(alpha=0.5, color='orange')
                warp_bounds2.draw(alpha=0.5, color='green')
            # warp_bounds2.draw()
            valid_rchip2_mask2 = warp_bounds2.to_mask(dims=rchip2_dims).data
            rchip2_refine = rchip2_align * valid_rchip2_mask2[:, :, None]
            rchip1_refine = rchip1_refine * valid_rchip2_mask2[:, :, None]

            raw1 = kwimage.gaussian_blur(rchip1_refine, kernel=7)
            raw2 = kwimage.gaussian_blur(rchip2_refine, kernel=7)
            M = M2 @ M1

    x = raw1, raw2, M
    # kwplot.imshow(diff_img_RAW, fnum=7)

    # import itk
    # image_type = itk.Image[itk.F, 2]
    # regis = itk.ImageRegistrationMethod[image_type, image_type].New()
    # regis.SetFixedImage(slice1)
    # regis.SetMovingImage(slice2)
    # mi_tf = regis.GetOutput()
    # slice1 = itk.image_from_array(rchip1_align.mean(axis=2))
    # slice2 = itk.image_from_array(rchip2_align.mean(axis=2))
    # diff_filter = itk.SimilarityIndexImageFilter[image_type, image_type].New()
    # diff_filter.SetInput1(slice1)
    # diff_filter.SetInput2(slice2)
    # diff_filter.Update()
    # indx = diff_filter.GetSimilarityIndex()

    # rchip1_align
    # diff_img = np.abs(rchip1_align - rchip2_align)
    # diff_img = np.linalg.norm(diff_img, axis=2)
    # rchip1_align_final = rchip1_align.copy()
    # rchip2_align_final = rchip2_align.copy()
    # rchip1_align_final = raw1.copy()
    # rchip2_align_final = raw2.copy()
    return x


def utm_epsg_from_latlon(lat, lon):
    """
    Find a reasonable UTM CRS for a given lat / lon

    The purpose of this function is to get a reasonable CRS for computing
    distances in meters. If the region of interest is very large, this may not
    be valid.

    Args:
        lat (float): degrees in latitude
        lon (float): degrees in longitude

    Returns:
        int : the ESPG code of the UTM zone

    References:
        https://gis.stackexchange.com/questions/190198/how-to-get-appropriate-crs-for-a-position-specified-in-lat-lon-coordinates
        https://gis.stackexchange.com/questions/365584/convert-utm-zone-into-epsg-code

    Example:
        >>> epsg_code = utm_epsg_from_latlon(0, 0)
        >>> print('epsg_code = {!r}'.format(epsg_code))
        epsg_code = 32631
    """
    import utm
    # easting, northing, zone_num, zone_code = utm.from_latlon(min_lat, min_lon)
    zone_num = utm.latlon_to_zone_number(lat, lon)

    # Construction of EPSG code from UTM zone number
    south = lat < 0
    epsg_code = 32600
    epsg_code += int(zone_num)
    if south is True:
        epsg_code += 100
    return epsg_code


def annotate():
    """
    import kwcoco
    coco_dset = kwcoco.CocoDataset('/data/store/data/shit-pics/data.kwcoco.json')
    """
    import kwimage
    import plottool_ibeis
    import kwplot
    import shitspotter
    coco_dset = shitspotter.open_shit_coco()

    plt = kwplot.autoplt(force='Qt5Agg')
    gpaths = [c.primary_image_filepath() for c in coco_dset.images().coco_images]

    img_ind = 4
    imdata = kwimage.imread(gpaths[img_ind], space='bgr')

    def commit_callback(*args, **kw):
        print('commit args = {!r}'.format(args))
        print('commit kw = {!r}'.format(kw))

    def prev_callback(*args, **kw):
        print('prev_callback args = {!r}'.format(args))
        print('prev_callback kw = {!r}'.format(kw))

    def next_callback(*args, **kw):
        print('next_callback args = {!r}'.format(args))
        print('next_callback kw = {!r}'.format(kw))

    verts_list = None
    bbox_list = []
    theta_list = []
    species_list = []
    metadata_list = []

    interact_obj = plottool_ibeis.interact_annotations.AnnotationInteraction(
        img=imdata,
        img_ind=img_ind,
        commit_callback=commit_callback,
        verts_list=verts_list,
        bbox_list=bbox_list,
        theta_list=theta_list,
        species_list=species_list,
        metadata_list=metadata_list,
        prev_callback=prev_callback,
        next_callback=next_callback,
    )
    interact_obj.start()

    # interact_obj.update_image_and_callbacks(
    #     imdata,
    #     bbox_list=bbox_list,
    #     theta_list=theta_list,
    #     species_list=species_list,
    #     metadata_list=metadata_list,
    #     next_callback=nextcb,
    #     prev_callback=prevcb,
    # )

    # interact_obj = plottool_ibeis.interact_multi_image.MultiImageInteraction(gpaths)
    # interact_obj.start()

    plt.show()


def data_on_maps(coco_dset):
    """
    This requires some care to get a reasonable visualization

    Requirements:
        pip install mplcairo osmnx

    Ignore:
        from shitspotter.plots import *  # NOQA
        import shitspotter
        coco_dset = shitspotter.open_shit_coco()

        import matplotlib
        import matplotlib.font_manager as fm
        import matplotlib.pyplot as plt
        import mplcairo
        matplotlib.use("module://mplcairo.qt")
    """
    import networkx as nx
    import geopandas as gpd
    import osmnx as ox
    from shapely import geometry
    from pyproj import CRS
    # image_locs
    rows = []
    for gid, img in coco_dset.index.imgs.items():
        row = img.copy()
        if 'geos_point' in img:
            geos_point = img['geos_point']
            if isinstance(geos_point, dict):
                coords = geos_point['coordinates']
                point = [Rational.coerce(x) for x in coords]
                row['geometry'] = geometry.Point(point)
                rows.append(row)
    img_locs = gpd.GeoDataFrame(rows, crs='crs84')

    crs84 = CRS.from_user_input('crs84')

    # wld_map_crs84_gdf = gpd.read_file(
    #     gpd.datasets.get_path('naturalearth_lowres')
    # ).to_crs(crs84)

    utm_crs = img_locs.estimate_utm_crs()
    img_locs_utm = img_locs.to_crs(utm_crs)

    aoi_utm = gpd.GeoDataFrame({
        'geometry': [img_locs_utm.unary_union.convex_hull]
    }, crs=img_locs_utm.crs)
    mediod_utm = gpd.GeoDataFrame({
        'geometry': [geometry.Point(np.median(np.array([(x.x, x.y) for x in img_locs_utm.geometry]), axis=0))]
    }, crs=img_locs_utm.crs).convex_hull

    mediod_crs84 = mediod_utm.to_crs(crs84)
    aoi_crs84 = aoi_utm.to_crs(crs84)
    medoid_lon, medoid_lat = map(lambda x: x[0], mediod_crs84.iloc[0].xy)
    mediod_wgs84 = (medoid_lat, medoid_lon)

    # import kwplot
    # sns = kwplot.autosns()
    # ax = kwplot.figure().gca()
    # wld_map_crs84_gdf.plot(ax=ax)

    # https://automating-gis-processes.github.io/CSC18/lessons/L3/retrieve-osm-data.html
    if 0:
        graph = ox.graph_from_polygon(aoi_crs84.geometry.iloc[0])

    graph = ox.graph_from_point(mediod_wgs84)
    graphs = []
    # exterior_points = np.array([np.array(_) for _ in img_locs.unary_union.convex_hull.exterior.coords.xy]).T
    # for x, y in ub.ProgIter(list(exterior_points)):
    #     graph = ox.graph_from_point((y, x))
    #     graphs.append(graph)
    combo = nx.disjoint_union_all(graphs + [graph])

    # https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
    import kwplot
    import matplotlib
    import matplotlib.font_manager as fm  # NOQA
    import matplotlib.pyplot as plt  # NOQA
    import mplcairo  # NOQA
    import contextily as cx

    ax = kwplot.figure(fnum=1, docla=True).gca()
    fig, ax = ox.plot_graph(combo, bgcolor='lawngreen', node_color='dodgerblue', edge_color='skyblue', ax=ax)
    matplotlib.use("module://mplcairo.qt")
    print(matplotlib.get_backend())

    for x, y in zip(img_locs.geometry.x, img_locs.geometry.y):
        ax.annotate('💩', xy=(x, y), fontname='symbola', color='brown', fontsize=20)

    if 0:
        minx, miny, maxx, maxy = img_locs.unary_union.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    cx.add_basemap(ax, crs=img_locs.crs)
    # , xytext=(0, 0), textcoords="offset points")
    ax = kwplot.figure(fnum=1, docla=True).gca()

    # import matplotlib
    # import matplotlib.font_manager as fm
    # import matplotlib.pyplot as plt
    # import mplcairo
    # matplotlib.use("module://mplcairo.qt")
    print(matplotlib.get_backend())
    ax = kwplot.figure(fnum=1, docla=True).gca()
    img_locs.plot(ax=ax)
    for x, y in zip(img_locs.geometry.x, img_locs.geometry.y):
        ax.annotate('💩', xy=(x, y), fontname='symbola', color='brown', fontsize=20)


def demo_osm():
    import osmnx as ox
    point = (42.8505339, -73.7710063)
    graph = ox.graph_from_point(point, dist=1000)

    import kwplot
    kwplot.autompl()
    fig = kwplot.figure()
    ax = fig.gca()
    ox.plot_graph(graph, bgcolor='black', node_color='dodgerblue', edge_color='skyblue', ax=ax)


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/shitspotter/plots.py
    """
    import fire
    fire.Fire()
