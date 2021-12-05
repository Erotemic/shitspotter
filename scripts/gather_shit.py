import fractions
from dateutil.parser import parse as parse_datetime
import os
import dateutil.parser
from os.path import join
import xdev
import pandas as pd
import kwcoco
import ubelt as ub
# import math
import pathlib
import datetime


def main():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/shitspotter/scripts'))
    from gather_shit import *  # NOQA
    """
    dpath = '/data/store/data/shit-pics/'

    rows = []
    seen = set()
    all_fpaths = []
    change_point = dateutil.parser.parse('2021-05-11T120000')
    for r, ds, fs in os.walk(dpath):
        to_remove = []
        for idx, d in enumerate(ds):
            if not d.startswith('poop-'):
                to_remove.append(idx)

        for idx in to_remove[::-1]:
            del ds[idx]

        dname = os.path.relpath(r, dpath)
        if dname.startswith('poop-'):
            timestr = dname.split('poop-')[1]
            datestamp = dateutil.parser.parse(timestr)

            is_double = datestamp < change_point

            for fname in fs:
                gpath = join(r, fname)
                all_fpaths.append(gpath)
                if fname.endswith('.mp4'):
                    continue
                if fname in seen:
                    print('SEEN fname = {!r}'.format(fname))
                    continue
                seen.add(fname)
                rows.append({
                    'gpath': gpath,
                    'name': pathlib.Path(fname).stem,
                    'datestamp': datestamp,
                    'is_double': is_double,
                })

    dupidxs = ub.find_duplicates(all_fpaths, key=lambda x: pathlib.Path(x).name)
    assert len(dupidxs) == 0

    if 0:
        # Test to make sure we didn't mess up
        # TODO: clean up duplicate files
        to_remove = []
        to_keep = []
        for idxs in dupidxs.values():
            dups = list(ub.take(all_fpaths, idxs))
            hashes = [ub.hash_file(d) for d in dups]
            assert ub.allsame(hashes)
            deltas = {}
            folder_dts = []
            image_dts = []
            for d in dups:
                p = pathlib.Path(d)
                parent_dname = p.parent.name
                image_dt = parse_datetime(p.name.split('_')[1])
                folder_dt = parse_datetime(parent_dname.split('-', 1)[1])
                delta = (folder_dt - image_dt)
                folder_dts.append(folder_dt)
                image_dts.append(image_dt)
                assert delta.total_seconds() >= 0
                deltas[d] = delta
            assert len(set(image_dts)) == 1
            keep_gpath = ub.argmin(deltas)
            remove_gpaths = set(dups) - {keep_gpath}
            image_dt = image_dts[0]
            print('image_dt = {!r}'.format(image_dt))
            print('folder_dts = {}'.format(ub.repr2(folder_dts, nl=1)))
            print('deltas = {!r}'.format(deltas))
            print('dups = {!r}'.format(dups))
            to_keep.append(keep_gpath)
            to_remove.extend(list(remove_gpaths))

        assert set(to_remove).isdisjoint(set(to_keep))

        # Dont remove, because I'm afraid to do that programtically atm
        # just move to a trash folder
        dpath = pathlib.Path('/data/store/data/shit-pics/_trash_dups')
        dpath.mkdir(exist_ok=True)
        import shutil
        for p in to_remove:
            p = pathlib.Path(p)
            dst = dpath / p.name
            shutil.move(p, dst)

    if 0:
        import kwimage
        # Test we can read overviews
        fpath = rows[-1]['gpath']
        with ub.Timer('imread-o0'):
            imdata = kwimage.imread(fpath, backend='gdal')
            print('imdata.shape = {!r}'.format(imdata.shape))
        with ub.Timer('imread-o1'):
            imdata = kwimage.imread(fpath, overview=2, backend='gdal')
            print('imdata.shape = {!r}'.format(imdata.shape))

    for row in ub.ProgIter(rows):
        gpath = row['gpath']
        row['nbytes'] = os.stat(gpath).st_size
        row['nbytes_str'] = xdev.byte_str(row['nbytes'])
        exif = extract_exif_metadata(gpath)
        if 'ImageWidth' in row:
            row['width'] = row['ImageWidth']
            row['height'] = row['ImageHeight']
        try:
            # parse_datetime(exif['DateTime'])
            exif_datetime = exif['DateTime']
            dt = datetime.datetime.strptime(exif_datetime, '%Y:%m:%d %H:%M:%S')
        except Exception:
            # dt = date_parser.parse(exif['DateTime'])
            # row['datetime'] = dt.isoformat()
            raise
        # TODO: exif 'OffsetTime': '-05:00',
        row['datetime'] = dt.isoformat()
        exif_ori = exif['Orientation']
        print('exif_ori = {!r}'.format(exif_ori))
        geos_point = exif.get('GPSInfo', None)
        if geos_point is not None and 'GPSLatitude' in geos_point:
            lat_degrees, lat_minutes, lat_seconds = map(Rational.coerce, geos_point['GPSLatitude'])
            lon_degrees, lon_minutes, lon_seconds = map(Rational.coerce, geos_point['GPSLongitude'])
            lat_sign = {'N': 1, 'S': -1}[geos_point['GPSLatitudeRef']]
            lon_sign = {'E': 1, 'W': -1}[geos_point['GPSLongitudeRef']]
            lat = lat_sign * (lat_degrees + lat_minutes / 60 + lat_seconds / 3600)
            lon = lon_sign * (lon_degrees + lon_minutes / 60 + lon_seconds / 3600)
            # Can geojson handle rationals?
            row['geos_point'] = {'type': 'Point', 'coordinates': (lon.__smalljson__(), lat.__smalljson__()), 'properties': {'crs': 'CRS84'}}

    img_info_df = pd.DataFrame(rows)
    img_info_df = img_info_df.sort_values('datetime')
    print(img_info_df)
    print(xdev.byte_str(sum(img_info_df.nbytes)))

    coco_dset = kwcoco.CocoDataset()
    for row in img_info_df.to_dict('records'):
        row = row.copy()
        row.pop('nbytes_str', None)
        row.pop('is_double', None)
        row['file_name'] = row.pop('gpath')
        row.pop('datestamp', None)
        coco_dset.add_image(**row)

    coco_dset.conform(workers=8)
    coco_dset.validate()

    coco_dset.fpath = str(pathlib.Path(dpath) / 'data.kwcoco.json')
    coco_dset._check_json_serializable()
    coco_dset._ensure_json_serializable()
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    coco_dset.reroot(absolute=False)
    coco_dset.dump(coco_dset.fpath, newlines=True)


def check_exif_orientation(coco_dset):
    """
    Notes on orientation:
        https://jdhao.github.io/2019/07/31/image_rotation_exif_info/

        1: Upright
        8: Rot 90 clockwise
        3: Rot 180
        6: Rot 270 clockwise

        2: Flip + Upright
        7: Flip + Rot 90 clockwise
        4: Flip + Rot 180
        5: Flip + Rot 270 clockwise
    """
    import kwplot
    import xdev
    import kwimage
    kwplot.autompl()
    gids = list(coco_dset.index.imgs.keys())
    giditer = xdev.InteractiveIter(gids)
    for gid in giditer:
        # for gid in gids:
        fpath = coco_dset.get_image_fpath(gid)
        exif = extract_exif_metadata(fpath)
        exif_ori = exif.get('Orientation', None)
        print('exif_ori = {!r}'.format(exif_ori))
        # 'ExifImageHeight': 3024,
        # 'ExifImageWidth': 4032,
        # 'ImageLength': 3024,
        # 'ImageWidth': 4032,
        # Reading with GDAL/cv2 will NOT apply any exif orientation
        # but reading with skimage will
        imdata = kwimage.imread(fpath, backend='gdal', overview=-1)

        kwplot.imshow(imdata)
        xdev.InteractiveIter.draw()


def data_on_maps(coco_dset):
    """
    import kwcoco
    coco_dset = kwcoco.CocoDataset('/data/store/data/shit-pics/data.kwcoco.json')
    """
    import geopandas as gpd
    from shapely import geometry
    import numpy as np
    from pyproj import CRS
    import osmnx as ox
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
    import networkx as nx
    combo = nx.disjoint_union_all(graphs + [graph])

    # https://geopandas.org/en/stable/gallery/plotting_basemap_background.html
    import kwplot
    ax = kwplot.figure().gca()
    fig, ax = ox.plot_graph(combo, bgcolor='lawngreen', node_color='dodgerblue', edge_color='skyblue', ax=ax)

    for x, y in zip(img_locs.geometry.x, img_locs.geometry.y):
        ax.annotate('💩', xy=(x, y), fontname='symbola', color='brown', fontsize=20)

    if 0:
        minx, miny, maxx, maxy = img_locs.unary_union.bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    import contextily as cx
    cx.add_basemap(ax, crs=img_locs.crs)
    # , xytext=(0, 0), textcoords="offset points")
    # img_locs.plot(ax=ax)


def scatterplot(coco_dset):
    """
    import kwcoco
    coco_dset = kwcoco.CocoDataset('/data/store/data/shit-pics/data.kwcoco.json')

    References:
        https://catherineh.github.io/programming/2017/10/24/emoji-data-markers-in-matplotlib

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
    import numpy as np

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

    import kwplot
    sns = kwplot.autosns()

    kwplot.figure(fnum=3, doclf=1)
    sns.histplot(data=img_locs, x='datetime_obj', kde=True, bins=24, stat='count')

    kwplot.figure(fnum=1, doclf=1)
    ax = sns.scatterplot(data=img_locs, x='distance', y='hour_of_day', facecolor=(0, 0, 0, 0), edgecolor=(0, 0, 0, 0))
    # hue='year_month')
    ax.set_xscale('log')
    ax.set_xlabel('Distance from home (meters)')
    ax.set_ylabel('Time of day (hours)')
    ax.set_title('Distribution of Images (n={})'.format(len(img_locs)))

    from PIL import Image, ImageFont, ImageDraw
    text = '💩'
    # text = 'P'
    # sudo apt install ttf-ancient-fonts-symbola
    symbola = ub.grabdata('https://github.com/gearit/ttf-symbola/raw/master/Symbola.ttf')
    font = ImageFont.truetype(symbola, 32, encoding='unic')
    # font = ImageFont.truetype('/usr/share/fonts/truetype/Sarai/Sarai.ttf', 60, encoding='unic')
    # font = ImageFont.truetype("/data/Steam/steamapps/common/Proton 6.3/dist/share/wine/fonts/symbol.ttf", 60, encoding='unic')

    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from kwplot.mpl_make import crop_border_by_color
    label_to_color = ub.dzip(hue_labels, kwplot.Color.distinct(len(hue_labels)))
    label_to_img = {}
    for label, color in label_to_color.items():
        col = kwplot.Color(list(color) + [1]).as255()
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

    idx = img_locs['distance'].argmin()

    # Answer jasons question
    cand = img_locs[img_locs['distance'] < 10]
    pt = cand.iloc[cand['distance'].argmin()]
    name = pt['name']
    name = img_locs.iloc[idx]['name']

    kwplot.phantom_legend(label_to_color)
    # ax.annotate('💩', (x, y))

    import kwimage
    kwplot.imshow(kwimage.stack_images([g.image._A for g in label_to_img.values()]), fnum=2, doclf=True)


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


def doggos():
    import kwimage
    from skimage import exposure  # NOQA
    from skimage.exposure import match_histograms
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
    kwimage.imwrite('dogos.jpg', kwimage.ensure_uint255(canvas))


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


def show_3_images(coco_dset):
    import kwimage
    import kwplot
    kwplot.autompl()

    all_gids = list(coco_dset.images())
    n = 0
    n += 3
    gids = all_gids[943 + n:]
    chosen_gids = gids[0:3]

    images = []
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
    kwplot.imshow(canvas, fnum=1)

    kwimage.imwrite('viz_three_images.jpg', canvas)


def autofind_pair_hueristic(coco_dset):
    import pandas as pd
    import dateutil
    import dateutil.parser
    import ubelt as ub
    import functools
    import kwimage
    import vtool_ibeis
    from vtool_ibeis import PairwiseMatch
    # from vtool_ibeis.matching import VSONE_FEAT_CONFIG

    image_df = pd.DataFrame(coco_dset.dataset['images'])
    ordered_gids = image_df.sort_values('datetime').id.tolist()
    feat_cfg = {
        'rotation_invariance': True,
        'affine_invariance': True,
    }

    # Fails on 31, 32

    @functools.lru_cache(maxsize=32)
    def cache_imread(gid):
        img = coco_dset.imgs[gid1]
        imdata = kwimage.imread(img['file_name'], backend='gdal', overview=2)
        rchip, sf_info = kwimage.imresize(imdata, max_dim=416,
                                          return_info=True)
        return rchip

    @functools.lru_cache(maxsize=32)
    def matchable_image(gid):
        import utool as ut
        img = coco_dset.imgs[gid1]
        dt = dateutil.parser.parse(img['datetime'])
        imdata = kwimage.imread(img['file_name'], backend='gdal', overview=2)
        rchip, sf_info = kwimage.imresize(imdata, max_dim=416,
                                          return_info=True)
        annot = ut.LazyDict({'rchip': rchip, 'dt': dt})
        vtool_ibeis.matching.ensure_metadata_feats(annot, feat_cfg)
        return annot

    scores = {}
    pairs = list(ub.iter_window(ordered_gids, 2))

    for gid1, gid2 in ub.ProgIter(pairs, verbose=3):
        pair = (gid1, gid2)
        if pair in scores:
            continue
        annot1 = matchable_image(gid1)
        annot2 = matchable_image(gid2)
        delta = (annot2['dt'] - annot1['dt'])
        delta_seconds = delta.total_seconds()
        if delta_seconds < 60 * 60:
            match_cfg = {
                'symetric': False,
                'K': 1,
                # 'ratio_thresh': 0.625,
                'ratio_thresh': 0.7,
                'refine_method': 'homog',
                'symmetric': True,
            }
            match = PairwiseMatch(annot1, annot2)
            match.apply_all(cfgdict=match_cfg)
            score = match.fs.sum()
            scores[pair] = (score, delta_seconds)
            print('score = {!r}'.format(score))

    import kwplot
    kwplot.autompl()
    import xdev
    matches = {k: v for k, v in scores.items() if v[0] >= 0}
    iiter = xdev.InteractiveIter(list(matches.items()))
    for pair, compatability in iiter:
        gid1, gid2 = pair
        score, delta = compatability
        imdata1 = cache_imread(gid1)
        imdata2 = cache_imread(gid2)
        canvas = kwimage.stack_images([imdata1, imdata2], axis=1)
        kwplot.imshow(canvas, title='pair={}, score={:0.2f}, delta={}'.format(pair, score, delta))
        print('pair = {!r}'.format(pair))
        xdev.InteractiveIter.draw()


def imread_with_exif(fpath, overview=None):
    import kwimage
    import numpy as np
    exif = extract_exif_metadata(fpath)
    exif_ori = exif.get('Orientation')
    imdata = kwimage.imread(fpath, backend='gdal', overview=overview)
    if exif_ori is not None:
        if exif_ori == 1:
            pass
        elif exif_ori == 6:
            imdata = np.rot90(imdata, k=-1)
        elif exif_ori == 8:
            imdata = np.rot90(imdata, k=1)
        elif exif_ori == 3:
            imdata = np.rot90(imdata, k=2)
        else:
            raise NotImplementedError(exif_ori)
    return imdata


def dump_demo_warp_img(coco_dset):
    import kwplot
    import kwimage
    kwplot.autompl()
    gid1, gid2 = (3, 4)
    gid1, gid2 = (30, 31)
    gid1, gid2 = (34, 35)
    gid1, gid2 = (99, 100)

    fig, fig2 = demo_warp(coco_dset, gid1, gid2)

    ##
    ####
    # Plotting

    # fig2.set_size_inches(25.6 , 13.37)
    fig2.set_size_inches(25.6 / 2, 13.37 / 2)
    fig2.tight_layout()

    fig.set_size_inches(14.4 / 2, 24.84 / 2)
    fig.tight_layout()
    toshow_im1 = kwplot.render_figure_to_image(fig)
    tosave_im1 = kwplot.mpl_make.crop_border_by_color(toshow_im1)
    toshow_im2 = kwplot.render_figure_to_image(fig2)
    tosave_im2 = kwplot.mpl_make.crop_border_by_color(toshow_im2)

    kwimage.imwrite('viz_candidate_ann.png', tosave_im1)
    kwimage.imwrite('viz_align_process.png', tosave_im2)

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


def demo_warp(coco_dset, gid1, gid2):
    """
    import kwcoco
    coco_dset = kwcoco.CocoDataset('/data/store/data/shit-pics/data.kwcoco.json')

    gid1, gid2 = 24, 25
    gid1, gid2 = 339, 340  # paralax

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

    fpath1 = coco_dset.get_image_fpath(gid1)
    fpath2 = coco_dset.get_image_fpath(gid2)
    imdata1 = imread_with_exif(fpath1, overview=2)
    imdata2 = imread_with_exif(fpath2, overview=2)

    maxdim = max(max(imdata1.shape[0:2]), max(imdata2.shape[0:2]))
    maxdim = max(512, maxdim)

    # rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=512, return_info=True)
    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=512, return_info=True)
    # rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=800, return_info=True)
    # rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=800, return_info=True)
    rchip1_rgb, sf_info1 = kwimage.imresize(imdata1, max_dim=maxdim, return_info=True)
    rchip2_rgb, sf_info2 = kwimage.imresize(imdata2, max_dim=maxdim, return_info=True)

    undo_scale = kwimage.Affine.coerce(sf_info1).inv()

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
        'ratio_thresh': 0.625,
        'refine_method': 'homog',
        'rotation_invariance': True,
        'affine_invariance': True,
    }
    match = PairwiseMatch(annot1, annot2)
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
    rchip1_bounds = kwimage.Boxes([[0, 0, rchip1_dsize[0] - 1, rchip1_dsize[1] - 1]], 'xywh').to_polygons()[0]
    rchip2_bounds = kwimage.Boxes([[0, 0, rchip2_dsize[0] - 1, rchip2_dsize[1] - 1]], 'xywh').to_polygons()[0]
    warp_bounds1 = rchip1_bounds.warp(M1)
    shpb1 = warp_bounds1.to_shapely()
    # Only keep valid regions
    warp_bounds1 = kwimage.Polygon.from_shapely(shpb1.intersection(rchip2_bounds.to_shapely()))
    # warp_bounds.draw()
    valid_rchip2_mask1 = warp_bounds1.to_mask(dims=rchip2_dims).data
    rchip2_align = rchip2 * valid_rchip2_mask1[:, :, None]

    if 0:
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
        else:
            raw1 = kwimage.gaussian_blur(rchip1_align, kernel=7)
            raw2 = kwimage.gaussian_blur(rchip2_align, kernel=7)
            rchip1_refine = rchip1_align
            rchip2_refine = rchip2_align
            M = M1.copy()
    # raw1 = raw1.mean(axis=2, keepdims=True)
    # raw2 = raw2.mean(axis=2, keepdims=True)
    diff_img_RAW = np.linalg.norm(np.abs(raw1 - raw2), axis=2)
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

    fig = kwplot.figure(fnum=3)

    pnum_a = kwplot.PlotNums(nRows=3, nCols=1)

    kwplot.imshow(rstack, fnum=3, pnum=pnum_a[0], title='Raw Before / After Image Pair')
    ax = kwplot.figure(fnum=3, pnum=pnum_a[1]).gca()
    ax.set_title('SIFT Features Matches (used to align the images)')
    match.show(ax=ax, show_ell=1, show_lines=False, ell_alpha=0.2, vert=False)

    kwplot.imshow(align_stack1, fnum=3, pnum=pnum_a[2], title='Aligned Images')
    # kwplot.imshow(align_stack2, fnum=3, pnum=pnum_a[3], title='Refined Alignment')

    fig3 = kwplot.figure(fnum=5)
    kwplot.imshow(diff_stack, fnum=5, title='Difference Image -> Binary Mask')
    # kwplot.imshow(attention_rchip1, fnum=3, pnum=pnum_a[4], title='Candidate Annotation Regions')

    fig2 = kwplot.figure(fnum=4)
    kwplot.imshow(attention_imdata1, fnum=4, title='Candidate Annotation Regions')

    fig4 = kwplot.figure(fnum=6)
    kwplot.imshow(align_stack1, fnum=6, title='Aligned')
    return fig, fig2, fig3, fig4


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


# coco_dset.conform()
class Rational(fractions.Fraction):
    """
    Extension of the Fraction class, mostly to make printing nicer

    >>> 3 * -(Rational(3) / 2)
    """
    def __str__(self):
        if self.denominator == 1:
            return str(self.numerator)
        else:
            return '{}'.format(self.numerator / self.denominator)
            # return '({}/{})'.format(self.numerator, self.denominator)

    def __json__(self):
        return {
            'type': 'rational',
            'numerator': self.numerator,
            'denominator': self.denominator,
        }

    def __smalljson__(self):
        return '{:d}/{:d}'.format(self.numerator, self.denominator)

    @classmethod
    def coerce(cls, data):
        from PIL.TiffImagePlugin import IFDRational
        if isinstance(data, dict):
            return cls.from_json(data)
        elif isinstance(data, IFDRational):
            return cls(data.numerator, data.denominator)
        elif isinstance(data, int):
            return cls(data, 1)
        elif isinstance(data, str):
            return cls(*map(int, data.split('/')))
        else:
            raise TypeError

    @classmethod
    def from_json(cls, data):
        return cls(data['numerator'], data['denominator'])

    def __repr__(self):
        return str(self)

    def __neg__(self):
        return Rational(super().__neg__())

    def __add__(self, other):
        return Rational(super().__add__(other))

    def __radd__(self, other):
        return Rational(super().__radd__(other))

    def __sub__(self, other):
        return Rational(super().__sub__(other))

    def __mul__(self, other):
        return Rational(super().__mul__(other))

    def __rmul__(self, other):
        return Rational(super().__rmul__(other))

    def __truediv__(self, other):
        return Rational(super().__truediv__(other))

    def __floordiv__(self, other):
        return Rational(super().__floordiv__(other))


def extract_exif_metadata(fpath):
    from PIL import Image, ExifTags
    from PIL.ExifTags import GPSTAGS
    import ubelt as ub

    img = Image.open(fpath)
    exif = {ExifTags.TAGS[k]: v for k, v in img._getexif().items()
            if k in ExifTags.TAGS}
    if 'GPSInfo' in exif:
        # TODO: get raw rationals?
        exif['GPSInfo'] = ub.map_keys(GPSTAGS, exif['GPSInfo'])
    return exif


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/shitspotter/scripts/gather_shit.py
    """
    main()
