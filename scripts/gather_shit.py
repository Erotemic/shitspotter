import fractions


def main():
    """
    """
    import os
    import dateutil.parser
    from os.path import join
    import xdev
    import pandas as pd
    import kwcoco
    import ubelt as ub
    # import math
    import pathlib
    from dateutil.parser import parse as parse_datetime
    import datetime
    dpath = '/data/store/data/shit-pics/'

    total_items = 0
    num_images = 0

    rows = []
    seen = set()
    duplicates = []
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
            is_triple = not is_double

            for fname in fs:
                if fname.endswith('.mp4'):
                    continue
                if fname in seen:
                    print('SEEN fname = {!r}'.format(fname))
                    duplicates.append(fname)
                    continue
                seen.add(fname)
                gpath = join(r, fname)
                rows.append({
                    'gpath': gpath,
                    'name': pathlib.Path(fname).name,
                    'datestamp': datestamp,
                })

            num_files = len(fs)
            num_images += num_files
            if is_triple:
                num_items = num_files // 3
            else:
                num_items = num_files // 2
            total_items += num_items

    print('num_images = {!r}'.format(num_images))
    print('total_items = {!r}'.format(total_items))

    for row in ub.ProgIter(rows):
        gpath = row['gpath']
        row['nbytes'] = os.stat(gpath).st_size
        row['nbytes_str'] = xdev.byte_str(row['nbytes'])
        exif = extract_image_metadata(gpath)
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
        row['datetime'] = dt.isoformat()
        geos_point = exif.get('GPSInfo', None)
        if geos_point is not None and 'GPSLatitude' in geos_point:
            lat_degrees, lat_minutes, lat_seconds = map(rat_to_frac, geos_point['GPSLatitude'])
            lon_degrees, lon_minutes, lon_seconds = map(rat_to_frac, geos_point['GPSLongitude'])
            lat_sign = {'N': 1, 'S': -1}[geos_point['GPSLatitudeRef']]
            lon_sign = {'E': 1, 'W': -1}[geos_point['GPSLongitudeRef']]
            lat = lat_sign * lat_degrees + lat_minutes / 60 + lat_seconds / 3600
            lon = lon_sign * lon_degrees + lon_minutes / 60 + lon_seconds / 3600
            # Can geojson handle rationals?
            row['geos_point'] = {'type': 'Point', 'coordinates': (lon.__json__(), lat.__json__()), 'properties': {'crs': 'CRS84'}}

    img_info_df = pd.DataFrame(rows)
    img_info_df = img_info_df.sort_values('datetime')
    print(img_info_df)
    print(xdev.byte_str(sum(img_info_df.nbytes)))

    coco_dset = kwcoco.CocoDataset()
    for row in img_info_df.to_dict('records'):
        row = row.copy()
        row['file_name'] = row.pop('gpath')
        row.pop('datestamp', None)
        coco_dset.add_image(**row)

    coco_dset.conform(workers=8)
    coco_dset.validate()

    coco_dset.fpath = str(pathlib.Path(dpath) / 'data.kwcoco.json')
    coco_dset._check_json_serializable()
    coco_dset._ensure_json_serializable()
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    coco_dset.dump(coco_dset.fpath, newlines=True)

    import geopandas as gpd
    from shapely import geometry
    from pyproj import CRS
    import numpy as np

    # image_locs
    rows = []
    for gid, img in coco_dset.index.imgs.items():
        row = {}
        if 'geos_point' in img:
            row['geometry'] = geometry.Point(img['geos_point']['coordinates'])
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

    center_utm_xy = img_utm_xy.mean(axis=0)
    distances = ((img_utm_xy - center_utm_xy) ** 2).sum(axis=1) ** 0.5
    distances = np.array(distances)
    img_locs['distance'] = distances
    datetimes = [parse_datetime(x) for x in img_locs['datetime']]
    img_locs['datetime'] = datetimes
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

    img_locs['only_paired'] = img_locs['datetime'] < change_point

    num_pair_images = img_locs['only_paired'].sum()
    num_triple_images = (~img_locs['only_paired']).sum()
    num_trip_groups = num_triple_images // 3
    pair_groups = num_pair_images // 2
    print('num_trip_groups = {!r}'.format(num_trip_groups))
    print('pair_groups = {!r}'.format(pair_groups))

    import kwplot
    sns = kwplot.autosns()

    kwplot.figure(fnum=3, doclf=1)
    sns.histplot(data=img_locs, x='datetime', kde=True, bins=24, stat='count')

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

    kwplot.phantom_legend(label_to_color)
    # ax.annotate('💩', (x, y))

    import kwimage
    kwplot.imshow(kwimage.stack_images([g.image._A for g in label_to_img.values()]), fnum=2, doclf=True)


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
        imdata = kwimage.imread(img['file_name'])
        rchip, sf_info = kwimage.imresize(imdata, max_dim=416,
                                          return_info=True)
        return rchip

    @functools.lru_cache(maxsize=32)
    def matchable_image(gid):
        import utool as ut
        img = coco_dset.imgs[gid1]
        dt = dateutil.parser.parse(img['datetime'])
        imdata = kwimage.imread(img['file_name'])
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


def demo_warp(coco_dset):
    """
    import kwcoco
    coco_dset = kwcoco.CocoDataset('/data/store/data/shit-pics/data.kwcoco.json')
    """
    import numpy as np

    gid1, gid2 = (3, 4)
    gid1, gid2 = (30, 31)
    gid1, gid2 = (34, 35)

    img1 = coco_dset.coco_image(gid1)
    img2 = coco_dset.coco_image(gid2)

    imdata1 = img1.delay().finalize()
    imdata2 = img2.delay().finalize()

    # imdata1 = np.rot90(imdata1)
    # imdata2 = np.rot90(imdata2)

    import kwimage
    rchip1, sf_info1 = kwimage.imresize(imdata1, max_dim=416, return_info=True)
    rchip2, sf_info2 = kwimage.imresize(imdata2, max_dim=416, return_info=True)

    undo_scale = kwimage.Affine.coerce(sf_info1).inv()

    # import kwplot
    # kwplot.autompl()
    # kwplot.imshow(imdata1, fnum=1, pnum=(1, 2, 1))
    # kwplot.imshow(imdata2, fnum=1, pnum=(1, 2, 2))

    from vtool_ibeis import PairwiseMatch
    # from vtool_ibeis.inspect_matches import MatchInspector

    annot1 = {'rchip': rchip1}
    annot2 = {'rchip': rchip2}

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

    rchip1 = kwimage.ensure_float01(match.annot1['rchip'])
    rchip2 = kwimage.ensure_float01(match.annot2['rchip'])

    import cv2
    dims = rchip2.shape[0:2]
    dsize = dims[::-1]
    M = match.H_12

    rchip1_align = cv2.warpPerspective(rchip1, M, dsize)

    rchip1_bounds = kwimage.Boxes([[0, 0, dsize[0], dsize[1]]], 'xywh').to_polygons()[0]
    warp_bounds = rchip1_bounds.warp(M)
    # warp_bounds.draw()
    valid_mask = warp_bounds.to_mask(dims=dims).data
    rchip2_align = rchip2 * valid_mask[:, :, None]

    diff_img = np.abs(rchip1_align - rchip2_align)
    diff_img = np.linalg.norm(diff_img, axis=2)
    mask = kwimage.gaussian_blur(diff_img, sigma=3.0)
    mask = kwimage.morphology(mask, 'close')
    mask = kwimage.morphology((mask > 0.4).astype(np.float32), 'dilate', kernel=10)
    mask = kwimage.gaussian_blur(mask, sigma=3.0)
    mask = (mask > 0.2).astype(np.float32)
    mask = kwimage.morphology(mask, 'close', kernel=30)
    mask = kwimage.morphology((mask).astype(np.float32), 'dilate', kernel=30)

    # Warp mask back onto original image
    tf_orig_from_align = np.asarray(undo_scale @ np.linalg.inv(M))
    orig_dsize = imdata1.shape[0:2][::-1]
    mask1_orig = cv2.warpPerspective(mask, tf_orig_from_align, orig_dsize)
    attention_imdata1 = imdata1 * mask1_orig[..., None]

    import kwplot
    kwplot.autompl()
    rstack = kwimage.stack_images([rchip1, rchip2], pad=10, axis=1)
    align_stack = kwimage.stack_images([rchip1_align, rchip2_align], pad=10, axis=1)
    diff_stack = kwimage.stack_images([diff_img, mask], pad=10, axis=1)

    fig = kwplot.figure(fnum=3)

    pnum_a = kwplot.PlotNums(nRows=4, nCols=1)

    kwplot.imshow(rstack, fnum=3, pnum=pnum_a[0], title='Raw Before / After Image Pair')

    ax = kwplot.figure(fnum=3, pnum=pnum_a[1]).gca()
    ax.set_title('SIFT Features Matches (used to align the images)')
    match.show(ax=ax, show_ell=1, show_lines=False, ell_alpha=0.2, vert=False)

    kwplot.imshow(align_stack, fnum=3, pnum=pnum_a[2], title='Aligned Images')

    kwplot.imshow(diff_stack, fnum=3, pnum=pnum_a[3], title='Difference Image -> Binary Mask')

    # kwplot.imshow(attention_rchip1, fnum=3, pnum=pnum_a[4], title='Candidate Annotation Regions')

    fig2 = kwplot.figure(fnum=4)
    kwplot.imshow(attention_imdata1, fnum=4, title='Candidate Annotation Regions')
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


def rat_to_frac(rat):
    return Rational(rat.numerator, rat.denominator)


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


def extract_image_metadata(fpath):
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
