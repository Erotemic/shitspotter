import fractions


def main():
    """
    """
    import os
    import dateutil.parser
    from os.path import join
    dpath = '/data/store/data/shit-pics/'

    total_items = 0
    num_images = 0

    gpath_list = []

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
            timestamp = dateutil.parser.parse(timestr)

            is_double = timestamp < change_point
            is_triple = not is_double

            for fname in fs:
                if fname.endswith('.mp4'):
                    continue
                gpath = join(r, fname)
                gpath_list.append(gpath)

            num_files = len(fs)
            num_images += num_files
            if is_triple:
                num_items = num_files // 3
            else:
                num_items = num_files // 2
            print('num_items = {!r}'.format(num_items))
            total_items += num_items

    print('num_images = {!r}'.format(num_images))
    print('total_items = {!r}'.format(total_items))

    import kwcoco
    import ubelt as ub
    coco_dset = kwcoco.CocoDataset.from_image_paths(gpath_list)

    for gid in ub.ProgIter(list(coco_dset.images())):
        coco_img = coco_dset.coco_image(gid)
        extract_metadata(coco_img)

    # from dateutil import parser as date_parser
    from dateutil.parser import parse as parse_datetime
    import datetime
    for gid in ub.ProgIter(list(coco_dset.images())):
        coco_img = coco_dset.coco_image(gid)
        img = coco_img.img
        exif = img['exif']

        try:
            # parse_datetime(exif['DateTime'])
            exif_datetime = exif['DateTime']
            dt = datetime.datetime.strptime(exif_datetime, '%Y:%m:%d %H:%M:%S')
        except Exception:
            # dt = date_parser.parse(exif['DateTime'])
            # img['date_captured'] = dt.isoformat()
            raise
        print(exif_datetime)
        print(dt.isoformat())
        img['date_captured'] = dt.isoformat()

        geos_point = exif.get('GPSInfo', None)
        if geos_point is not None and 'GPSLatitude' in geos_point:
            def rat_to_frac(rat):
                return Rational(rat.numerator, rat.denominator)
            lat_degrees, lat_minutes, lat_seconds = map(rat_to_frac, geos_point['GPSLatitude'])
            lon_degrees, lon_minutes, lon_seconds = map(rat_to_frac, geos_point['GPSLongitude'])
            lat_sign = {'N': 1, 'S': -1}[geos_point['GPSLatitudeRef']]
            lon_sign = {'E': 1, 'W': -1}[geos_point['GPSLongitudeRef']]
            lat = lat_sign * lat_degrees + lat_minutes / 60 + lat_seconds / 3600
            lon = lon_sign * lon_degrees + lon_minutes / 60 + lon_seconds / 3600
            # Can geojson handle rationals?
            img['geos_point'] = {'type': 'Point', 'coordinates': (lon, lat), 'properties': {'crs': 'CRS84'}}

    import geopandas as gpd
    from shapely import geometry

    # image_locs
    rows = []
    for gid, img in coco_dset.index.imgs.items():
        row = {}
        row['date_captured'] = img['date_captured']
        if 'geos_point' in img:
            row['geometry'] = geometry.Point(img['geos_point']['coordinates'])
            rows.append(row)
    img_locs = gpd.GeoDataFrame(rows, crs='crs84')

    # ip_pt = ip_loc.iloc[0].geometry
    # from pyproj import CRS
    utm_crs = CRS.from_epsg(utm_epsg_from_latlon(img_locs.iloc[0].geometry.y, img_locs.iloc[0].geometry.x))  # NOQA
    img_utm_loc = img_locs.to_crs(utm_crs)

    import numpy as np
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
    datetimes = [parse_datetime(x) for x in img_locs['date_captured']]
    img_locs['datetime'] = datetimes
    img_locs['timestamp'] = [x.timestamp() for x in datetimes]
    img_locs['date'] = [x.date() for x in datetimes]
    img_locs['time'] = [x.time() for x in datetimes]
    img_locs['year_month'] = [x.strftime('%Y-%m') for x in datetimes]
    img_locs['hour_of_day'] = (img_locs['timestamp'] / (60 * 60)) % (24)
    hue_labels = sorted(img_locs['year_month'].unique())
    # dt = img_locs['datetime'].iloc[0]
    # date = dt.date()
    # time = dt.time()

    import kwplot
    sns = kwplot.autosns()
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


def extract_metadata(coco_img):
    import pathlib
    bundle_dpath = pathlib.Path(coco_img.dset.bundle_dpath)
    for obj in coco_img.iter_asset_objs():
        fpath = bundle_dpath / obj['file_name']
        exif = extract_image_metadata(fpath)
        obj['exif'] = exif


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
