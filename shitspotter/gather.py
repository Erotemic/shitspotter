#!/usr/bin/env python
"""
Gather raw data into a kwcoco file.

This also handles anonymization if secrets are mounted.

Usage:
    python -m shitspotter.gather
"""
# import math
from dateutil.parser import parse as parse_datetime
from os.path import join
import datetime
import dateutil.parser
import kwcoco
import os
import pandas as pd
import pathlib
import ubelt as ub
import xdev
from shitspotter.util import Rational, extract_exif_metadata


def gather_test_rows(dpath):
    """
    The data in the test set.
    """
    contrip_dpath = (dpath / 'assets/_contributions')
    image_rows = []
    seen = set()
    all_fpaths = []
    walk_prog = ub.ProgIter(desc='walking')
    extensions = set()
    block_extensions = ('.mp4', '.json', '.pkl')
    with walk_prog:
        for r, ds, fs in os.walk(contrip_dpath, followlinks=True):
            walk_prog.step()
            dname = ub.Path(r).stem
            if dname.startswith('racoon-poop'):
                continue

            for fname in fs:
                gpath = join(r, fname)
                all_fpaths.append(gpath)
                if fname.endswith(block_extensions):
                    continue
                if fname in seen:
                    print('SEEN fname = {!r}'.format(fname))
                    continue

                ext = fname.split('.')[-1]

                if ext == 'shitspotter':
                    raise Exception

                labelme_sidecar_fpath = ub.Path(gpath).augment(ext='.json')
                has_labelme = labelme_sidecar_fpath.exists()

                extensions.add(ext)
                seen.add(fname)
                image_info = {
                    'gpath': gpath,
                    'name': pathlib.Path(fname).stem,
                    'cohort': dname,
                    'has_labelme': has_labelme,
                }
                image_rows.append(image_info)
    # Only take test images with annotations
    image_rows = [r for r in image_rows if r['has_labelme']]
    return image_rows


def gather_learn_rows(dpath):
    """
    The data we are allowed to learn from.

    Ignore:
        import shitspotter
        dpath = shitspotter.util.find_shit_coco_fpath().parent
    """
    image_rows = []
    seen = set()
    all_fpaths = []
    protocol_change_point = dateutil.parser.parse('2021-05-11T120000')
    walk_prog = ub.ProgIter(desc='walking')
    extensions = set()
    block_extensions = ('.mp4', '.json', '.pkl')
    with walk_prog:
        for r, ds, fs in os.walk(dpath, followlinks=True):
            walk_prog.step()
            dname = ub.Path(r).stem
            if dname.startswith('poop-'):
                timestr = dname.split('poop-')[1]
                datestamp = dateutil.parser.parse(timestr)

                is_double = datestamp < protocol_change_point

                for fname in fs:
                    gpath = join(r, fname)
                    all_fpaths.append(gpath)
                    if fname.endswith(block_extensions):
                        continue
                    if fname in seen:
                        print('SEEN fname = {!r}'.format(fname))
                        continue

                    ext = fname.split('.')[-1]

                    if ext == 'shitspotter':
                        raise Exception

                    labelme_sidecar_fpath = ub.Path(gpath).augment(ext='.json')
                    has_labelme = labelme_sidecar_fpath.exists()

                    extensions.add(ext)
                    seen.add(fname)
                    image_info = {
                        'gpath': gpath,
                        'name': pathlib.Path(fname).stem,
                        'cohort': dname,
                        'datestamp': datestamp,
                        'is_double': is_double,
                        'has_labelme': has_labelme,
                    }
                    image_rows.append(image_info)
    cohort_to_num_labels = {}
    total_numer = 0
    total_denom = 0
    for cohort, group in sorted(ub.group_items(image_rows, key=lambda x: x['cohort']).items()):
        num_labels = sum([g['has_labelme'] for g in group])
        group_size = len(group)
        numer = (num_labels * 3)
        complete_frac = numer / group_size
        if complete_frac > 1.2:
            numer = (num_labels * 2)
        complete_frac =  numer / group_size
        total_numer += numer
        total_denom += group_size
        complete_percent = min(1, complete_frac) * 100
        cohort_to_num_labels[cohort] = f'{num_labels} / {len(group)} - ~{complete_percent:0.2f}%'
    import rich
    rich.print('cohort_to_num_labels = {}'.format(ub.urepr(cohort_to_num_labels, nl=1, align=' - ')))
    total_complete = (total_numer / total_denom) * 100
    print(f'total_complete = {total_complete:0.2f}%')

    dupidxs = ub.find_duplicates(all_fpaths, key=lambda x: pathlib.Path(x).name)
    # assert len(dupidxs) == 0

    if len(dupidxs) > 0:
        print('ERROR: DUPLICATE DATA')
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
                delta = folder_dt - image_dt
                folder_dts.append(folder_dt)
                image_dts.append(image_dt)
                assert delta.total_seconds() >= 0
                deltas[d] = delta
            # if any(d.total_seconds() < 0 for d in deltas.values()):
            #     break
            assert len(set(image_dts)) == 1
            keep_gpath = ub.argmin(deltas)
            remove_gpaths = set(dups) - {keep_gpath}
            image_dt = image_dts[0]
            print('image_dt = {!r}'.format(image_dt))
            print('folder_dts = {}'.format(ub.repr2(folder_dts, nl=1)))
            print('deltas = {!r}'.format(deltas))
            print(f'dups = {ub.urepr(dups, nl=1)}')
            to_keep.append(keep_gpath)
            to_remove.extend(list(remove_gpaths))

        raise Exception('Duplicate data')
        assert set(to_remove).isdisjoint(set(to_keep))

        # Dont remove, because I'm afraid to do that programtically atm
        # just move to a trash folder
        # pathlib.Path('/data/store/data/shit-pics/_trash_dups')
        trash_dpath = dpath / 'assets/_trash_dups'
        trash_dpath.mkdir(exist_ok=True)
        import shutil
        for p in to_remove:
            p = pathlib.Path(p)
            dst = trash_dpath / p.name
            shutil.move(p, dst)
    return image_rows


def extract_exif_gps(exif):
    geos_point = exif.get('GPSInfo', None)
    if geos_point is not None and 'GPSLatitude' in geos_point:
        lat_degrees, lat_minutes, lat_seconds = map(Rational.coerce, geos_point['GPSLatitude'])
        lon_degrees, lon_minutes, lon_seconds = map(Rational.coerce, geos_point['GPSLongitude'])
        lat_sign = {'N': 1, 'S': -1}[geos_point['GPSLatitudeRef']]
        lon_sign = {'E': 1, 'W': -1}[geos_point['GPSLongitudeRef']]
        lat = lat_sign * (lat_degrees + lat_minutes / 60 + lat_seconds / 3600)
        lon = lon_sign * (lon_degrees + lon_minutes / 60 + lon_seconds / 3600)
        # Can geojson handle rationals?
        geos_point = {'type': 'Point', 'coordinates': (lon.__smalljson__(), lat.__smalljson__()), 'properties': {'crs': 'CRS84'}}
    else:
        geos_point = None
    return geos_point


def process_image_rows(image_rows, coco_dset=None):
    """
    TODO: we only need to process new images.  This does it each time, and that
    is kinda slow.
    """
    if coco_dset is None:
        coco_dset = kwcoco.CocoDataset()

    new_image_rows = []
    bundle_dpath = ub.Path(coco_dset.bundle_dpath)
    for row in ub.ProgIter(image_rows):
        gpath = row['gpath']

        file_name = ub.Path(gpath).relative_to(bundle_dpath)
        if os.fspath(file_name) in coco_dset.index.file_name_to_img:
            # Skip existing images
            continue

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
        except KeyError:
            ...
        except Exception:
            # dt = date_parser.parse(exif['DateTime'])
            # row['datetime'] = dt.isoformat()
            raise
        else:
            # TODO: exif 'OffsetTime': '-05:00',
            row['datetime'] = dt.isoformat()

        exif_ori = exif.get('Orientation', None)
        try:
            exif_ori = int(exif_ori)  # defaults to a float for whatever reason
        except Exception:
            ...
        row['exif_ori'] = exif_ori

        # print('exif_ori = {!r}'.format(exif_ori))
        geos_point = extract_exif_gps(exif)
        if geos_point is not None:
            row['geos_point'] = geos_point
        new_image_rows.append(row)

    img_info_df = pd.DataFrame(new_image_rows)

    # HANDLE_PRIVACY = True
    # if HANDLE_PRIVACY:
    #     """
    #     sudo apt-get install xdelta3
    #     """
    #     # Strip out privacy relevant data. Depending on the privacy policy,
    #     # either discard it entirely or save it into a secure encrypted form.
    #     # (in the latter case this allows the metadata to be released later
    #     # after it is no longer sensitive)
    #     from shitspotter.util.util_data import find_secret_dpath
    #     from shitspotter.util.util_data import is_probably_encrypted
    #     secret_dpath = find_secret_dpath()
    #     privacy_rules_fpath = secret_dpath / 'privacy_rules.py'
    #     if is_probably_encrypted(privacy_rules_fpath):
    #         raise EnvironmentError('The privacy rules file is still encrypted')
    #     privacy_rules = ub.import_module_from_path(privacy_rules_fpath)
    #     img_info_df = privacy_rules.apply_privacy_rules(img_info_df)
    # else:
    #     raise NotImplementedError(
    #         'todo: without privacy rules, we just copy from staging to the repo')
    #     ...

    print(img_info_df)
    print(xdev.byte_str(sum(img_info_df.nbytes)))

    for row in img_info_df.to_dict('records'):
        row = row.copy()
        row.pop('nbytes_str', None)
        row.pop('is_double', None)
        row['file_name'] = row.pop('gpath')
        row.pop('datestamp', None)
        coco_dset.add_image(**row)

    # Hacks so geowatch handles the dataset nicely
    # In the future geowatch should be more robust
    for img in coco_dset.dataset['images']:
        img['sensor_coarse'] = 'phone'
        img['datetime_captured'] = img['datetime']
        img['channels'] = 'red|green|blue'

    coco_dset.conform(workers=8)
    coco_dset.validate()
    coco_dset._check_json_serializable()
    coco_dset._ensure_json_serializable()
    return coco_dset


def _new_generic_gather_image_rows(dpath):
    """
    experimental consolidation of logic.

    TODO: consolidate with gather_learn_rows and gather_test_rows

    Ignore:
        from shitspotter.gather import *  # NOQA
        from shitspotter.gather import _new_generic_gather_image_rows
        dpath = '/data/joncrall/dvc-repos/shitspotter_dvc/assets/poop-2024-11-22-T195205'
        image_rows = _new_generic_gather_image_rows(dpath)
        image_gdf = _new_generic_image_gdf_expand(image_rows)
    """
    image_rows = []
    seen = set()
    all_fpaths = []
    walk_prog = ub.ProgIter(desc='walking')
    block_extensions = ('.mp4', '.json', '.pkl')
    with walk_prog:
        for r, ds, fs in os.walk(dpath, followlinks=True):
            walk_prog.step()
            dname = ub.Path(r).stem
            for fname in fs:
                gpath = join(r, fname)
                all_fpaths.append(gpath)
                if fname.endswith(block_extensions):
                    continue
                if fname in seen:
                    # print('SEEN fname = {!r}'.format(fname))
                    continue

                ext = fname.split('.')[-1]

                if ext == 'shitspotter':
                    raise Exception

                seen.add(fname)
                image_info = {
                    'gpath': gpath,
                    'name': pathlib.Path(fname).stem,
                    'cohort': dname,
                }
                image_rows.append(image_info)
    return image_rows


def _new_generic_image_gdf_expand(image_rows):
    """
    Todo: consolidate with process_image_rows
    """
    import geopandas as gpd
    from kwgis.utils import util_gis
    import kwutil
    new_image_rows = []
    for row in ub.ProgIter(image_rows):
        gpath = ub.Path(row['gpath'])

        labelme_sidecar_fpath = ub.Path(gpath).augment(ext='.json')
        has_labelme = labelme_sidecar_fpath.exists()
        row['has_labelme'] = has_labelme

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
        except KeyError:
            ...
        except Exception:
            # dt = date_parser.parse(exif['DateTime'])
            # row['datetime'] = dt.isoformat()
            raise
        else:
            # TODO: exif 'OffsetTime': '-05:00',
            row['datetime'] = dt.isoformat()

        exif_ori = exif.get('Orientation', None)
        try:
            exif_ori = int(exif_ori)  # defaults to a float for whatever reason
        except Exception:
            ...
        row['exif_ori'] = exif_ori

        # print('exif_ori = {!r}'.format(exif_ori))
        geos_point = extract_exif_gps(exif)
        if geos_point is not None:
            row['geos_point'] = geos_point
        new_image_rows.append(row)

    img_info_df = pd.DataFrame(new_image_rows)

    geos_points = gpd.GeoSeries([
        geos_to_shapely_point(p) for p in img_info_df['geos_point']
    ])
    image_gdf = gpd.GeoDataFrame(img_info_df, geometry=geos_points, crs=util_gis.get_crs84())
    image_gdf['time'] = image_gdf['datetime'].apply(kwutil.datetime.coerce)

    image_gdf = image_gdf.sort_values('datetime')

    # CHECK:
    if 0:
        for _, row in image_gdf.iterrows():
            gpath = row['gpath']
            exif = extract_exif_metadata(gpath)
            geos_point = extract_exif_gps(exif)
            point1 = geos_to_shapely_point(geos_point)
            point2 = row['geometry']
            if point1 != point2:
                print(geos_point)
                print(row['geos_point'])
                print('ERROR')
                print(point1.xy)
                print(point2.xy)
                raise Exception
            else:
                print('OK')

    return image_gdf


def geos_to_shapely_point(geos_point):
    from shapely.geometry import Point
    from shitspotter.util import Rational
    if pd.isna(geos_point):
        return None
    else:
        point = Point(Rational.coerce(geos_point['coordinates'][0]),
                      Rational.coerce(geos_point['coordinates'][1]))
    return point


def read_ann_labelme_anns_into_coco(coco_dset):
    bundle_dpath = ub.Path(coco_dset.bundle_dpath)
    for img in coco_dset.dataset['images']:
        if img['has_labelme']:
            for ann in load_labelme_anns(bundle_dpath, img):
                catname = ann.pop('category_name')
                cid = coco_dset.ensure_category(catname)
                ann['category_id'] = cid
                coco_dset.add_annotation(**ann)

            # if 0:
            #     import kwplot
            #     kwplot.autompl(recheck=1, force='QtAgg')
            #     if not inv_exif.isclose_identity():
            #         coco_dset.show_image(img['id'])
            #         if img['id'] not in {0, 1575, 7, 1554}:
            #             raise Exception(img['id'])


def load_labelme_anns(bundle_dpath, img):
    import kwimage
    gpath = bundle_dpath / img['file_name']
    labelme_fpath = gpath.augment(ext='.json')
    assert labelme_fpath.exists()

    labelme_data = read_labelme_data(labelme_fpath)

    # labelme_data = json.loads(fpath.read_text())
    from kwcoco.formats.labelme import labelme_to_coco_structure
    imginfo, annsinfo = labelme_to_coco_structure(labelme_data, special_options=True)
    # image_name = imginfo['file_name'].rsplit('.', 1)[0]

    # Construct the inverted exif transform
    # (From exif space -> raw space)
    rot_ccw = 0
    flip_axis = None
    if img['exif_ori'] == 8:
        rot_ccw = 3
    elif img['exif_ori'] == 3:
        rot_ccw = 2
    elif img['exif_ori'] == 6:
        rot_ccw = 1
    elif img['exif_ori'] == 7:
        flip_axis = 1
        rot_ccw = 3
    elif img['exif_ori'] == 4:
        flip_axis = 1
        rot_ccw = 2
    elif img['exif_ori'] == 5:
        flip_axis = 1
        rot_ccw = 1
    exif_canvas_dsize = (labelme_data['imageWidth'], labelme_data['imageHeight'])
    inv_exif = kwimage.Affine.fliprot(
        flip_axis=flip_axis, rot_k=rot_ccw,
        canvas_dsize=exif_canvas_dsize
    )

    for ann in annsinfo:
        ann = ann.copy()
        poly = kwimage.Polygon.from_coco(ann['segmentation'])

        if not inv_exif.isclose_identity():
            # if img['id'] not in {0}:
            #     raise Exception(img['id'])
            # LabelMe Polygons are annotated in EXIF space, but
            # we need them in raw space for kwcoco.
            poly = poly.warp(inv_exif)

        ann['segmentation'] = poly.to_coco(style='new')
        ann['bbox'] = poly.box().quantize().to_coco()

        ann['image_id'] = img['id']
        yield ann


def read_labelme_data(labelme_fpath):
    import json
    # Fixup labelme json files
    # Remove image data, fix bad labels
    labelme_data = json.loads(labelme_fpath.read_text())
    needs_write = 0
    if labelme_data.get('imageData', None) is not None:
        labelme_data['imageData'] = None
        needs_write = 1

    for shape in labelme_data['shapes']:
        if shape['label'] == 'poop;':
            shape['label'] = 'poop'
            needs_write = 1

    if needs_write:
        labelme_fpath.write_text(json.dumps(labelme_data))
    return labelme_data


def main():
    """
    Used to update the images in the kwcoco file, reconstructs the entire
    thing.

    Walk the coco bundle dpath and reconstruct the kwcoco file with all
    currently available images.

    Ignore:
        from shitspotter.gather import *  # NOQA
    """
    import shitspotter
    learn_coco_fpath = shitspotter.util.find_shit_coco_fpath()
    test_coco_fpath = learn_coco_fpath.augment(stem='test', multidot=True, ext='.kwcoco.zip')
    dpath = learn_coco_fpath.parent

    if 0 and learn_coco_fpath.exists():
        learn_coco_dset = kwcoco.CocoDataset(learn_coco_fpath)
    else:
        learn_coco_dset = kwcoco.CocoDataset()
        learn_coco_dset.fpath = learn_coco_fpath

    if 1 and test_coco_fpath.exists():
        test_coco_dset = kwcoco.CocoDataset(test_coco_fpath)
    else:
        test_coco_dset = kwcoco.CocoDataset()
        test_coco_dset.fpath = test_coco_fpath

    if 0:
        coco_dset = learn_coco_dset  # NOQA
        image_rows = learn_image_rows  # NOQA

    learn_image_rows = gather_learn_rows(dpath)
    learn_coco_dset = process_image_rows(learn_image_rows, coco_dset=learn_coco_dset)
    learn_coco_dset.fpath = str(learn_coco_fpath)

    test_image_rows = gather_test_rows(dpath)
    test_coco_dset = process_image_rows(test_image_rows, coco_dset=test_coco_dset)
    test_coco_dset.fpath = str(test_coco_fpath)

    print(f'test_coco_dset.fpath={test_coco_dset.fpath}')
    print(f'learn_coco_dset.fpath={learn_coco_dset.fpath}')
    learn_coco_dset.reroot(absolute=False)
    test_coco_dset.reroot(absolute=False)
    # coco_dset.clear_annotations()

    ADD_LABELME_ANNOTS = 1
    if ADD_LABELME_ANNOTS:
        read_ann_labelme_anns_into_coco(learn_coco_dset)
        read_ann_labelme_anns_into_coco(test_coco_dset)

    # learn_code = build_code(learn_coco_dset)
    test_code = build_code(test_coco_dset)
    test_coco_dset.fpath = os.fspath(dpath / ('test_' + test_code + '.kwcoco.zip'))

    import rich
    for dset in [learn_coco_dset, test_coco_dset]:
        rich.print('dset = {}'.format(ub.urepr(dset, nl=1)))
        rich.print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))
        rich.print(ub.urepr(dset.stats(extended=True), nl=2))
        dset.dump(dset.fpath, newlines=True)

    print('Wrote:')
    for dset in [learn_coco_dset, test_coco_dset]:
        rich.print('dset.fpath = {}'.format(ub.urepr(dset.fpath, nl=1)))


def build_code(coco_dset):
    hashid = coco_dset._build_hashid()[0:8]
    return f'imgs{coco_dset.n_images}_{hashid}'


if __name__ == '__main__':
    """
    CommandLine:
        python -m shitspotter.gather
    """
    main()
