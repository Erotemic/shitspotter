"""
Gather raw data into a kwcoco file
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


def main():
    """
    Used to update the images in the kwcoco file, reconstructs the entire
    thing.

    Walk the coco bundle dpath and reconstruct the kwcoco file with all
    currently available images.

    from shitspotter.gather import *  # NOQA
    """
    import shitspotter
    coco_fpath = shitspotter.util.find_shit_coco_fpath()

    dpath = coco_fpath.parent

    rows = []
    seen = set()
    all_fpaths = []
    change_point = dateutil.parser.parse('2021-05-11T120000')
    walk_prog = ub.ProgIter(desc='walking')
    with walk_prog:
        for r, ds, fs in os.walk(dpath, followlinks=True):
            walk_prog.step()
            dname = ub.Path(r).stem
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
            print('dups = {!r}'.format(dups))
            to_keep.append(keep_gpath)
            to_remove.extend(list(remove_gpaths))

        assert set(to_remove).isdisjoint(set(to_keep))

        # Dont remove, because I'm afraid to do that programtically atm
        # just move to a trash folder
        # pathlib.Path('/data/store/data/shit-pics/_trash_dups')
        trash_dpath = coco_fpath.parent / 'assets/_trash_dups'
        trash_dpath.mkdir(exist_ok=True)
        import shutil
        for p in to_remove:
            p = pathlib.Path(p)
            dst = trash_dpath / p.name
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
        # exif_ori = exif.get('Orientation', None)
        # print('exif_ori = {!r}'.format(exif_ori))
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

    coco_dset.fpath = str(coco_fpath)
    coco_dset._check_json_serializable()
    coco_dset._ensure_json_serializable()
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    coco_dset.reroot(absolute=False)

    #
    coco_dset.dump(coco_dset.fpath, newlines=True)


if __name__ == '__main__':
    """
    CommandLine:
        python -m shitspotter.gather
    """
    main()
