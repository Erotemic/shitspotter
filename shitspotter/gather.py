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
    from shitspotter.gather import *  # NOQA
    """
    import shitspotter
    coco_fpath = shitspotter.util.find_shit_coco_fpath()

    dpath = coco_fpath.parent

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
        # pathlib.Path('/data/store/data/shit-pics/_trash_dups')
        dpath = coco_fpath.parent / 'assets/_trash_dups'
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

    coco_dset.fpath = str(pathlib.Path(dpath) / 'data.kwcoco.json')
    coco_dset._check_json_serializable()
    coco_dset._ensure_json_serializable()
    print('coco_dset.fpath = {!r}'.format(coco_dset.fpath))
    coco_dset.reroot(absolute=False)

    #
    coco_dset.dump(coco_dset.fpath, newlines=True)


def autofind_pair_hueristic(coco_dset):
    """
    import shitspotter
    shitspotter.util.find_shit_coco_fpath()
    import kwcoco
    coco_dset = kwcoco.CocoDataset('/data/store/data/shit-pics/data.kwcoco.json')
    """
    import pandas as pd
    import dateutil
    import dateutil.parser
    import ubelt as ub
    import functools
    import kwimage
    import vtool_ibeis
    import numpy as np
    from vtool_ibeis import PairwiseMatch
    # from vtool_ibeis.matching import VSONE_FEAT_CONFIG

    image_df = pd.DataFrame(coco_dset.dataset['images'])
    ordered_gids = image_df.sort_values('datetime').id.tolist()
    feat_cfg = {
        'rotation_invariance': True,
        'affine_invariance': True,
    }

    # Fails on 31, 32

    # Save a table of images matches in the dataset
    coco_dset.dataset.setdefault('image_matches', [])

    @functools.lru_cache(maxsize=32)
    def cache_imread(gid):
        coco_img = coco_dset.coco_image(gid)
        fpath = coco_img.primary_image_filepath()
        imdata = kwimage.imread(fpath, backend='gdal', overview=0)
        rchip, sf_info = kwimage.imresize(imdata, max_dim=1024,
                                          return_info=True, antialias=True)
        return rchip

    @functools.lru_cache(maxsize=32)
    def matchable_image(gid):
        import utool as ut
        coco_img = coco_dset.coco_image(gid)
        fpath = coco_img.primary_image_filepath()
        img = coco_img.img
        dt = dateutil.parser.parse(img['datetime'])
        imdata = kwimage.imread(fpath, backend='gdal', overview=0)
        rchip, sf_info = kwimage.imresize(imdata, max_dim=1024,
                                          return_info=True, antialias=True)
        annot = ut.LazyDict({'rchip': rchip, 'dt': dt})
        vtool_ibeis.matching.ensure_metadata_feats(annot, feat_cfg)
        return annot

    image_matches = {}
    for match in coco_dset.dataset['image_matches']:
        name1 = match['name1']
        name2 = match['name2']
        image_matches[(name1, name2)] = match

    pairs = list(ub.iter_window(ordered_gids, 2))
    for gid1, gid2 in ub.ProgIter(pairs, verbose=3):
        name1 = coco_dset.index.imgs[gid1]['name']
        name2 = coco_dset.index.imgs[gid2]['name']
        pair = (name1, name2)
        if pair  in image_matches:
            continue
        annot1 = matchable_image(gid1)
        annot2 = matchable_image(gid2)
        delta = (annot2['dt'] - annot1['dt'])
        delta_seconds = delta.total_seconds()
        if delta_seconds < 60 * 60:
            match_cfg = {
                'symetric': True,
                'K': 1,
                'ratio_thresh': 0.625,
                # 'ratio_thresh': 0.7,
                'refine_method': 'homog',
                'symmetric': True,
            }
            match = PairwiseMatch(annot1, annot2)
            match.apply_all(cfgdict=match_cfg)
            score = match.fs.sum()
            match = {
                'name1': name1,
                'name2': name2,
                'score': score,
                'H_12': match.H_12,
            }

            from kwcoco.util import util_json
            match = util_json.ensure_json_serializable(match)
            print('match = {}'.format(ub.repr2(match, nl=1)))
            image_matches[pair] = match
            coco_dset.dataset['image_matches'].append(match)

    # Save the match table
    coco_dset.dump(coco_dset.fpath, newlines=True)

    score_dict = ub.ddict(dict)
    for match in image_matches.values():
        name1 = match['name1']
        name2 = match['name2']
        if match['score'] > 400:
            score_dict[(name1, name2)] = match['score']
    assignment = maxvalue_assignment2(score_dict)

    ordered_names = coco_dset.images(ordered_gids).lookup('name')
    ub.find_duplicates(ub.flatten(assignment))
    node_to_pairid = {a: i for i, pair in enumerate(assignment) for a in pair}
    node_to_pair = {a: pair for pair in assignment for a in pair}
    rows = []
    for name in ordered_names:
        dt = coco_dset.index.name_to_img[name]['datetime']
        pair = node_to_pair.get(name, None)
        score = score_dict.get(pair, None)
        if score is None and pair is not None:
            score = score_dict.get(pair[::-1], None)
        rows.append({
            'name': name,
            'pair_id': node_to_pairid.get(name, None),
            'score': score,
            'datetime': dt,
        })
    df = pd.DataFrame(rows)
    print(df.to_string())

    # Check on the automatic protocol
    change_point = dateutil.parser.parse('2021-05-11T120000')

    pairwise_df = df[
        pd.to_datetime(df['datetime']) <= change_point
    ]

    triple_df = df[
        pd.to_datetime(df['datetime']) > change_point
    ]
    triple_df = triple_df.assign(in_sequence=([False] * len(triple_df)))
    pairwise_df = pairwise_df.assign(in_sequence=([False] * len(pairwise_df)))

    good_pairs = 0
    bad_pairwise_items = 0

    idx = 0
    good_pairwise_idxs = []
    while idx < len(pairwise_df) - 1:
        a = pairwise_df.iloc[idx]
        b = pairwise_df.iloc[idx + 1]
        if a.pair_id == b.pair_id:
            good_pairs += 1
            good_pairwise_idxs.append(idx)
            good_pairwise_idxs.append(idx + 1)
            idx += 2
        else:
            bad_pairwise_items += 1
            idx += 1

    for _ in range(idx, len(pairwise_df)):
        bad_pairwise_items += 1

    v = pairwise_df['in_sequence'].values
    v[good_pairwise_idxs] = 1
    pairwise_df['in_sequence'] = v

    good_triples = 0
    bad_triple_items = 0
    good_triple_idxs = []
    idx = 0
    while idx < len(triple_df) - 2:
        a = triple_df.iloc[idx]
        b = triple_df.iloc[idx + 1]
        c = triple_df.iloc[idx + 2]
        if a.pair_id == b.pair_id and np.isnan(c.pair_id):
            good_triple_idxs.append(idx)
            good_triple_idxs.append(idx + 1)
            good_triple_idxs.append(idx + 2)
            good_triples += 1
            idx += 3
        else:
            bad_triple_items += 1
            idx += 1
    for _ in range(idx, len(pairwise_df)):
        bad_pairwise_items += 1

    v = triple_df['in_sequence'].values
    v[good_triple_idxs] = 1
    triple_df['in_sequence'] = v

    print(pairwise_df.to_string())
    print(triple_df.to_string())

    print('good_pairs = {!r}'.format(good_pairs))
    print('good_triples = {!r}'.format(good_triples))
    print('bad_pairwise_items = {!r}'.format(bad_pairwise_items))
    print('bad_triple_items = {!r}'.format(bad_triple_items))

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


def maxvalue_assignment2(score_dict):
    """
    TODO: upate kwarray with alternative formulation
    """
    import networkx as nx
    graph = nx.Graph()
    for (name1, name2), score in score_dict.items():
        graph.add_edge(name1, name2, score=score)
    assignment = nx.algorithms.matching.max_weight_matching(graph, weight='score')
    return assignment
