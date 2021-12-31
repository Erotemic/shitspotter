import dateutil
import dateutil.parser
import functools
import kwimage
import numpy as np
import pandas as pd
import ubelt as ub


def autofind_pair_hueristic(coco_dset):
    """
    import shitspotter
    coco_dset = shitspotter.open_shit_coco()
    """
    import vtool_ibeis
    from kwcoco.util import util_json
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

    import shelve
    cache_dpath = ub.Path(coco_dset.cache_dpath).ensuredir()
    cache_fpath = cache_dpath / 'pairwise_score_cache.shelf'
    shelf = shelve.open(str(cache_fpath))

    # .setdefault('image_matches', {})
    image_matches = shelf
    # for match in coco_dset.dataset['image_matches']:
    #     name1 = match['name1']
    #     name2 = match['name2']
    #     pair = (name1, name2)
    #     if pair not in image_matches:
    #         image_matches[pair] = match

    pairs = list(ub.iter_window(ordered_gids, 2))
    for gid1, gid2 in ub.ProgIter(pairs, verbose=3):
        name1 = coco_dset.index.imgs[gid1]['name']
        name2 = coco_dset.index.imgs[gid2]['name']
        pair = (name1, name2)
        key = ub.repr2(pair, compact=1)
        if key  in image_matches:
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

            match = util_json.ensure_json_serializable(match)
            print('match = {}'.format(ub.repr2(match, nl=1)))
            image_matches[key] = match

    # Save the match table
    image_matches.sync()

    # coco_dset.dump(coco_dset.fpath, newlines=True)

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

    bad_pair_estimate = bad_pairwise_items // 2
    bad_triple_estimate = bad_triple_items // 2

    total_unmatchable_tups = bad_pair_estimate + bad_triple_estimate
    total_matchable_tups = good_pairs + good_triples
    total_estimated_number_of_tups = total_matchable_tups + total_unmatchable_tups

    print('total_matchable_tups = {!r}'.format(total_matchable_tups))
    print('total_unmatchable_tups = {!r}'.format(total_unmatchable_tups))
    print('total_est_pairs= {!r}'.format(good_pairs + bad_pair_estimate))
    print('total_unmatchable_tups = {!r}'.format(total_unmatchable_tups))

    print('total_estimated_number_of_tups = {!r}'.format(total_estimated_number_of_tups))
    print('total_estimated_number_of_pairs = {!r}'.format(total_estimated_number_of_pairs))


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
