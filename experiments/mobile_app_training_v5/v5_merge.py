#!/usr/bin/env python3
"""
Merge positive tiles + hard negatives into a single training kwcoco
for the next training round.

Round-loop semantics (see DESIGN.md):

  Round 0 training kwcoco =
      all positive tiles
    + a random subsample of negative tiles, ratio NEG_OVER_POS

  Round N training kwcoco =
      all positive tiles
    + hard negatives mined from round N-1

The positive-tile pool is constant across rounds. Only the negative
half changes — round 0 uses a uniform random sample, later rounds use
the previous round's false-positive predictions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import scriptconfig as scfg


class V5MergeCLI(scfg.DataConfig):
    pos_kwcoco = scfg.Value(None, help='kwcoco bundle of positive tiles', required=True)
    neg_kwcoco = scfg.Value(None,
        help='kwcoco bundle of negative tiles (round 0) OR hard negatives (round N>0)',
        required=True)
    dst = scfg.Value(None, help='output kwcoco for training', required=True)

    neg_over_pos = scfg.Value(3.0,
        help='target ratio of negatives to positives in the output. '
             'Capped by the actual neg pool size. Set <=0 to keep ALL negatives.')
    seed = scfg.Value(0, help='RNG seed for negative subsampling')
    round_index = scfg.Value(0, help='which mining round this merge is for (informational)')

    @classmethod
    def main(cls, argv=1, **kwargs):
        config = cls.cli(argv=argv, data=kwargs, strict=True)
        run(config)


def run(config):
    import kwcoco
    import numpy as np

    pos_fpath = Path(str(config.pos_kwcoco)).expanduser().resolve()
    neg_fpath = Path(str(config.neg_kwcoco)).expanduser().resolve()
    dst_fpath = Path(str(config.dst)).expanduser().resolve()

    pos_dset = kwcoco.CocoDataset.coerce(str(pos_fpath))
    neg_dset = kwcoco.CocoDataset.coerce(str(neg_fpath))

    # Filter both sides to the right role so a misclassified mix doesn't
    # silently leak negatives into pos or vice versa.
    pos_gids = [img['id'] for img in pos_dset.images().objs
                if img.get('tile_role', 'positive') == 'positive']
    neg_gids = [img['id'] for img in neg_dset.images().objs
                if img.get('tile_role', 'negative') == 'negative']
    if not pos_gids:
        raise RuntimeError(f'no positive tiles in {pos_fpath}')

    rng = np.random.RandomState(int(config.seed))
    if float(config.neg_over_pos) > 0:
        target_n_neg = int(round(float(config.neg_over_pos) * len(pos_gids)))
        target_n_neg = min(target_n_neg, len(neg_gids))
    else:
        target_n_neg = len(neg_gids)

    if len(neg_gids) > target_n_neg:
        neg_gids_picked = list(
            rng.choice(neg_gids, size=target_n_neg, replace=False))
    else:
        neg_gids_picked = list(neg_gids)

    print(f'v5_merge: round={int(config.round_index)} '
          f'pos={len(pos_gids)} neg_pool={len(neg_gids)} '
          f'neg_picked={len(neg_gids_picked)} '
          f'(target ratio neg/pos={float(config.neg_over_pos)})')

    # Build the output kwcoco. We materialise as a new dataset because
    # subset operations across two different source bundles need
    # careful gid remapping.
    out_dset = kwcoco.CocoDataset()
    out_dset.fpath = str(dst_fpath)
    cat_id = out_dset.add_category(name='poop')

    # ---- positive side: copy images + their annotations ----
    pos_set = set(pos_gids)
    src_gid_to_new_gid = {}
    for img in pos_dset.images().objs:
        if img['id'] not in pos_set:
            continue
        new_img = {k: v for k, v in img.items() if k != 'id'}
        new_gid = out_dset.add_image(**new_img)
        src_gid_to_new_gid[('pos', img['id'])] = new_gid

    pos_cats_by_id = {c['id']: c for c in pos_dset.dataset.get('categories', [])}
    for ann in pos_dset.dataset.get('annotations', []):
        src_gid = ann.get('image_id')
        if src_gid not in pos_set:
            continue
        src_cat = pos_cats_by_id.get(ann.get('category_id'))
        if not src_cat or src_cat['name'] != 'poop':
            continue
        new_gid = src_gid_to_new_gid[('pos', src_gid)]
        bbox = ann.get('bbox')
        if not bbox:
            continue
        out_dset.add_annotation(
            image_id=new_gid, category_id=cat_id,
            bbox=list(bbox), area=float(ann.get('area', bbox[2] * bbox[3])),
            iscrowd=int(ann.get('iscrowd', 0)),
        )

    # ---- negative side: images only, no annotations ----
    neg_set = set(neg_gids_picked)
    for img in neg_dset.images().objs:
        if img['id'] not in neg_set:
            continue
        new_img = {k: v for k, v in img.items() if k != 'id'}
        # Make sure file_name resolves relative to the neg_dset's bundle
        # if it was relative. kwcoco's add_image will normalise on dump.
        out_dset.add_image(**new_img)

    out_dset.dump()
    pos_after = sum(1 for img in out_dset.images().objs
                    if img.get('tile_role') == 'positive')
    neg_after = sum(1 for img in out_dset.images().objs
                    if img.get('tile_role') == 'negative')
    print(f'  wrote {pos_after} pos + {neg_after} neg images, '
          f'{out_dset.n_annots} annots to {dst_fpath}')


__cli__ = V5MergeCLI


if __name__ == '__main__':
    __cli__.main()
