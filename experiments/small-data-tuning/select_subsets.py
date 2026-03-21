#!/usr/bin/env python3
"""
Deterministically materialize small benchmark cohorts for fast ShitSpotter
training experiments.

The main design goal is auditability. Every generated cohort includes:

* the source dataset paths,
* the subset-selection parameters,
* the exact selected image ids,
* summary statistics for each split,
* exported dataset files in both kwcoco and json form.

The initial selection method is intentionally simple: stratified random sampling
over positive-vs-negative images. The structure is written so future selection
methods can slot into the same manifest format.
"""

from __future__ import annotations

import argparse
import datetime as datetime_mod
import hashlib
import json
import pathlib
import random
from typing import Dict, List


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Create deterministic small-data benchmark cohorts for ShitSpotter.',
    )
    parser.add_argument('--train_src', required=True, help='Source training kwcoco file')
    parser.add_argument('--vali_src', required=True, help='Source validation kwcoco file')
    parser.add_argument('--test_src', required=True, help='Source test kwcoco file')
    parser.add_argument('--out_root', required=True, help='Directory where cohort directories will be written')
    parser.add_argument(
        '--train_sizes',
        nargs='+',
        type=int,
        default=[128, 256, 512],
        help='One or more training subset sizes to materialize',
    )
    parser.add_argument('--vali_size', type=int, default=64, help='Validation subset size')
    parser.add_argument('--test_size', type=int, default=64, help='Test subset size')
    parser.add_argument(
        '--selector',
        default='random',
        choices=['random'],
        help='Subset-selection method. Only random is implemented today.',
    )
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument(
        '--stratify',
        default='positive_negative',
        choices=['none', 'positive_negative'],
        help='Selection stratification policy',
    )
    parser.add_argument(
        '--absolute_paths',
        action='store_true',
        help='If set, reroot exported datasets to absolute image paths',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If set, replace existing cohort directories',
    )
    return parser


def _iso_now() -> str:
    return datetime_mod.datetime.now(datetime_mod.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _hash_list(values: List[int]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(values), sort_keys=True).encode('utf-8'))
    return digest.hexdigest()


def _load_kwcoco_dataset(fpath):
    import kwcoco

    dset = kwcoco.CocoDataset(fpath)
    if getattr(dset, 'fpath', None) is None:
        dset.fpath = str(pathlib.Path(fpath).expanduser().resolve())
    return dset


def _annotation_counts_by_image(dset) -> Dict[int, int]:
    gid_to_num_annots = {gid: 0 for gid in dset.images()}
    for ann in dset.dataset.get('annotations', []):
        gid = ann['image_id']
        gid_to_num_annots[gid] = gid_to_num_annots.get(gid, 0) + 1
    return gid_to_num_annots


def _choose_random_subset(
    gids: List[int],
    gid_to_num_annots: Dict[int, int],
    size: int,
    rng: random.Random,
    stratify: str,
) -> List[int]:
    if size >= len(gids):
        return sorted(gids)

    if stratify == 'none':
        return sorted(rng.sample(gids, size))

    positive_gids = [gid for gid in gids if gid_to_num_annots.get(gid, 0) > 0]
    negative_gids = [gid for gid in gids if gid_to_num_annots.get(gid, 0) == 0]
    total = len(gids)
    positive_target = round(size * (len(positive_gids) / total)) if total else 0

    if positive_gids and size > 0:
        positive_target = max(1, positive_target)
    positive_target = min(positive_target, len(positive_gids))
    negative_target = size - positive_target
    negative_target = min(negative_target, len(negative_gids))

    # Fill any remaining slots from whichever pool still has capacity.
    while positive_target + negative_target < size:
        if positive_target < len(positive_gids):
            positive_target += 1
        elif negative_target < len(negative_gids):
            negative_target += 1
        else:
            break

    chosen = []
    if positive_target:
        chosen.extend(rng.sample(positive_gids, positive_target))
    if negative_target:
        chosen.extend(rng.sample(negative_gids, negative_target))
    chosen = sorted(chosen)
    if len(chosen) != min(size, len(gids)):
        raise AssertionError('Subset selection produced an unexpected number of gids')
    return chosen


def _subset_stats(dset, chosen_gids: List[int]) -> Dict[str, object]:
    gid_to_num_annots = _annotation_counts_by_image(dset)
    num_positive = sum(gid_to_num_annots.get(gid, 0) > 0 for gid in chosen_gids)
    num_negative = len(chosen_gids) - num_positive
    chosen_anns = []
    category_hist = {}
    for ann in dset.dataset.get('annotations', []):
        if ann['image_id'] in set(chosen_gids):
            chosen_anns.append(ann)
            cid = ann['category_id']
            category_hist[cid] = category_hist.get(cid, 0) + 1
    return {
        'num_images': len(chosen_gids),
        'num_positive_images': num_positive,
        'num_negative_images': num_negative,
        'num_annotations': len(chosen_anns),
        'category_hist': category_hist,
        'selected_gids_hash': _hash_list(chosen_gids),
    }


def _dump_subset(
    dset,
    chosen_gids: List[int],
    kwcoco_fpath: pathlib.Path,
    mscoco_fpath: pathlib.Path,
    absolute_paths: bool,
) -> None:
    subset = dset.subset(chosen_gids)
    if absolute_paths:
        subset.reroot(absolute=True)
    subset.fpath = str(kwcoco_fpath)
    subset.dump(str(kwcoco_fpath))
    subset.fpath = str(mscoco_fpath)
    subset.dump(str(mscoco_fpath))


def _cohort_name(seed: int, train_size: int, vali_size: int, test_size: int, selector: str) -> str:
    return f'{selector}_seed{seed}_train{train_size:04d}_vali{vali_size:04d}_test{test_size:04d}'


def _materialize_one_cohort(args, train_size: int) -> pathlib.Path:
    train_dset = _load_kwcoco_dataset(args.train_src)
    vali_dset = _load_kwcoco_dataset(args.vali_src)
    test_dset = _load_kwcoco_dataset(args.test_src)

    cohort_name = _cohort_name(args.seed, train_size, args.vali_size, args.test_size, args.selector)
    cohort_dpath = pathlib.Path(args.out_root) / cohort_name
    if cohort_dpath.exists() and not args.overwrite:
        raise FileExistsError(f'Cohort already exists: {cohort_dpath}')
    cohort_dpath.mkdir(parents=True, exist_ok=True)

    split_to_dset = {
        'train': train_dset,
        'vali': vali_dset,
        'test': test_dset,
    }
    split_to_size = {
        'train': train_size,
        'vali': args.vali_size,
        'test': args.test_size,
    }

    manifest = {
        'cohort_name': cohort_name,
        'created_at_utc': _iso_now(),
        'selector': {
            'method': args.selector,
            'seed': args.seed,
            'stratify': args.stratify,
            'absolute_paths': bool(args.absolute_paths),
        },
        'sources': {},
        'subsets': {},
    }

    for split_name, dset in split_to_dset.items():
        split_seed = args.seed + {'train': 0, 'vali': 1000, 'test': 2000}[split_name]
        rng = random.Random(split_seed)
        gid_to_num_annots = _annotation_counts_by_image(dset)
        chosen_gids = _choose_random_subset(
            gids=sorted(list(dset.images())),
            gid_to_num_annots=gid_to_num_annots,
            size=split_to_size[split_name],
            rng=rng,
            stratify=args.stratify,
        )
        stats = _subset_stats(dset, chosen_gids)
        kwcoco_fpath = cohort_dpath / f'{split_name}.kwcoco.zip'
        mscoco_fpath = cohort_dpath / f'{split_name}.mscoco.json'
        gid_list_fpath = cohort_dpath / f'{split_name}_selected_gids.json'

        _dump_subset(
            dset=dset,
            chosen_gids=chosen_gids,
            kwcoco_fpath=kwcoco_fpath,
            mscoco_fpath=mscoco_fpath,
            absolute_paths=args.absolute_paths,
        )
        gid_list_fpath.write_text(json.dumps(chosen_gids, indent=2))

        manifest['sources'][split_name] = {
            'src_fpath': str(pathlib.Path(dset.fpath).expanduser().resolve()),
            'requested_num_images': split_to_size[split_name],
        }
        manifest['subsets'][split_name] = {
            'kwcoco_fpath': str(kwcoco_fpath),
            'mscoco_fpath': str(mscoco_fpath),
            'selected_gids_fpath': str(gid_list_fpath),
            'selected_gids': chosen_gids,
            'selection_seed': split_seed,
            'stats': stats,
        }

    summary_rows = []
    for split_name in ['train', 'vali', 'test']:
        stats = manifest['subsets'][split_name]['stats']
        summary_rows.append({
            'split': split_name,
            'num_images': stats['num_images'],
            'num_positive_images': stats['num_positive_images'],
            'num_negative_images': stats['num_negative_images'],
            'num_annotations': stats['num_annotations'],
            'selected_gids_hash': stats['selected_gids_hash'],
        })
    (cohort_dpath / 'split_summary.json').write_text(json.dumps(summary_rows, indent=2))
    (cohort_dpath / 'cohort_manifest.json').write_text(json.dumps(manifest, indent=2))
    return cohort_dpath


def main(argv=None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    out_root = pathlib.Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written = []
    for train_size in args.train_sizes:
        cohort_dpath = _materialize_one_cohort(args, train_size=train_size)
        written.append(str(cohort_dpath))

    print(json.dumps({
        'written_cohorts': written,
        'selector': args.selector,
        'seed': args.seed,
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
