#!/usr/bin/env python3
"""
Materialize a reproducible detector benchmark for DINOv2-vs-DINOv3
comparisons.

This script intentionally prepares the shared data substrate once so that
OpenGroundingDINO and DEIMv2 consume the same preprocessed train / validation /
test datasets. The current benchmark recipe is:

* training subsets are positives-only random samples from the canonical train
  split,
* validation and test are derived from the canonical full splits,
* all splits are rewritten to a poop-only view,
* all splits are resized offline,
* all splits are box-simplified after resizing,
* metadata records both source and prepared dataset statistics.
"""

from __future__ import annotations

import argparse
import datetime as datetime_mod
import hashlib
import json
import pathlib
import random
import shutil
from typing import Iterable


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Prepare a shared DINO detector benchmark for small-data tuning.',
    )
    parser.add_argument('--train_src', required=True)
    parser.add_argument('--vali_src', required=True)
    parser.add_argument('--test_src', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--train_sizes', nargs='+', type=int, default=[128, 256, 512])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--selector', default='random_positives_only')
    parser.add_argument('--resize_max_dim', type=int, default=640)
    parser.add_argument('--resize_output_ext', default='.jpg')
    parser.add_argument(
        '--simplify_minimum_instances',
        type=int,
        default=1,
        help='Applied after filtering to poop-only so the sole class is kept.',
    )
    parser.add_argument('--overwrite', action='store_true')
    return parser


def _iso_now() -> str:
    return datetime_mod.datetime.now(datetime_mod.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _truthy_hash(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(values), sort_keys=True).encode('utf-8'))
    return digest.hexdigest()


def _load_dset(fpath):
    import kwcoco

    dset = kwcoco.CocoDataset.coerce(fpath)
    dset.reroot(absolute=True)
    return dset


def _stats_for_dset(dset) -> dict:
    annots = dset.annots().objs
    gids = list(dset.images())
    gid_to_aids = dset.index.gid_to_aids
    num_positive = sum(len(gid_to_aids.get(gid, [])) > 0 for gid in gids)
    num_negative = len(gids) - num_positive
    category_hist = {}
    for ann in annots:
        cid = ann.get('category_id', None)
        category_hist[cid] = category_hist.get(cid, 0) + 1
    return {
        'num_images': len(gids),
        'num_positive_images': num_positive,
        'num_negative_images': num_negative,
        'num_annotations': len(annots),
        'category_hist': category_hist,
    }


def _positive_poop_gids(dset) -> list[int]:
    poop_cids = [cid for cid, cat in dset.cats.items() if cat.get('name') == 'poop']
    poop_cids = set(poop_cids)
    positive = []
    for gid in dset.images():
        aids = dset.index.gid_to_aids.get(gid, [])
        if any(dset.anns[aid].get('category_id') in poop_cids for aid in aids):
            positive.append(gid)
    return sorted(positive)


def _random_subset(gids: list[int], size: int, rng: random.Random) -> list[int]:
    if size >= len(gids):
        return list(gids)
    return sorted(rng.sample(gids, size))


def _poop_only_subset(src_dset, chosen_gids: list[int], keep_only_positive_images: bool):
    subset = src_dset.subset(chosen_gids)
    poop_cids = [cid for cid, cat in subset.cats.items() if cat.get('name') == 'poop']
    remove_cids = [cid for cid in subset.cats if cid not in poop_cids]
    if remove_cids:
        subset.remove_categories(remove_cids)
    if keep_only_positive_images:
        empty_gids = [gid for gid, aids in subset.index.gid_to_aids.items() if len(aids) == 0]
        if empty_gids:
            subset.remove_images(empty_gids)
    subset.reroot(absolute=True)
    return subset


def _dump_json(data, fpath: pathlib.Path) -> None:
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(data, indent=2))


def _run_resize(src_fpath: pathlib.Path, dst_fpath: pathlib.Path, asset_dname: str, max_dim: int, output_ext: str):
    from shitspotter.cli.resize_kwcoco import resize_kwcoco

    resize_kwcoco(
        src=src_fpath,
        dst=dst_fpath,
        max_dim=max_dim,
        asset_dname=asset_dname,
        output_ext=output_ext,
    )


def _run_simplify(src_fpath: pathlib.Path, dst_fpath: pathlib.Path, minimum_instances: int):
    from shitspotter.cli.simplify_kwcoco import SimplifyKwcocoCLI

    SimplifyKwcocoCLI.main(
        argv=0,
        src=str(src_fpath),
        dst=str(dst_fpath),
        minimum_instances=minimum_instances,
    )


def _export_coco(src_kwcoco: pathlib.Path, dst_json: pathlib.Path):
    from shitspotter.algo_foundation_v3.coco_adapter import _build_coco_export

    _build_coco_export(
        src=src_kwcoco,
        dst=dst_json,
        category_name='poop',
        include_segmentations=True,
        category_id=0,
    )


def _materialize_prepared_split(src_dset, chosen_gids: list[int], split_dpath: pathlib.Path, *,
                                split_tag: str, keep_only_positive_images: bool,
                                resize_max_dim: int, resize_output_ext: str,
                                simplify_minimum_instances: int) -> dict:
    raw_kwcoco = split_dpath / f'{split_tag}.poop_only.kwcoco.zip'
    resized_kwcoco = split_dpath / f'{split_tag}.poop_only_maxdim{resize_max_dim}.kwcoco.zip'
    simplified_kwcoco = split_dpath / f'{split_tag}.poop_only_maxdim{resize_max_dim}.simplified.kwcoco.zip'
    exported_json = split_dpath / f'{split_tag}.poop_only_maxdim{resize_max_dim}.simplified.mscoco.json'
    selected_gids_fpath = split_dpath / f'{split_tag}.selected_gids.json'

    subset = _poop_only_subset(src_dset, chosen_gids, keep_only_positive_images=keep_only_positive_images)
    subset.fpath = str(raw_kwcoco)
    subset.dump(str(raw_kwcoco))
    _dump_json(chosen_gids, selected_gids_fpath)

    _run_resize(
        src_fpath=raw_kwcoco,
        dst_fpath=resized_kwcoco,
        asset_dname=f'{split_tag}_assets_maxdim{resize_max_dim}',
        max_dim=resize_max_dim,
        output_ext=resize_output_ext,
    )
    _run_simplify(
        src_fpath=resized_kwcoco,
        dst_fpath=simplified_kwcoco,
        minimum_instances=simplify_minimum_instances,
    )
    _export_coco(
        src_kwcoco=simplified_kwcoco,
        dst_json=exported_json,
    )

    raw_stats = _stats_for_dset(_load_dset(raw_kwcoco))
    resized_stats = _stats_for_dset(_load_dset(resized_kwcoco))
    simplified_stats = _stats_for_dset(_load_dset(simplified_kwcoco))

    return {
        'selected_gids_fpath': str(selected_gids_fpath),
        'selected_gids_hash': _truthy_hash(chosen_gids),
        'num_selected_gids': len(chosen_gids),
        'raw_kwcoco_fpath': str(raw_kwcoco),
        'resized_kwcoco_fpath': str(resized_kwcoco),
        'prepared_kwcoco_fpath': str(simplified_kwcoco),
        'prepared_mscoco_fpath': str(exported_json),
        'raw_stats': raw_stats,
        'resized_stats': resized_stats,
        'prepared_stats': simplified_stats,
    }


def main(argv=1, **kwargs) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv if isinstance(argv, list) else None)
    if kwargs:
        for key, value in kwargs.items():
            setattr(args, key, value)

    out_root = pathlib.Path(args.out_root)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_dset = _load_dset(args.train_src)
    vali_dset = _load_dset(args.vali_src)
    test_dset = _load_dset(args.test_src)

    manifest = {
        'benchmark_name': 'dino_detector_benchmark_v1',
        'created_at_utc': _iso_now(),
        'selector': {
            'method': args.selector,
            'seed': args.seed,
            'train_positive_only': True,
            'eval_positive_only': True,
        },
        'preprocessing': {
            'category_filter': 'poop_only',
            'resize_max_dim': args.resize_max_dim,
            'resize_output_ext': args.resize_output_ext,
            'box_simplify': True,
            'simplify_minimum_instances': args.simplify_minimum_instances,
            'notes': [
                'Validation and test originate from the canonical full splits, then are converted into poop-only positive-image prepared datasets for detector benchmarking.',
                'Prepared kwcoco and mscoco exports are shared between OpenGroundingDINO and DEIMv2 to preserve apples-to-apples semantics.',
            ],
        },
        'sources': {
            'train': str(pathlib.Path(args.train_src).resolve()),
            'vali': str(pathlib.Path(args.vali_src).resolve()),
            'test': str(pathlib.Path(args.test_src).resolve()),
        },
        'source_stats': {
            'train': _stats_for_dset(train_dset),
            'vali': _stats_for_dset(vali_dset),
            'test': _stats_for_dset(test_dset),
        },
        'train_subsets': {},
        'eval_sets': {},
    }

    eval_dpath = out_root / 'eval_sets'
    eval_dpath.mkdir(parents=True, exist_ok=True)
    for split_name, src_dset in [('vali', vali_dset), ('test', test_dset)]:
        chosen_gids = _positive_poop_gids(src_dset)
        split_info = _materialize_prepared_split(
            src_dset=src_dset,
            chosen_gids=chosen_gids,
            split_dpath=eval_dpath / split_name,
            split_tag=split_name,
            keep_only_positive_images=True,
            resize_max_dim=args.resize_max_dim,
            resize_output_ext=args.resize_output_ext,
            simplify_minimum_instances=args.simplify_minimum_instances,
        )
        split_info['source_num_positive_poop_images'] = len(chosen_gids)
        manifest['eval_sets'][split_name] = split_info

    positive_train_gids = _positive_poop_gids(train_dset)
    for train_size in args.train_sizes:
        subset_name = f'train{train_size:04d}'
        rng = random.Random(args.seed + train_size)
        chosen_gids = _random_subset(positive_train_gids, train_size, rng)
        split_info = _materialize_prepared_split(
            src_dset=train_dset,
            chosen_gids=chosen_gids,
            split_dpath=out_root / 'train_subsets' / subset_name,
            split_tag=subset_name,
            keep_only_positive_images=True,
            resize_max_dim=args.resize_max_dim,
            resize_output_ext=args.resize_output_ext,
            simplify_minimum_instances=args.simplify_minimum_instances,
        )
        split_info['train_size'] = train_size
        split_info['selection_seed'] = args.seed + train_size
        manifest['train_subsets'][subset_name] = split_info

    _dump_json(manifest, out_root / 'benchmark_manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
