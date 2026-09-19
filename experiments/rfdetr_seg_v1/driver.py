#!/usr/bin/env python
"""Staged, artifact-derived setup for the ShitSpotter RF-DETR campaign.

No command in this driver starts the long production run. ``prepare`` writes
and prints the exact round-0 command after its inputs validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def load_config(fpath=HERE / "config.yaml"):
    config = yaml.safe_load(Path(fpath).read_text())
    if root := os.environ.get("SHITSPOTTER_RFDETR_ROOT"):
        config["paths"]["output_root"] = str(Path(root).expanduser().resolve())
    if cache := os.environ.get("SHITSPOTTER_RFDETR_CACHE"):
        config["paths"]["cache"] = str(Path(cache).expanduser().resolve())
    elif root := os.environ.get("SHITSPOTTER_RFDETR_ROOT"):
        config["paths"]["cache"] = str(Path(root).expanduser().resolve() / "tile_cache")
    return config


def atomic_json_dump(data, fpath):
    fpath = Path(fpath)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=fpath.parent, delete=False) as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")
        tmp = Path(file.name)
    os.replace(tmp, fpath)


def sha256_file(fpath, chunk_size=1024 * 1024):
    hasher = hashlib.sha256()
    with open(fpath, "rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_kdk(config):
    import sys
    repo = str(Path(config["paths"]["kdk_repo"]).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def census(config, hash_assets=False):
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.tile_geometry import segmentation_kind

    report = {"schema_version": 1, "splits": {}}
    for split, src in config["splits"].items():
        src = Path(src).resolve()
        dset = kwcoco.CocoDataset.coerce(str(src))
        target_cids = {
            cat["id"] for cat in dset.dataset.get("categories", [])
            if cat["name"] in config["category_names"]
        }
        anns = [ann for ann in dset.annots().objs if ann.get("category_id") in target_cids]
        positive_gids = {ann["image_id"] for ann in anns}
        seg_types = Counter(segmentation_kind(ann.get("segmentation")) for ann in anns)
        image_sizes = Counter(
            (int(img.get("width", 0)), int(img.get("height", 0)))
            for img in dset.images().objs
        )
        split_report = {
            "manifest": str(src),
            "manifest_sha256": sha256_file(src),
            "num_images": dset.n_images,
            "num_target_annotations": len(anns),
            "num_positive_images": len(positive_gids),
            "num_zero_target_images": dset.n_images - len(positive_gids),
            "segmentation_kinds": dict(sorted(seg_types.items())),
            "common_image_sizes": [
                {"width": wh[0], "height": wh[1], "count": count}
                for wh, count in image_sizes.most_common(20)
            ],
        }
        if hash_assets:
            assets = []
            for img in dset.images().objs:
                path = Path(dset.get_image_fpath(img["id"])).resolve()
                assets.append({
                    "image_id": img["id"], "path": str(path),
                    "num_bytes": path.stat().st_size, "sha256": sha256_file(path),
                })
            split_report["source_assets"] = assets
        report["splits"][split] = split_report
    dst = Path(config["paths"]["output_root"]) / "census.json"
    atomic_json_dump(report, dst)
    print(dst)


def _run_tile(config, split, dst):
    require_kdk(config)
    from kwcoco_detector_kit.data.tile import TileConfig, run
    policy = config["tiles"]
    tile_config = TileConfig.cli(argv=False, data={
        "src": config["splits"][split], "dst": str(dst),
        "mode": "multiscale", "category_names": ",".join(config["category_names"]),
        "tile_size": policy["size"], "oversize_factor": policy["oversize_factor"],
        "source_scales": ",".join(map(str, policy["source_scales"])),
        "stride_frac": policy["stride_frac"],
        "min_keep_fraction": policy["min_keep_fraction"],
        "min_gt_area_frac": policy["min_gt_area_frac"],
        "negative_safety_margin": policy["negative_safety_margin"],
        "jpeg_quality": policy["jpeg_quality"],
        "cache_dpath": config["paths"]["cache"], "keep_negative": True,
    })
    run(tile_config)


def build_pools(config, force=False):
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.merge import MergeConfig, run as merge_run

    root = Path(config["paths"]["output_root"])
    pools = root / "pools"
    pools.mkdir(parents=True, exist_ok=True)
    train_tiles = pools / "train_all_tiles.kwcoco.zip"
    vali_tiles = pools / "train_validation_tiles.kwcoco.zip"
    round0 = root / "rounds" / "round0" / "train.kwcoco.zip"
    for split, dst in [("train", train_tiles), ("validation", vali_tiles)]:
        if dst.exists() and not force:
            kwcoco.CocoDataset.coerce(str(dst)).validate()
            print(f"reuse valid pool: {dst}")
        else:
            _run_tile(config, split, dst)
            kwcoco.CocoDataset.coerce(str(dst)).validate()
    if round0.exists() and not force:
        kwcoco.CocoDataset.coerce(str(round0)).validate()
        print(f"reuse valid round: {round0}")
    else:
        round0.parent.mkdir(parents=True, exist_ok=True)
        merge_cfg = MergeConfig.cli(argv=False, data={
            "pos_kwcoco": str(train_tiles), "neg_kwcoco": str(train_tiles),
            "dst": str(round0),
            "category_names": ",".join(config["category_names"]),
            "neg_over_pos": config["round0"]["negative_over_positive"],
            "seed": config["round0"]["seed"], "round_index": 0,
        })
        merge_run(merge_cfg)
        kwcoco.CocoDataset.coerce(str(round0)).validate()


def prepare(config):
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.trainers._registry import get_trainer

    root = Path(config["paths"]["output_root"])
    train = root / "rounds" / "round0" / "train.kwcoco.zip"
    vali = root / "pools" / "train_validation_tiles.kwcoco.zip"
    for path in [train, vali]:
        if not path.exists():
            raise FileNotFoundError(f"build-pools must produce {path}")
        kwcoco.CocoDataset.coerce(str(path)).validate()
    policy = config["rfdetr"]
    workdir = root / "rounds" / "round0" / "workdir"
    cfg_path = get_trainer("rfdetr").generate_config(
        train, vali, workdir, variant=policy["variant"],
        input_hw=tuple(policy["input_hw"]), train_policy="fixed",
        num_classes=len(config["category_names"]),
        batch_size=policy["batch_size_per_gpu"],
        val_batch_size=policy["validation_batch_size_per_gpu"],
        num_epochs=policy["epochs"], lr=policy["lr"],
        backbone_lr=policy["backbone_lr"], use_amp=policy["use_amp"],
        channels="r|g|b", scale_tier="2XL", num_gpus=policy["num_gpus"],
        data_format="kwcoco", extra={
            "category_names": config["category_names"],
            "grad_accum_steps": policy["grad_accum_steps"],
            "num_workers": policy["num_workers"],
        },
    )
    kdk = Path(config["paths"]["kdk_repo"]).resolve()
    launcher = cfg_path.parent / "launch_rfdetr.py"
    command = (
        "docker run --rm --gpus all --ipc=host --shm-size=64g "
        f"-v /data:/data -v {kdk}:{kdk} -w {kdk} "
        f"{policy['image']} \"python -m torch.distributed.run "
        f"--nproc_per_node={policy['num_gpus']} {launcher} --config {cfg_path}\""
    )
    launch_fpath = workdir / "ROUND0_COMMAND.txt"
    launch_fpath.write_text(command + "\n")
    print("READY COMMAND (not executed):")
    print(command)


def status(config):
    import kwcoco
    root = Path(config["paths"]["output_root"])
    required = {
        "census": root / "census.json",
        "train_tiles": root / "pools" / "train_all_tiles.kwcoco.zip",
        "train_validation_tiles": root / "pools" / "train_validation_tiles.kwcoco.zip",
        "round0_manifest": root / "rounds" / "round0" / "train.kwcoco.zip",
        "rfdetr_config": root / "rounds" / "round0" / "workdir" / "generated_configs" / "rfdetr_train.json",
        "round0_command": root / "rounds" / "round0" / "workdir" / "ROUND0_COMMAND.txt",
    }
    for name, path in required.items():
        valid = path.is_file()
        if valid and ".kwcoco." in path.name:
            try:
                kwcoco.CocoDataset.coerce(str(path)).validate()
            except Exception:
                valid = False
        print(f"{'complete' if valid else 'not-ready':10s} {name:24s} {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["status", "census", "build-pools", "prepare"])
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--hash-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "status":
        status(config)
    elif args.command == "census":
        census(config, hash_assets=args.hash_assets)
    elif args.command == "build-pools":
        build_pools(config, force=args.force)
    elif args.command == "prepare":
        prepare(config)


if __name__ == "__main__":
    main()
