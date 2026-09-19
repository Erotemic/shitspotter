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
    config_fpath = Path(fpath).expanduser().resolve()
    config = yaml.safe_load(config_fpath.read_text())
    paths = config["paths"]
    if data_root := os.environ.get("SHITSPOTTER_DATA_DPATH"):
        paths["data_root"] = data_root
    data_root = Path(paths["data_root"]).expanduser().resolve()
    paths["data_root"] = str(data_root)
    for key, value in config["splits"].items():
        value = Path(value).expanduser()
        config["splits"][key] = str(
            value.resolve() if value.is_absolute() else (data_root / value).resolve()
        )
    kdk_repo = Path(paths["kdk_repo"]).expanduser()
    paths["kdk_repo"] = str(
        kdk_repo.resolve()
        if kdk_repo.is_absolute()
        else (config_fpath.parent / kdk_repo).resolve()
    )
    if root := os.environ.get("SHITSPOTTER_RFDETR_ROOT"):
        paths["output_root"] = str(Path(root).expanduser().resolve())
    else:
        paths["output_root"] = str(Path(paths["output_root"]).expanduser().resolve())
    if cache := os.environ.get("SHITSPOTTER_RFDETR_CACHE"):
        paths["cache"] = str(Path(cache).expanduser().resolve())
    elif root := os.environ.get("SHITSPOTTER_RFDETR_ROOT"):
        paths["cache"] = str(Path(root).expanduser().resolve() / "tile_cache")
    else:
        cache = Path(paths["cache"]).expanduser()
        paths["cache"] = str(
            cache.resolve()
            if cache.is_absolute()
            else (Path(paths["output_root"]) / cache).resolve()
        )
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


def config_fingerprint(config):
    """Fingerprint the fully resolved campaign recipe used by local artifacts."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_inputs(config, decode_samples=8):
    """Validate manifests, all referenced paths, and sampled image decoding."""
    import kwcoco
    import numpy as np

    report = {
        "schema_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "data_root": config["paths"]["data_root"],
        "splits": {},
    }
    for split, src in config["splits"].items():
        src = Path(src)
        if not src.is_file():
            raise FileNotFoundError(src)
        dset = kwcoco.CocoDataset.coerce(str(src))
        validation = dset.validate()
        # These source bundles intentionally contain a small number of
        # non-detection metadata annotations (category_id=None / no bbox).
        # RF-DETR export filters to the requested target categories, so record
        # the broader schema diagnostics but gate only the target rows here.
        target_cids = {
            cat["id"] for cat in dset.dataset.get("categories", [])
            if cat["name"] in config["category_names"]
        }
        bad_targets = [
            ann["id"] for ann in dset.annots().objs
            if ann.get("category_id") in target_cids
            and ann.get("bbox") is None
            and ann.get("segmentation") is None
        ]
        if bad_targets:
            raise RuntimeError(
                f"{split}: target annotations lack bbox and segmentation: {bad_targets[:10]}"
            )
        missing = []
        for img in dset.images().objs:
            fpath = Path(dset.get_image_fpath(img["id"]))
            if not fpath.is_file():
                missing.append({"image_id": img["id"], "path": str(fpath)})
        if missing:
            raise FileNotFoundError(
                f"{split}: {len(missing)} referenced assets are missing; examples={missing[:5]}"
            )
        gids = sorted(dset.images())
        if decode_samples and gids:
            sample_idxs = np.linspace(0, len(gids) - 1, min(int(decode_samples), len(gids))).astype(int)
            decoded = []
            for idx in sample_idxs:
                gid = gids[int(idx)]
                arr = dset.coco_image(gid).imdelay().finalize()
                decoded.append({
                    "image_id": gid,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                })
        else:
            decoded = []
        report["splits"][split] = {
            "manifest": str(src.resolve()),
            "manifest_sha256": sha256_file(src),
            "num_images": dset.n_images,
            "num_annotations": dset.n_annots,
            "num_missing_assets": 0,
            "source_validation": validation,
            "num_invalid_target_annotations": 0,
            "decoded_samples": decoded,
        }
    dst = Path(config["paths"]["output_root"]) / "input_verification.json"
    atomic_json_dump(report, dst)
    print(dst)
    return dst


def require_kdk(config):
    import sys
    repo = str(Path(config["paths"]["kdk_repo"]).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def census(config, hash_assets=False):
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.tile_geometry import segmentation_kind

    report = {
        "schema_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "splits": {},
    }
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
        bbox_widths = [float(ann["bbox"][2]) for ann in anns if ann.get("bbox")]
        bbox_heights = [float(ann["bbox"][3]) for ann in anns if ann.get("bbox")]
        mask_areas = [float(ann.get("area", 0)) for ann in anns if float(ann.get("area", 0)) > 0]
        normalized_areas = []
        for ann in anns:
            image = dset.imgs[ann["image_id"]]
            denom = float(image.get("width", 0) * image.get("height", 0))
            if denom > 0 and float(ann.get("area", 0)) > 0:
                normalized_areas.append(float(ann["area"]) / denom)

        def _quantiles(values):
            import numpy as np
            if not values:
                return {}
            return {
                str(q): float(np.quantile(values, q))
                for q in [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1]
            }

        split_report = {
            "manifest": str(src),
            "manifest_sha256": sha256_file(src),
            "num_images": dset.n_images,
            "num_target_annotations": len(anns),
            "num_positive_images": len(positive_gids),
            "num_zero_target_images": dset.n_images - len(positive_gids),
            "segmentation_kinds": dict(sorted(seg_types.items())),
            "bbox_width_quantiles_px": _quantiles(bbox_widths),
            "bbox_height_quantiles_px": _quantiles(bbox_heights),
            "mask_area_quantiles_px2": _quantiles(mask_areas),
            "mask_area_fraction_quantiles": _quantiles(normalized_areas),
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


def simulate_policy(config):
    """Count tile roles without decoding or encoding source imagery."""
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.tile import (
        _clip_annotation_geometry,
        _grid_positions,
        _keep_negative_window,
        _parse_scales,
    )

    policy = config["tiles"]
    base_tile = int(policy["size"])
    disk_tile = max(1, int(round(base_tile * float(policy["oversize_factor"]))))
    stride = max(1, int(round(disk_tile * float(policy["stride_frac"]))))
    min_area = float(policy["min_gt_area_frac"]) * base_tile * base_tile
    min_keep = float(policy["min_keep_fraction"])
    margin = int(policy["negative_safety_margin"])
    min_long_side = int(policy["min_source_scale_long_side"])
    scales = _parse_scales(policy["source_scales"])
    report = {
        "schema_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "tile_policy": policy,
        "splits": {},
    }

    for split in ["train", "validation"]:
        dset = kwcoco.CocoDataset.coerce(config["splits"][split])
        target_cids = {
            cat["id"] for cat in dset.dataset.get("categories", [])
            if cat["name"] in config["category_names"]
        }
        counts = {
            name: Counter()
            for name in ["positive", "negative", "ignore", "dropped_negative"]
        }
        negative_keep_fraction = float(policy["negative_keep_fraction"][split])
        anns_by_gid = {
            gid: [
                ann for ann in dset.annots(gid=gid).objs
                if ann.get("category_id") in target_cids
                and (ann.get("bbox") is not None or ann.get("segmentation") is not None)
            ]
            for gid in dset.images()
        }
        for image in dset.images().objs:
            width, height = int(image["width"]), int(image["height"])
            anns = anns_by_gid[image["id"]]
            for scale_name, requested_scale in scales:
                scaled_w = max(1, int(round(width * requested_scale)))
                scaled_h = max(1, int(round(height * requested_scale)))
                if max(scaled_w, scaled_h) < min_long_side:
                    continue
                actual_scale = (
                    scaled_w / float(width), scaled_h / float(height),
                )
                scaled_ann_boxes = []
                for ann in anns:
                    bbox = ann.get("bbox")
                    if bbox is None:
                        import kwimage
                        bbox = list(
                            kwimage.Segmentation.coerce(ann["segmentation"])
                            .to_multi_polygon().box().to_coco()
                        )
                    bx, by, bw, bh = map(float, bbox)
                    scaled_ann_boxes.append((
                        bx * actual_scale[0], by * actual_scale[1],
                        (bx + bw) * actual_scale[0], (by + bh) * actual_scale[1],
                    ))
                xs = _grid_positions(scaled_w, disk_tile, stride)
                ys = _grid_positions(scaled_h, disk_tile, stride)
                for x0 in xs:
                    for y0 in ys:
                        num_intersecting = 0
                        num_kept = 0
                        kept_area = 0.0
                        unsafe = False
                        crop = (x0, y0, x0 + disk_tile, y0 + disk_tile)
                        for ann, ann_box in zip(anns, scaled_ann_boxes):
                            ax0, ay0, ax1, ay1 = ann_box
                            exact_bbox_hit = (
                                min(x0 + disk_tile, ax1) > max(x0, ax0)
                                and min(y0 + disk_tile, ay1) > max(y0, ay0)
                            )
                            margin_bbox_hit = margin and (
                                min(x0 + disk_tile + margin, ax1) > max(x0 - margin, ax0)
                                and min(y0 + disk_tile + margin, ay1) > max(y0 - margin, ay0)
                            )
                            if not exact_bbox_hit and not margin_bbox_hit:
                                continue
                            if not exact_bbox_hit:
                                margin_geom = _clip_annotation_geometry(
                                    ann, source_dims=(height, width),
                                    scale=actual_scale,
                                    crop_xyxy=(
                                        x0 - margin, y0 - margin,
                                        x0 + disk_tile + margin,
                                        y0 + disk_tile + margin,
                                    ),
                                    output_dims=(
                                        disk_tile + 2 * margin,
                                        disk_tile + 2 * margin,
                                    ),
                                )
                                unsafe |= margin_geom is not None
                                continue
                            geom = _clip_annotation_geometry(
                                ann, source_dims=(height, width),
                                scale=actual_scale, crop_xyxy=crop,
                                output_dims=(disk_tile, disk_tile),
                            )
                            if geom is None:
                                if margin_bbox_hit:
                                    margin_geom = _clip_annotation_geometry(
                                        ann, source_dims=(height, width),
                                        scale=actual_scale,
                                        crop_xyxy=(
                                            x0 - margin, y0 - margin,
                                            x0 + disk_tile + margin,
                                            y0 + disk_tile + margin,
                                        ),
                                        output_dims=(
                                            disk_tile + 2 * margin,
                                            disk_tile + 2 * margin,
                                        ),
                                    )
                                    unsafe |= margin_geom is not None
                                continue
                            num_intersecting += 1
                            if geom.visible_fraction < min_keep:
                                unsafe = True
                            else:
                                num_kept += 1
                                kept_area += geom.area
                        if num_intersecting:
                            role = (
                                "positive"
                                if not unsafe and num_kept == num_intersecting and kept_area >= min_area
                                else "ignore"
                            )
                        else:
                            role = "ignore" if unsafe else "negative"
                        if role == "negative" and not _keep_negative_window(
                            fraction=negative_keep_fraction,
                            seed=config["round0"]["seed"],
                            source_gid=image["id"],
                            scale_name=scale_name,
                            x0=x0,
                            y0=y0,
                        ):
                            role = "dropped_negative"
                        counts[role][scale_name] += 1
        by_role = {
            role: {"total": sum(per_scale.values()), "by_scale": dict(per_scale)}
            for role, per_scale in counts.items()
        }
        report["splits"][split] = by_role

    smoke_root = Path(config["paths"]["output_root"]) / "smoke"
    smoke_manifests = [
        smoke_root / "train_all_tiles.kwcoco.zip",
        smoke_root / "train_validation_tiles.kwcoco.zip",
    ]
    encoded = []
    if all(path.is_file() for path in smoke_manifests):
        encoded = sorted({
            Path(image["file_name"])
            for manifest in smoke_manifests
            for image in kwcoco.CocoDataset.coerce(str(manifest)).images().objs
        })
    if encoded:
        total_bytes = sum(path.stat().st_size for path in encoded)
        avg_bytes = total_bytes / len(encoded)
        train_counts = report["splits"]["train"]
        vali_counts = report["splits"]["validation"]
        round0_neg = min(
            train_counts["negative"]["total"],
            round(float(config["round0"]["negative_over_positive"]) * train_counts["positive"]["total"]),
        )
        materialized = (
            train_counts["positive"]["total"] + round0_neg
            + vali_counts["positive"]["total"] + vali_counts["negative"]["total"]
        )
        persistent_pool = (
            train_counts["positive"]["total"] + train_counts["negative"]["total"]
            + vali_counts["positive"]["total"] + vali_counts["negative"]["total"]
        )
        report["storage_estimate"] = {
            "basis_num_smoke_jpegs": len(encoded),
            "basis_average_jpeg_bytes": avg_bytes,
            "persistent_pool_materialized_tiles": persistent_pool,
            "persistent_pool_estimated_bytes": persistent_pool * avg_bytes,
            "round0_materialized_tiles": materialized,
            "round0_estimated_bytes": materialized * avg_bytes,
            "note": "linear estimate from the real-data smoke cache; measure the full pool before training",
        }
    dst = Path(config["paths"]["output_root"]) / "tile_policy_simulation.json"
    atomic_json_dump(report, dst)
    print(dst)
    return dst


def _tile_expected(config, split, src, negative_keep_fraction, source_dataset_fingerprint):
    policy = config["tiles"]
    return {
        "src": str(Path(src).resolve()),
        "source_manifest_sha256": sha256_file(src),
        "config": {
            "mode": "multiscale",
            "category_names": ",".join(config["category_names"]),
            "jpeg_quality": policy["jpeg_quality"],
            "cache_dpath": str(Path(config["paths"]["cache"]).resolve()),
            "source_dataset_fingerprint": source_dataset_fingerprint,
            "oversize_factor": policy["oversize_factor"],
            "min_keep_fraction": policy["min_keep_fraction"],
            "tile_size": policy["size"],
            "source_scales": ",".join(map(str, policy["source_scales"])),
            "stride_frac": policy["stride_frac"],
            "min_gt_area_frac": policy["min_gt_area_frac"],
            "min_source_scale_long_side": policy["min_source_scale_long_side"],
            "negative_safety_margin": policy["negative_safety_margin"],
            "keep_negative": True,
            "negative_keep_fraction": negative_keep_fraction,
            "seed": config["round0"]["seed"],
        },
    }


def _tile_artifact_matches(config, split, path, *, src=None, negative_keep_fraction=None):
    """Return whether a tile manifest is valid and belongs to this recipe."""
    import kwcoco

    src = str(src or config["splits"][split])
    if negative_keep_fraction is None:
        negative_keep_fraction = config["tiles"]["negative_keep_fraction"][split]
    source_dataset_fingerprint = (
        sha256_file(config["splits"][split])
        if Path(src).resolve() != Path(config["splits"][split]).resolve()
        else None
    )
    path = Path(path)
    if not path.is_file():
        return False
    try:
        dset = kwcoco.CocoDataset.coerce(str(path))
        dset.validate()
        info = next(
            item for item in dset.dataset.get("info", [])
            if item.get("name") == "kwcoco_detector_kit.data.tile"
        )
    except Exception:
        return False
    expected = _tile_expected(
        config, split, src, negative_keep_fraction, source_dataset_fingerprint,
    )
    if info.get("src") != expected["src"]:
        return False
    if info.get("source_manifest_sha256") != expected["source_manifest_sha256"]:
        return False
    actual_config = info.get("config", {})
    return (
        all(actual_config.get(key) == value for key, value in expected["config"].items())
        and all("tile_actual_scale_xy" in image for image in dset.images().objs)
    )


def build_candidates(config, splits=("train", "validation")):
    """Build the complete virtual safe-negative control plane."""
    require_kdk(config)
    from kwcoco_detector_kit.data.candidates import CandidateConfig, enumerate_candidates

    root = Path(config["paths"]["output_root"]) / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    policy = config["tiles"]
    for split in splits:
        dst = root / f"{split}_negative_candidates.json"
        candidate_config = CandidateConfig.cli(argv=False, data={
            "src": config["splits"][split], "dst": str(dst),
            "category_names": ",".join(config["category_names"]),
            "tile_size": policy["size"],
            "oversize_factor": policy["oversize_factor"],
            "source_scales": ",".join(map(str, policy["source_scales"])),
            "stride_frac": policy["stride_frac"],
            "min_keep_fraction": policy["min_keep_fraction"],
            "min_gt_area_frac": policy["min_gt_area_frac"],
            "negative_safety_margin": policy["negative_safety_margin"],
            "min_source_scale_long_side": policy["min_source_scale_long_side"],
            "source_dataset_fingerprint": sha256_file(config["splits"][split]),
        })
        enumerate_candidates(candidate_config)
        print(dst)


def _run_tile(config, split, dst, *, src=None, negative_keep_fraction=None):
    require_kdk(config)
    from kwcoco_detector_kit.data.tile import TileConfig, run
    policy = config["tiles"]
    if negative_keep_fraction is None:
        negative_keep_fraction = policy["negative_keep_fraction"][split]
    source_dataset_fingerprint = (
        sha256_file(config["splits"][split])
        if src is not None and Path(src).resolve() != Path(config["splits"][split]).resolve()
        else None
    )
    tile_config = TileConfig.cli(argv=False, data={
        "src": str(src or config["splits"][split]), "dst": str(dst),
        "mode": "multiscale", "category_names": ",".join(config["category_names"]),
        "tile_size": policy["size"], "oversize_factor": policy["oversize_factor"],
        "source_scales": ",".join(map(str, policy["source_scales"])),
        "stride_frac": policy["stride_frac"],
        "min_keep_fraction": policy["min_keep_fraction"],
        "min_gt_area_frac": policy["min_gt_area_frac"],
        "min_source_scale_long_side": policy["min_source_scale_long_side"],
        "negative_safety_margin": policy["negative_safety_margin"],
        "negative_keep_fraction": negative_keep_fraction,
        "seed": config["round0"]["seed"],
        "jpeg_quality": policy["jpeg_quality"],
        "cache_dpath": config["paths"]["cache"], "keep_negative": True,
        "source_dataset_fingerprint": source_dataset_fingerprint,
    })
    run(tile_config)


def _source_subset(config, split, dst, *, n_positive, n_negative, seed):
    """Freeze a deterministic real-data source subset with absolute assets."""
    import kwcoco
    import numpy as np

    dset = kwcoco.CocoDataset.coerce(config["splits"][split])
    target_cids = {
        cat["id"] for cat in dset.dataset.get("categories", [])
        if cat["name"] in config["category_names"]
    }
    positive_gids = {
        ann["image_id"] for ann in dset.annots().objs
        if ann.get("category_id") in target_cids
    }
    all_gids = set(dset.images())
    negative_gids = all_gids - positive_gids
    rng = np.random.RandomState(int(seed))

    def _pick(pool, count):
        pool = np.array(sorted(pool), dtype=np.int64)
        if count > len(pool):
            raise ValueError(f"requested {count} images from pool of {len(pool)}")
        return sorted(map(int, rng.choice(pool, size=count, replace=False)))

    chosen = _pick(positive_gids, int(n_positive)) + _pick(negative_gids, int(n_negative))
    subset = dset.subset(chosen)
    subset.reroot(absolute=True)
    subset.dataset.setdefault("info", []).append({
        "name": "rfdetr_seg_v1 real-data smoke source subset",
        "source_manifest": str(Path(config["splits"][split]).resolve()),
        "selection_seed": int(seed),
        "num_positive_images": int(n_positive),
        "num_negative_images": int(n_negative),
    })
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subset.fpath = str(dst)
    subset.dump()
    subset.validate()
    return dst


def _generate_rfdetr_config(config, train, vali, workdir, *, smoke=False):
    require_kdk(config)
    from kwcoco_detector_kit.trainers._registry import get_trainer

    policy = config["rfdetr"]
    return get_trainer("rfdetr").generate_config(
        train, vali, workdir, variant=policy["variant"],
        input_hw=tuple(policy["input_hw"]), train_policy="fixed",
        num_classes=len(config["category_names"]),
        batch_size=1 if smoke else policy["batch_size_per_gpu"],
        val_batch_size=1 if smoke else policy["validation_batch_size_per_gpu"],
        num_epochs=1 if smoke else policy["epochs"], lr=policy["lr"],
        backbone_lr=policy["backbone_lr"], use_amp=policy["use_amp"],
        channels="r|g|b", scale_tier="2XL",
        num_gpus=1 if smoke else policy["num_gpus"],
        data_format="kwcoco", extra={
            "category_names": config["category_names"],
            "grad_accum_steps": 1 if smoke else policy["grad_accum_steps"],
            "num_workers": 2 if smoke else policy["num_workers"],
            "checkpoint_interval": 1,
        },
    )


def prepare_smoke(config, force=False):
    """Build a small real-data corpus through every pre-GPU data boundary."""
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.merge import MergeConfig, run as merge_run

    root = Path(config["paths"]["output_root"]) / "smoke"
    policy = config["smoke"]
    train_src = root / "source_train.kwcoco.zip"
    vali_src = root / "source_validation.kwcoco.zip"
    train_tiles = root / "train_all_tiles.kwcoco.zip"
    vali_tiles = root / "train_validation_tiles.kwcoco.zip"
    train_round = root / "train_round0.kwcoco.zip"
    outputs = [train_src, vali_src, train_tiles, vali_tiles, train_round]
    manifest_path = root / "smoke_manifest.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    source_manifest_sha256 = {
        split: sha256_file(path) for split, path in config["splits"].items()
    }
    reusable = (
        not force
        and prior.get("config_fingerprint") == config_fingerprint(config)
        and prior.get("source_manifest_sha256") == source_manifest_sha256
        and all(path.is_file() for path in outputs)
        and _tile_artifact_matches(
            config, "train", train_tiles, src=train_src,
            negative_keep_fraction=1.0,
        )
        and _tile_artifact_matches(
            config, "validation", vali_tiles, src=vali_src,
            negative_keep_fraction=1.0,
        )
    )
    if reusable:
        for path in outputs:
            kwcoco.CocoDataset.coerce(str(path)).validate()
        print(f"reuse valid real-data smoke corpus: {root}")
    else:
        _source_subset(
            config, "train", train_src,
            n_positive=policy["train_positive_images"],
            n_negative=policy["train_negative_images"], seed=policy["seed"],
        )
        _source_subset(
            config, "validation", vali_src,
            n_positive=policy["validation_positive_images"],
            n_negative=policy["validation_negative_images"], seed=policy["seed"] + 1,
        )
        # The smoke corpus is intentionally exhaustive so the simulator can
        # be checked against every legal window on this real-data subset.
        _run_tile(
            config, "train", train_tiles, src=train_src,
            negative_keep_fraction=1.0,
        )
        _run_tile(
            config, "validation", vali_tiles, src=vali_src,
            negative_keep_fraction=1.0,
        )
        merge_cfg = MergeConfig.cli(argv=False, data={
            "pos_kwcoco": str(train_tiles), "neg_kwcoco": str(train_tiles),
            "dst": str(train_round),
            "category_names": ",".join(config["category_names"]),
            "neg_over_pos": config["round0"]["negative_over_positive"],
            "seed": config["round0"]["seed"], "round_index": 0,
        })
        merge_run(merge_cfg)
        for path in outputs:
            kwcoco.CocoDataset.coerce(str(path)).validate()

    workdir = root / "workdir"
    cfg_path = _generate_rfdetr_config(
        config, train_round, vali_tiles, workdir, smoke=True,
    )
    summary = {
        "schema_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "source_manifest_sha256": source_manifest_sha256,
        "artifacts": {},
        "rfdetr_config": str(cfg_path),
    }
    for name, path in {
        "source_train": train_src, "source_validation": vali_src,
        "train_tiles": train_tiles, "train_validation_tiles": vali_tiles,
        "train_round0": train_round,
    }.items():
        dset = kwcoco.CocoDataset.coerce(str(path))
        summary["artifacts"][name] = {
            "path": str(path), "sha256": sha256_file(path),
            "num_images": dset.n_images, "num_annotations": dset.n_annots,
        }
    atomic_json_dump(summary, manifest_path)
    print(manifest_path)
    return root


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
        if not force and _tile_artifact_matches(config, split, dst):
            print(f"reuse valid pool: {dst}")
        else:
            _run_tile(config, split, dst)
            kwcoco.CocoDataset.coerce(str(dst)).validate()
    # Composition is cheap and recreating it prevents an older selection
    # policy from surviving after a valid tile pool is rebuilt.
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

    root = Path(config["paths"]["output_root"])
    train = root / "rounds" / "round0" / "train.kwcoco.zip"
    vali = root / "pools" / "train_validation_tiles.kwcoco.zip"
    for path in [train, vali]:
        if not path.exists():
            raise FileNotFoundError(f"build-pools must produce {path}")
        kwcoco.CocoDataset.coerce(str(path)).validate()
    workdir = root / "rounds" / "round0" / "workdir"
    cfg_path = _generate_rfdetr_config(config, train, vali, workdir)
    policy = config["rfdetr"]
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
        "input_verification": root / "input_verification.json",
        "census": root / "census.json",
        "tile_policy_simulation": root / "tile_policy_simulation.json",
        "train_candidate_index": root / "candidates" / "train_negative_candidates.json",
        "validation_candidate_index": root / "candidates" / "validation_negative_candidates.json",
        "train_tiles": root / "pools" / "train_all_tiles.kwcoco.zip",
        "train_validation_tiles": root / "pools" / "train_validation_tiles.kwcoco.zip",
        "round0_manifest": root / "rounds" / "round0" / "train.kwcoco.zip",
        "rfdetr_config": root / "rounds" / "round0" / "workdir" / "generated_configs" / "rfdetr_train.json",
        "round0_command": root / "rounds" / "round0" / "workdir" / "ROUND0_COMMAND.txt",
        "smoke_manifest": root / "smoke" / "smoke_manifest.json",
    }
    for name, path in required.items():
        valid = path.is_file()
        if valid and name in {"input_verification", "census", "tile_policy_simulation"}:
            try:
                valid = json.loads(path.read_text()).get("config_fingerprint") == config_fingerprint(config)
            except Exception:
                valid = False
        elif valid and name == "train_tiles":
            valid = _tile_artifact_matches(config, "train", path)
        elif valid and name == "train_validation_tiles":
            valid = _tile_artifact_matches(config, "validation", path)
        elif valid and name == "smoke_manifest":
            try:
                smoke = json.loads(path.read_text())
                valid = (
                    smoke.get("config_fingerprint") == config_fingerprint(config)
                    and smoke.get("source_manifest_sha256") == {
                        split: sha256_file(src) for split, src in config["splits"].items()
                    }
                )
            except Exception:
                valid = False
        if valid and ".kwcoco." in path.name:
            try:
                kwcoco.CocoDataset.coerce(str(path)).validate()
            except Exception:
                valid = False
        print(f"{'complete' if valid else 'not-ready':10s} {name:24s} {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "status", "verify-inputs", "census", "prepare-smoke",
            "simulate-policy", "build-candidates", "build-pools", "prepare",
        ],
    )
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--hash-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "status":
        status(config)
    elif args.command == "verify-inputs":
        verify_inputs(config)
    elif args.command == "census":
        census(config, hash_assets=args.hash_assets)
    elif args.command == "simulate-policy":
        simulate_policy(config)
    elif args.command == "prepare-smoke":
        prepare_smoke(config, force=args.force)
    elif args.command == "build-candidates":
        build_candidates(config)
    elif args.command == "build-pools":
        build_pools(config, force=args.force)
    elif args.command == "prepare":
        prepare(config)


if __name__ == "__main__":
    main()
