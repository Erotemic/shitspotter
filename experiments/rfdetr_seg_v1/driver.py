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
import shlex
import tempfile
from collections import Counter
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
_SHA256_CACHE = {}


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
    fpath = Path(fpath).resolve()
    stat = fpath.stat()
    cache_key = (str(fpath), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    hasher = hashlib.sha256()
    with open(fpath, "rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    _SHA256_CACHE[cache_key] = digest
    return digest


def config_fingerprint(config):
    """Fingerprint the fully resolved campaign recipe used by local artifacts."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def simulation_policy_fingerprint(config):
    """Fingerprint only inputs that can change the tile-policy simulation."""
    payload = {
        "category_names": config["category_names"],
        "splits": {
            key: config["splits"][key]
            for key in ["train", "validation"]
        },
        "tiles": config["tiles"],
        "negative_over_positive": {
            "train": config["round0"]["negative_over_positive"],
            "validation": config["validation"]["negative_over_positive"],
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _artifact_stat_signature(path):
    """Cheap identity check for immutable generated artifacts."""
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "num_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _artifact_receipt(path, dset):
    """Record the result of an expensive validation at the build boundary."""
    receipt = _artifact_stat_signature(path)
    receipt.update({
        "num_images": int(dset.n_images),
        "num_annotations": int(dset.n_annots),
        "validated": True,
    })
    return receipt


def _artifact_stat_matches(path, receipt):
    """Check an immutable artifact without reopening its KWCoco payload."""
    path = Path(path)
    if not path.is_file() or not isinstance(receipt, dict):
        return False
    try:
        current = _artifact_stat_signature(path)
    except OSError:
        return False
    return all(current.get(key) == receipt.get(key) for key in [
        "path", "num_bytes", "mtime_ns",
    ])


def _pool_paths(config):
    root = Path(config["paths"]["output_root"])
    pools = root / "pools"
    return {
        "train_positive_tiles": pools / "train_positive_tiles.kwcoco.zip",
        "train_negative_tiles": pools / "train_negative_tiles.kwcoco.zip",
        "validation_positive_tiles": pools / "validation_positive_tiles.kwcoco.zip",
        "validation_negative_tiles": pools / "validation_negative_tiles.kwcoco.zip",
        "train_validation_tiles": pools / "train_validation_tiles.kwcoco.zip",
        "round0_manifest": root / "rounds" / "round0" / "train.kwcoco.zip",
        "pool_manifest": pools / "pool_manifest.json",
    }


def _pool_manifest_matches(config, *, details=False):
    """Check the durable pool receipt, optionally revalidating KWCoco files.

    ``build-pools`` is the expensive validation boundary. Normal ``status``
    and ``prepare`` calls trust that immutable result and avoid reopening the
    large KWCoco manifests. Schema-3 receipts add cheap size/mtime checks;
    schema-2 receipts are accepted for already-built pools. ``details=True``
    is the explicit audit path that re-runs full KWCoco validation.
    """
    paths = _pool_paths(config)
    manifest_path = paths["pool_manifest"]
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if manifest.get("schema_version") not in {2, 3}:
        return False

    expected_source_hashes = {
        split: sha256_file(config["splits"][split])
        for split in ["train", "validation"]
    }
    if manifest.get("source_manifest_sha256") != expected_source_hashes:
        return False

    train = manifest.get("train") or {}
    validation = manifest.get("validation") or {}
    for recorded, policy in [
        (train, config["round0"]),
        (validation, config["validation"]),
    ]:
        if recorded.get("negative_strategy") != policy["negative_strategy"]:
            return False
        if recorded.get("negative_seed") != policy["seed"]:
            return False
        positive = recorded.get("positive_tiles")
        negative = recorded.get("negative_tiles")
        if not isinstance(positive, int) or positive <= 0:
            return False
        expected_negative = round(float(policy["negative_over_positive"]) * positive)
        if negative != expected_negative:
            return False

    if Path(train.get("round_manifest", "")).resolve() != paths["round0_manifest"].resolve():
        return False
    if Path(validation.get("manifest", "")).resolve() != paths["train_validation_tiles"].resolve():
        return False

    artifact_keys = [
        "train_positive_tiles",
        "train_negative_tiles",
        "validation_positive_tiles",
        "validation_negative_tiles",
        "train_validation_tiles",
        "round0_manifest",
    ]
    if not all(paths[key].is_file() for key in artifact_keys):
        return False

    if manifest.get("schema_version") >= 3:
        receipts = manifest.get("artifacts") or {}
        if not all(
            _artifact_stat_matches(paths[key], receipts.get(key))
            for key in artifact_keys
        ):
            return False

    if details:
        import kwcoco
        try:
            dsets = {}
            for key in artifact_keys:
                dset = kwcoco.CocoDataset.coerce(str(paths[key]))
                dset.validate()
                dsets[key] = dset

            # Inspect producer metadata on the already-loaded datasets rather
            # than calling helpers that would deserialize/validate them again.
            positive_src = (
                Path(config["paths"]["output_root"])
                / "pools" / "train_positive_sources.kwcoco.zip"
            )
            positive_info = next(
                item for item in dsets["train_positive_tiles"].dataset.get("info", [])
                if item.get("name") == "kwcoco_detector_kit.data.tile"
            )
            positive_expected = _tile_expected(
                config, "train", positive_src, 0.0,
                sha256_file(config["splits"]["train"]), keep_negative=False,
            )
            if positive_info.get("src") != positive_expected["src"]:
                return False
            if positive_info.get("source_manifest_sha256") != positive_expected["source_manifest_sha256"]:
                return False
            actual_positive_config = positive_info.get("config", {})
            if not all(
                actual_positive_config.get(key) == value
                for key, value in positive_expected["config"].items()
            ):
                return False
            if not all(
                "tile_actual_scale_xy" in image
                for image in dsets["train_positive_tiles"].images().objs
            ):
                return False

            _, train_index = _load_valid_candidate_index(config, "train")
            negative_info = next(
                item for item in dsets["train_negative_tiles"].dataset.get("info", [])
                if item.get("name") == "rfdetr_seg_v1 selected virtual negatives"
            )
            negative_expected = {
                "candidate_content_digest": train_index["candidate_content_digest"],
                "candidate_policy_fingerprint": train_index["policy_fingerprint"],
                "selection_strategy": str(train["negative_strategy"]),
                "selection_seed": int(train["negative_seed"]),
                "selection_budget": int(train["negative_tiles"]),
                "jpeg_quality": int(config["tiles"]["jpeg_quality"]),
            }
            if not all(
                negative_info.get(key) == value
                for key, value in negative_expected.items()
            ):
                return False
            train_neg_dset = dsets["train_negative_tiles"]
            if train_neg_dset.n_images != int(train["negative_tiles"]):
                return False
            if not all(
                image.get("tile_role") == "negative"
                for image in train_neg_dset.images().objs
            ):
                return False
        except Exception:
            return False
    return True


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
        "schema_version": 2,
        "config_fingerprint": config_fingerprint(config),
        "simulation_policy_fingerprint": simulation_policy_fingerprint(config),
        "source_manifest_sha256": {
            split: sha256_file(config["splits"][split])
            for split in ["train", "validation"]
        },
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
            for name in ["positive", "negative", "ignore"]
        }
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
        train_neg = min(
            train_counts["negative"]["total"],
            round(float(config["round0"]["negative_over_positive"]) * train_counts["positive"]["total"]),
        )
        vali_ratio = float(config["validation"]["negative_over_positive"])
        vali_neg = min(
            vali_counts["negative"]["total"],
            round(vali_ratio * vali_counts["positive"]["total"]),
        )
        materialized = (
            train_counts["positive"]["total"] + train_neg
            + vali_counts["positive"]["total"] + vali_neg
        )
        report["materialization_plan"] = {
            "train": {
                "positive": train_counts["positive"]["total"],
                "negative": train_neg,
                "negative_over_positive": float(config["round0"]["negative_over_positive"]),
            },
            "validation": {
                "positive": vali_counts["positive"]["total"],
                "negative": vali_neg,
                "negative_over_positive": vali_ratio,
            },
        }
        report["storage_estimate"] = {
            "basis_num_smoke_jpegs": len(encoded),
            "basis_average_jpeg_bytes": avg_bytes,
            "materialized_tiles": materialized,
            "estimated_bytes": materialized * avg_bytes,
            "note": "linear estimate from the real-data smoke cache; measure the full pool before training",
        }
    dst = Path(config["paths"]["output_root"]) / "tile_policy_simulation.json"
    atomic_json_dump(report, dst)
    print(dst)
    return dst


def _tile_expected(config, split, src, negative_keep_fraction, source_dataset_fingerprint,
                   *, keep_negative=True):
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
            "keep_negative": bool(keep_negative),
            "negative_keep_fraction": negative_keep_fraction,
            "seed": config["round0"]["seed"],
        },
    }


def _tile_artifact_matches(config, split, path, *, src=None, negative_keep_fraction=None,
                           keep_negative=True):
    """Return whether a tile manifest is valid and belongs to this recipe."""
    import kwcoco

    src = str(src or config["splits"][split])
    if negative_keep_fraction is None:
        negative_keep_fraction = 1.0 if keep_negative else 0.0
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
        keep_negative=keep_negative,
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
        dst = root / f"{split}_negative_candidates"
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


def _run_tile(config, split, dst, *, src=None, keep_negative=True,
              negative_keep_fraction=None):
    require_kdk(config)
    from kwcoco_detector_kit.data.tile import TileConfig, run
    policy = config["tiles"]
    if negative_keep_fraction is None:
        negative_keep_fraction = 1.0 if keep_negative else 0.0
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
        "cache_dpath": config["paths"]["cache"],
        "keep_negative": bool(keep_negative),
        "source_dataset_fingerprint": source_dataset_fingerprint,
    })
    run(tile_config)


def _target_positive_gids(config, split, dset=None):
    import kwcoco
    dset = dset or kwcoco.CocoDataset.coerce(config["splits"][split])
    target_cids = {
        cat["id"] for cat in dset.dataset.get("categories", [])
        if cat["name"] in config["category_names"]
    }
    return sorted({
        ann["image_id"] for ann in dset.annots().objs
        if ann.get("category_id") in target_cids
        and (ann.get("bbox") is not None or ann.get("segmentation") is not None)
    })


def _write_positive_source_subset(config, split, dst):
    """Freeze all target-positive sources so the positive pass skips empty images."""
    import kwcoco

    dset = kwcoco.CocoDataset.coerce(config["splits"][split])
    gids = _target_positive_gids(config, split, dset=dset)
    dst = Path(dst)
    expected = {
        "source_manifest": str(Path(config["splits"][split]).resolve()),
        "source_manifest_sha256": sha256_file(config["splits"][split]),
        "category_names": list(config["category_names"]),
        "num_source_images": len(gids),
    }
    if dst.is_file():
        try:
            prior = kwcoco.CocoDataset.coerce(str(dst))
            info = next(
                item for item in prior.dataset.get("info", [])
                if item.get("name") == "rfdetr_seg_v1 target-positive source subset"
            )
            if (
                all(info.get(key) == value for key, value in expected.items())
                and prior.n_images == len(gids)
                and sorted(prior.images()) == gids
            ):
                return dst
        except Exception:
            pass

    subset = dset.subset(gids)
    subset.reroot(absolute=True)
    subset.dataset.setdefault("info", []).append({
        "name": "rfdetr_seg_v1 target-positive source subset",
        **expected,
    })
    dst.parent.mkdir(parents=True, exist_ok=True)
    subset.fpath = str(dst)
    subset.dump()
    subset.validate()
    return dst


def _source_subset(config, split, dst, *, n_positive, n_negative, seed):
    """Freeze a deterministic real-data source subset with absolute assets."""
    import kwcoco
    import numpy as np

    dset = kwcoco.CocoDataset.coerce(config["splits"][split])
    positive_gids = set(_target_positive_gids(config, split, dset=dset))
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


def _candidate_index_expected(config, split):
    policy = config["tiles"]
    return {
        "source_kwcoco": str(Path(config["splits"][split]).resolve()),
        "source_dataset_fingerprint": sha256_file(config["splits"][split]),
        "policy": {
            "category_names": sorted(config["category_names"]),
            "tile_size": int(policy["size"]),
            "oversize_factor": float(policy["oversize_factor"]),
            "source_scales": [
                [f"s{int(round(float(scale) * 10)):02d}", float(scale)]
                for scale in policy["source_scales"]
            ],
            "stride_frac": float(policy["stride_frac"]),
            "min_keep_fraction": float(policy["min_keep_fraction"]),
            "min_gt_area_frac": float(policy["min_gt_area_frac"]),
            "negative_safety_margin": int(policy["negative_safety_margin"]),
            "min_source_scale_long_side": int(policy["min_source_scale_long_side"]),
        },
    }


def _load_valid_candidate_index(config, split):
    require_kdk(config)
    from kwcoco_detector_kit.data.candidates import load_candidate_index

    path = Path(config["paths"]["output_root"]) / "candidates" / f"{split}_negative_candidates"
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"build-candidates must produce {path}")
    index = load_candidate_index(path)
    expected = _candidate_index_expected(config, split)
    if index.get("source_kwcoco") != expected["source_kwcoco"]:
        raise RuntimeError(f"{split} candidate index source manifest mismatch")
    if index.get("source_dataset_fingerprint") != expected["source_dataset_fingerprint"]:
        raise RuntimeError(f"{split} candidate index dataset fingerprint mismatch")
    if index.get("policy") != expected["policy"]:
        raise RuntimeError(f"{split} candidate index tile-policy mismatch; rebuild candidates")
    return path, index


def _selected_negative_pool_matches(config, split, dst, *, index, budget, seed,
                                    strategy):
    import kwcoco

    dst = Path(dst)
    if not dst.is_file():
        return False
    try:
        dset = kwcoco.CocoDataset.coerce(str(dst))
        dset.validate()
        info = next(
            item for item in dset.dataset.get("info", [])
            if item.get("name") == "rfdetr_seg_v1 selected virtual negatives"
        )
    except Exception:
        return False
    expected = {
        "candidate_content_digest": index["candidate_content_digest"],
        "candidate_policy_fingerprint": index["policy_fingerprint"],
        "selection_strategy": str(strategy),
        "selection_seed": int(seed),
        "selection_budget": int(budget),
        "jpeg_quality": int(config["tiles"]["jpeg_quality"]),
    }
    return (
        all(info.get(key) == value for key, value in expected.items())
        and dset.n_images == int(budget)
        and all(img.get("tile_role") == "negative" for img in dset.images().objs)
    )


def _materialize_selected_negatives(config, split, dst, *, budget, seed, strategy,
                                    force=False):
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.candidates import (
        materialize_candidates,
        selected_candidate_record_factory,
    )

    index_path, index = _load_valid_candidate_index(config, split)
    budget = min(int(budget), int(index["num_candidates"]))
    if not force and _selected_negative_pool_matches(
        config, split, dst, index=index, budget=budget, seed=seed,
        strategy=strategy,
    ):
        print(f"reuse valid selected negative pool: {dst}")
        return Path(dst)

    factory = selected_candidate_record_factory(
        index, budget, seed=seed, strategy=strategy,
    )
    out = materialize_candidates(
        index, factory(), cache_dpath=config["paths"]["cache"],
        jpeg_quality=config["tiles"]["jpeg_quality"], batch_size=32,
    )
    out.dataset.setdefault("info", []).append({
        "name": "rfdetr_seg_v1 selected virtual negatives",
        "source_split": split,
        "candidate_index": str(index_path.resolve()),
        "candidate_content_digest": index["candidate_content_digest"],
        "candidate_policy_fingerprint": index["policy_fingerprint"],
        "selection_strategy": str(strategy),
        "selection_seed": int(seed),
        "selection_budget": int(budget),
        "jpeg_quality": int(config["tiles"]["jpeg_quality"]),
    })
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.fpath = str(dst)
    out.dump()
    kwcoco.CocoDataset.coerce(str(dst)).validate()
    print(f"materialized {budget} selected {split} negatives -> {dst}")
    return dst


def _rfdetr_workdir(config, round_index=0):
    """Return the active trainer workdir without clobbering prior run variants."""
    root = Path(config["paths"]["output_root"])
    base = root / "rounds" / f"round{int(round_index)}"
    run_name = config["rfdetr"].get("run_name")
    if not run_name:
        return base / "workdir"
    run_name = str(run_name)
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError(f"rfdetr.run_name must be a single path component, got {run_name!r}")
    return base / "runs" / run_name


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
            "lr_scheduler": policy.get("lr_scheduler", "step"),
            "lr_scheduler_kwargs": policy.get("lr_scheduler_kwargs", {}),
            "warmup_epochs": 0.0 if smoke else policy.get("warmup_epochs", 0.0),
            "best_model_metric": policy.get("best_model_metric", "map"),
            "early_stopping": False if smoke else policy.get("early_stopping", False),
            "early_stopping_patience": policy.get("early_stopping_patience", 10),
            "early_stopping_min_delta": policy.get("early_stopping_min_delta", 0.001),
            "early_stopping_use_ema": policy.get("early_stopping_use_ema", False),
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
            negative_keep_fraction=1.0, keep_negative=True,
        )
        and _tile_artifact_matches(
            config, "validation", vali_tiles, src=vali_src,
            negative_keep_fraction=1.0, keep_negative=True,
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
        _run_tile(
            config, "train", train_tiles, src=train_src,
            keep_negative=True, negative_keep_fraction=1.0,
        )
        _run_tile(
            config, "validation", vali_tiles, src=vali_src,
            keep_negative=True, negative_keep_fraction=1.0,
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
    """Materialize positives plus exact stratified negative quotas.

    The complete safe-negative universe remains virtual in ``candidates/``.
    This stage decodes positive source images only for the positive pass, then
    materializes exactly the configured train/validation negative budgets from
    the existing candidate indexes into the shared content-addressed cache.
    Full KWCoco validation happens here, once, and the resulting pool receipt
    lets later ``status`` / ``prepare`` calls use cheap immutable checks.
    """
    import kwcoco
    require_kdk(config)
    from kwcoco_detector_kit.data.merge import MergeConfig, run as merge_run

    root = Path(config["paths"]["output_root"])
    pools = root / "pools"
    pools.mkdir(parents=True, exist_ok=True)

    train_positive_src = pools / "train_positive_sources.kwcoco.zip"
    vali_positive_src = pools / "validation_positive_sources.kwcoco.zip"
    train_positive_tiles = pools / "train_positive_tiles.kwcoco.zip"
    vali_positive_tiles = pools / "validation_positive_tiles.kwcoco.zip"
    train_negative_tiles = pools / "train_negative_tiles.kwcoco.zip"
    vali_negative_tiles = pools / "validation_negative_tiles.kwcoco.zip"
    vali_tiles = pools / "train_validation_tiles.kwcoco.zip"
    round0 = root / "rounds" / "round0" / "train.kwcoco.zip"

    for split, source_subset, positive_tiles in [
        ("train", train_positive_src, train_positive_tiles),
        ("validation", vali_positive_src, vali_positive_tiles),
    ]:
        _write_positive_source_subset(config, split, source_subset)
        if not force and _tile_artifact_matches(
            config, split, positive_tiles, src=source_subset,
            negative_keep_fraction=0.0, keep_negative=False,
        ):
            print(f"reuse valid positive pool: {positive_tiles}")
        else:
            _run_tile(
                config, split, positive_tiles, src=source_subset,
                keep_negative=False, negative_keep_fraction=0.0,
            )
            kwcoco.CocoDataset.coerce(str(positive_tiles)).validate()

    train_pos = kwcoco.CocoDataset.coerce(str(train_positive_tiles))
    vali_pos = kwcoco.CocoDataset.coerce(str(vali_positive_tiles))
    n_train_pos = sum(
        img.get("tile_role") == "positive" for img in train_pos.images().objs
    )
    n_vali_pos = sum(
        img.get("tile_role") == "positive" for img in vali_pos.images().objs
    )
    if n_train_pos <= 0 or n_vali_pos <= 0:
        raise RuntimeError("positive tiling produced an empty training/validation pool")

    train_policy = config["round0"]
    vali_policy = config["validation"]
    train_budget = round(float(train_policy["negative_over_positive"]) * n_train_pos)
    vali_budget = round(float(vali_policy["negative_over_positive"]) * n_vali_pos)

    _materialize_selected_negatives(
        config, "train", train_negative_tiles,
        budget=train_budget,
        seed=train_policy["seed"],
        strategy=train_policy["negative_strategy"],
        force=force,
    )
    _materialize_selected_negatives(
        config, "validation", vali_negative_tiles,
        budget=vali_budget,
        seed=vali_policy["seed"],
        strategy=vali_policy["negative_strategy"],
        force=force,
    )
    train_neg = kwcoco.CocoDataset.coerce(str(train_negative_tiles))
    vali_neg = kwcoco.CocoDataset.coerce(str(vali_negative_tiles))

    round0.parent.mkdir(parents=True, exist_ok=True)
    train_merge = MergeConfig.cli(argv=False, data={
        "pos_kwcoco": str(train_positive_tiles),
        "neg_kwcoco": str(train_negative_tiles),
        "dst": str(round0),
        "category_names": ",".join(config["category_names"]),
        "neg_over_pos": 0,
        "seed": train_policy["seed"], "round_index": 0,
    })
    merge_run(train_merge)
    round0_dset = kwcoco.CocoDataset.coerce(str(round0))
    round0_dset.validate()

    vali_merge = MergeConfig.cli(argv=False, data={
        "pos_kwcoco": str(vali_positive_tiles),
        "neg_kwcoco": str(vali_negative_tiles),
        "dst": str(vali_tiles),
        "category_names": ",".join(config["category_names"]),
        "neg_over_pos": 0,
        "seed": vali_policy["seed"], "round_index": 0,
    })
    merge_run(vali_merge)
    vali_dset = kwcoco.CocoDataset.coerce(str(vali_tiles))
    vali_dset.validate()

    manifest = {
        "schema_version": 3,
        "source_manifest_sha256": {
            split: sha256_file(config["splits"][split])
            for split in ["train", "validation"]
        },
        "train": {
            "positive_tiles": n_train_pos,
            "negative_tiles": int(train_neg.n_images),
            "negative_strategy": train_policy["negative_strategy"],
            "negative_seed": train_policy["seed"],
            "round_manifest": str(round0),
        },
        "validation": {
            "positive_tiles": n_vali_pos,
            "negative_tiles": int(vali_neg.n_images),
            "negative_strategy": vali_policy["negative_strategy"],
            "negative_seed": vali_policy["seed"],
            "manifest": str(vali_tiles),
        },
        "artifacts": {
            "train_positive_tiles": _artifact_receipt(train_positive_tiles, train_pos),
            "train_negative_tiles": _artifact_receipt(train_negative_tiles, train_neg),
            "validation_positive_tiles": _artifact_receipt(vali_positive_tiles, vali_pos),
            "validation_negative_tiles": _artifact_receipt(vali_negative_tiles, vali_neg),
            "train_validation_tiles": _artifact_receipt(vali_tiles, vali_dset),
            "round0_manifest": _artifact_receipt(round0, round0_dset),
        },
    }
    atomic_json_dump(manifest, pools / "pool_manifest.json")
    print(pools / "pool_manifest.json")


def prepare(config, *, details=False):
    paths = _pool_paths(config)
    train = paths["round0_manifest"]
    vali = paths["train_validation_tiles"]
    if not _pool_manifest_matches(config, details=details):
        mode = "detailed validation" if details else "pool receipt"
        raise RuntimeError(
            f"build-pools artifacts failed {mode}; rerun build-pools or use "
            "status --details to diagnose"
        )
    workdir = _rfdetr_workdir(config, round_index=0)
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


def prepare_mining(config, round_index=0):
    """Write the four-rank score and rank-0 global-finalization recipe."""
    root = Path(config["paths"]["output_root"])
    candidate_index = root / "candidates" / "train_negative_candidates"
    if not (candidate_index / "manifest.json").is_file():
        raise FileNotFoundError(f"build-candidates must produce {candidate_index}")
    workdir = _rfdetr_workdir(config, round_index=round_index)
    mining_dir = root / "rounds" / f"round{round_index}" / "mining"
    mining_dir.mkdir(parents=True, exist_ok=True)
    policy = config["mining"]

    def quote(value):
        return shlex.quote(str(value))

    common = [
        "python", "-m", "kwcoco_detector_kit", "mine",
        "--candidate_index", str(candidate_index),
        "--cache_dpath", config["paths"]["cache"],
        "--workdir", str(workdir), "--trainer", "rfdetr",
        "--device", "cuda:0", "--num_shards", str(policy["num_shards"]),
        "--locality_chunk_size", str(policy["locality_chunk_size"]),
        "--batch_size", str(policy["batch_size_per_gpu"]),
        "--max_candidates", str(policy["max_candidates"]),
        "--candidate_seed", str(policy["candidate_seed"]),
        "--score_thresh", str(policy["score_thresh"]),
        "--max_hard_per_round", str(policy["max_hard_per_round"]),
        "--progress=1", "--allow_failures=0",
    ]
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "pids=()"]
    ledgers = []
    for rank in range(int(policy["num_shards"])):
        ledger = mining_dir / f"rank{rank}.mine_ledger.json"
        ledgers.append(ledger)
        rank_args = common + [
            "--shard_index", str(rank), "--ledger", str(ledger),
            "--dst", str(mining_dir / f"rank{rank}.unused.kwcoco.zip"),
        ]
        lines.append(
            f"CUDA_VISIBLE_DEVICES={rank} "
            + " ".join(map(quote, rank_args)) + " &"
        )
        lines.append("pids+=(\"$!\")")
    lines.extend([
        'for pid in "${pids[@]}"; do wait "$pid"; done',
        f"export KCD_ROUND={round_index + 1}",
    ])
    hard_negatives = mining_dir / "hard_negatives.kwcoco.zip"
    finalize_args = [
        "python", "-m", "kwcoco_detector_kit", "mine-finalize",
        "--candidate_index", str(candidate_index),
        "--ledgers", *map(str, ledgers),
        "--dst", str(hard_negatives),
        "--cache_dpath", config["paths"]["cache"],
        "--jpeg_quality", str(config["tiles"]["jpeg_quality"]),
        "--score_thresh", str(policy["score_thresh"]),
        "--max_hard_per_round", str(policy["max_hard_per_round"]),
        "--allow_failures=0",
    ]
    lines.append(" ".join(map(quote, finalize_args)))
    script = mining_dir / "RUN_MINING.sh"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    print(script)
    return script


def status(config, *, details=False):
    root = Path(config["paths"]["output_root"])
    pool_paths = _pool_paths(config)
    pool_valid = _pool_manifest_matches(config, details=details)
    required = {
        "input_verification": root / "input_verification.json",
        "census": root / "census.json",
        "tile_policy_simulation": root / "tile_policy_simulation.json",
        "train_candidate_index": root / "candidates" / "train_negative_candidates" / "manifest.json",
        "validation_candidate_index": root / "candidates" / "validation_negative_candidates" / "manifest.json",
        "train_positive_tiles": pool_paths["train_positive_tiles"],
        "train_negative_tiles": pool_paths["train_negative_tiles"],
        "train_validation_tiles": pool_paths["train_validation_tiles"],
        "pool_manifest": pool_paths["pool_manifest"],
        "round0_manifest": pool_paths["round0_manifest"],
        "rfdetr_config": _rfdetr_workdir(config, 0) / "generated_configs" / "rfdetr_train.json",
        "round0_command": _rfdetr_workdir(config, 0) / "ROUND0_COMMAND.txt",
        "smoke_manifest": root / "smoke" / "smoke_manifest.json",
    }
    pool_items = {
        "train_positive_tiles", "train_negative_tiles", "train_validation_tiles",
        "pool_manifest", "round0_manifest",
    }
    for name, path in required.items():
        valid = path.is_file()
        if valid and name in {"input_verification", "census"}:
            try:
                doc = json.loads(path.read_text())
                valid = all(
                    doc["splits"][split].get("manifest_sha256") == sha256_file(src)
                    for split, src in config["splits"].items()
                )
            except Exception:
                valid = False
        elif valid and name == "tile_policy_simulation":
            try:
                doc = json.loads(path.read_text())
                valid = doc.get("schema_version") == 2
                if valid and doc.get("simulation_policy_fingerprint") is not None:
                    valid = (
                        doc["simulation_policy_fingerprint"]
                        == simulation_policy_fingerprint(config)
                    )
                    source_hashes = doc.get("source_manifest_sha256") or {}
                    valid = valid and all(
                        source_hashes.get(split) == sha256_file(config["splits"][split])
                        for split in ["train", "validation"]
                    )
                elif valid:
                    # Schema-v2 artifacts written before run-specific training
                    # config was decoupled can still be reused when their
                    # actual simulated policy is unchanged.
                    train_plan = doc.get("materialization_plan", {}).get("train", {})
                    vali_plan = doc.get("materialization_plan", {}).get("validation", {})
                    valid = (
                        doc.get("tile_policy") == config["tiles"]
                        and train_plan.get("negative_over_positive")
                        == float(config["round0"]["negative_over_positive"])
                        and vali_plan.get("negative_over_positive")
                        == float(config["validation"]["negative_over_positive"])
                    )
            except Exception:
                valid = False
        elif valid and name == "train_candidate_index":
            try:
                _load_valid_candidate_index(config, "train")
                valid = True
            except Exception:
                valid = False
        elif valid and name == "validation_candidate_index":
            try:
                _load_valid_candidate_index(config, "validation")
                valid = True
            except Exception:
                valid = False
        elif name in pool_items:
            valid = pool_valid and path.is_file()
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
        print(f"{'complete' if valid else 'not-ready':10s} {name:24s} {path}")
    if not details:
        print("note: immutable KWCoco pools use their build receipt; pass --details for full validation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "status", "verify-inputs", "census", "prepare-smoke",
            "simulate-policy", "build-candidates", "build-pools", "prepare",
            "prepare-mining",
        ],
    )
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--hash-assets", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument(
        "--details", action="store_true",
        help="perform expensive full KWCoco validation for status/prepare",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "status":
        status(config, details=args.details)
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
        prepare(config, details=args.details)
    elif args.command == "prepare-mining":
        prepare_mining(config, round_index=args.round_index)


if __name__ == "__main__":
    main()
