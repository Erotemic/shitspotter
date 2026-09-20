#!/usr/bin/env python
"""Reproducible local RF-DETR package/predict/review workflow.

This is intentionally a thin ShitSpotter campaign wrapper. RF-DETR runtime
isolation and GPU execution live in KDK's ``docker/rfdetr/kcd-rfdetr`` helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CONFIG_FPATH = HERE / "config.yaml"
EXPECTED_TRAIN_SHA256 = "b3b6246531e525493e653917609cf0194d5c5eca2881314aa2c53c88160650a4"


def parse_bool(text):
    if isinstance(text, bool):
        return text
    value = str(text).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {text!r}")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_campaign_config() -> dict:
    return yaml.safe_load(CONFIG_FPATH.read_text())


def resolve_kdk_repo(config: dict) -> Path:
    override = os.environ.get("KCD_REPO_DPATH")
    if override:
        return Path(override).expanduser().resolve()
    path = Path(config["paths"]["kdk_repo"]).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (HERE / path).resolve()


def resolve_paths(snapshot_name: str, args) -> dict[str, Path]:
    kdk_repo = resolve_kdk_repo(load_campaign_config())
    snapshot = Path(
        args.snapshot
        or Path(os.environ.get("SHITSPOTTER_LOCAL_SNAPSHOT_ROOT", "~/data/shitspotter_rfdetr")).expanduser()
        / snapshot_name
    ).expanduser().resolve()
    src = Path(
        args.src
        or os.environ.get(
            "SHITSPOTTER_LOCAL_TRAIN_KWCOCO",
            str(REPO_ROOT / "shitspotter_dvc" / "train.kwcoco.zip"),
        )
    ).expanduser().resolve()
    model_root = Path(
        os.environ.get("SHITSPOTTER_LOCAL_MODEL_ROOT", "~/data/shitspotter_models")
    ).expanduser().resolve()
    review_root = Path(
        os.environ.get("SHITSPOTTER_LOCAL_REVIEW_ROOT", "~/data/shitspotter_review")
    ).expanduser().resolve() / snapshot_name
    model = Path(args.model).expanduser().resolve() if args.model else (
        model_root / f"shitspotter-rfdetr-{snapshot_name}.zip"
    )
    pred = Path(args.pred).expanduser().resolve() if args.pred else (
        review_root / "train_predictions.kwcoco.zip"
    )
    review = Path(args.review).expanduser().resolve() if args.review else (
        review_root / "review"
    )
    smoke_src = pred.parent / "smoke4.kwcoco.zip"
    smoke_pred = pred.parent / "smoke4.pred.kwcoco.zip"
    return {
        "kdk_repo": kdk_repo,
        "kdk_runner": kdk_repo / "docker" / "rfdetr" / "kcd-rfdetr",
        "snapshot": snapshot,
        "src": src,
        "model": model,
        "pred": pred,
        "review": review,
        "review_root": review_root,
        "smoke_src": smoke_src,
        "smoke_pred": smoke_pred,
    }


def verify_sha256_manifest(snapshot: Path) -> None:
    manifest = snapshot / "SHA256SUMS"
    if not manifest.is_file():
        raise FileNotFoundError(f"snapshot is missing SHA256SUMS: {manifest}")
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        path = snapshot / rel
        if not path.is_file():
            raise FileNotFoundError(f"snapshot checksum target is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"snapshot checksum mismatch: {path}: {actual} != {expected}")


def verify_inputs(paths: dict[str, Path], *, allow_source_hash_mismatch: bool) -> None:
    snapshot = paths["snapshot"]
    src = paths["src"]
    runner = paths["kdk_runner"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"local snapshot does not exist: {snapshot}")
    verify_sha256_manifest(snapshot)
    checkpoint_names = [
        "checkpoint_best_total.pth",
        "checkpoint_best_ema.pth",
        "last.ckpt",
    ]
    if not any((snapshot / name).is_file() for name in checkpoint_names):
        raise FileNotFoundError(
            f"snapshot has no canonical RF-DETR checkpoint: {snapshot}"
        )
    if not src.is_file():
        raise FileNotFoundError(f"training source KWCoco does not exist: {src}")
    source_sha = sha256_file(src)
    if source_sha != EXPECTED_TRAIN_SHA256 and not allow_source_hash_mismatch:
        raise RuntimeError(
            "training KWCoco hash is not the frozen round-0 source; refusing to "
            "silently change the annotation-QA population. Pass "
            "--allow-source-hash-mismatch only after intentionally regenerating truth.\n"
            f"expected={EXPECTED_TRAIN_SHA256}\nactual={source_sha}\npath={src}"
        )
    if not runner.is_file():
        raise FileNotFoundError(f"KDK RF-DETR Docker runner is missing: {runner}")


def run(command: list[str], *, dry_run: bool = False, env=None) -> None:
    print("+ " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def kdk(paths: dict[str, Path], *argv: str, dry_run: bool = False) -> None:
    env = os.environ.copy()
    env.setdefault("KCD_RFDETR_GPU", "0")
    extra_mounts = []
    for key in ["snapshot", "src", "model", "review_root"]:
        path = paths.get(key)
        if path is None:
            continue
        path = Path(path)
        mount = path if path.is_dir() else path.parent
        extra_mounts.append(str(mount))
    existing = env.get("KCD_RFDETR_EXTRA_MOUNTS")
    if existing:
        extra_mounts.extend(line for line in existing.splitlines() if line)
    env["KCD_RFDETR_EXTRA_MOUNTS"] = "\n".join(dict.fromkeys(extra_mounts))
    command = [str(paths["kdk_runner"]), *map(str, argv)]
    run(command, dry_run=dry_run, env=env)


def package(paths, args) -> None:
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    argv = [
        "package-build",
        f"--workdir={paths['snapshot']}",
        "--trainer=rfdetr",
        "--dataset-slug=shitspotter",
        "--experiment-slug=rfdetr_seg_v1_v3",
        f"--run-id={args.snapshot_name}",
        f"--train-kwcoco={paths['src']}",
        f"--score-thresh={args.score_thresh}",
        f"--out={paths['model']}",
    ]
    kdk(paths, *argv, dry_run=args.dry_run)


def make_smoke_source(paths, args) -> None:
    paths["review_root"].mkdir(parents=True, exist_ok=True)
    code = r'''
import sys
import kwcoco
src, dst = sys.argv[1:]
dset = kwcoco.CocoDataset.coerce(src)
target_cids = {c["id"] for c in dset.dataset.get("categories", []) if c.get("name") == "poop"}
pos = sorted({a["image_id"] for a in dset.dataset.get("annotations", []) if a.get("category_id") in target_cids})
all_gids = sorted(dset.images())
neg = [gid for gid in all_gids if gid not in set(pos)]
gids = pos[:2] + neg[:2]
sub = dset.subset(gids)
sub.reroot(absolute=True)
sub.fpath = dst
sub.dump()
print(dst)
print("gids=", gids)
'''.strip()
    kdk(
        paths,
        "exec", "python", "-c", code,
        str(paths["src"]), str(paths["smoke_src"]),
        dry_run=args.dry_run,
    )


def predict(paths, args, *, smoke=False) -> None:
    src = paths["smoke_src"] if smoke else paths["src"]
    dst = paths["smoke_pred"] if smoke else paths["pred"]
    dst.parent.mkdir(parents=True, exist_ok=True)
    kdk(
        paths,
        "predict",
        f"--model={paths['model']}",
        f"--src={src}",
        f"--dst={dst}",
        "--device=cuda:0",
        f"--backend={args.backend}",
        "--windowed=true",
        f"--window={args.window}",
        f"--overlap={args.overlap}",
        f"--batch-size={args.batch_size}",
        f"--score-thresh={args.score_thresh}",
        f"--pipeline={str(args.pipeline).lower()}",
        f"--source-workers={args.source_workers}",
        f"--source-prefetch={args.source_prefetch}",
        f"--window-prefetch={args.window_prefetch}",
        f"--postprocess-workers={args.postprocess_workers}",
        f"--postprocess-inflight={args.postprocess_inflight}",
        dry_run=args.dry_run,
    )


def review(paths, args) -> None:
    semantics = load_campaign_config()["truth_semantics"]
    paths["review"].mkdir(parents=True, exist_ok=True)
    kdk(
        paths,
        "prediction-review",
        f"--true={paths['src']}",
        f"--pred={paths['pred']}",
        f"--dst-dpath={paths['review']}",
        f"--target-categories={','.join(semantics['target_categories'])}",
        f"--ignore-categories={','.join(semantics['ignore_categories'])}",
        f"--uncategorized-annotation-policy={semantics['uncategorized_annotation_policy']}",
        f"--default-non-target-policy={semantics['default_non_target_policy']}",
        f"--unclassified-category-policy={semantics['unclassified_category_policy']}",
        f"--min-score={args.review_min_score}",
        f"--top-n={args.review_top_n}",
        dry_run=args.dry_run,
    )


def status(paths) -> None:
    data = {key: str(value) for key, value in paths.items()}
    data["expected_train_sha256"] = EXPECTED_TRAIN_SHA256
    if paths["src"].is_file():
        data["actual_train_sha256"] = sha256_file(paths["src"])
    print(json.dumps(data, indent=2, sort_keys=True))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Package, GPU-predict, and truth-review one immutable RF-DETR snapshot."
    )
    parser.add_argument(
        "command",
        choices=["status", "build-image", "package", "smoke", "predict", "review", "all"],
    )
    parser.add_argument("--snapshot-name", default=os.environ.get("SHITSPOTTER_RFDETR_SNAPSHOT"))
    parser.add_argument("--snapshot")
    parser.add_argument("--src")
    parser.add_argument("--model")
    parser.add_argument("--pred")
    parser.add_argument("--review")
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx", "auto"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=768)
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--score-thresh", type=float, default=0.01)
    parser.add_argument(
        "--pipeline", type=parse_bool, default=True, metavar="BOOL",
        help="overlap source/window I/O and CPU postprocess with GPU inference",
    )
    parser.add_argument("--source-workers", type=int, default=2)
    parser.add_argument("--source-prefetch", type=int, default=2)
    parser.add_argument("--window-prefetch", type=int, default=2)
    parser.add_argument("--postprocess-workers", type=int, default=1)
    parser.add_argument("--postprocess-inflight", type=int, default=2)
    parser.add_argument("--review-min-score", type=float, default=0.50)
    parser.add_argument("--review-top-n", type=int, default=20000)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--allow-source-hash-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "build-image":
        kdk_repo = resolve_kdk_repo(load_campaign_config())
        paths = {
            "kdk_runner": kdk_repo / "docker" / "rfdetr" / "kcd-rfdetr",
        }
        kdk(paths, "build", dry_run=args.dry_run)
        return
    if not args.snapshot_name:
        parser.error("--snapshot-name is required (or set SHITSPOTTER_RFDETR_SNAPSHOT)")
    paths = resolve_paths(args.snapshot_name, args)
    if args.command == "status":
        status(paths)
        return

    verify_inputs(
        paths,
        allow_source_hash_mismatch=args.allow_source_hash_mismatch,
    )
    if args.command == "package":
        package(paths, args)
    elif args.command == "smoke":
        if not paths["model"].is_file() and not args.dry_run:
            raise FileNotFoundError(f"package first: {paths['model']}")
        make_smoke_source(paths, args)
        predict(paths, args, smoke=True)
    elif args.command == "predict":
        if not paths["model"].is_file() and not args.dry_run:
            raise FileNotFoundError(f"package first: {paths['model']}")
        predict(paths, args)
    elif args.command == "review":
        if not paths["pred"].is_file() and not args.dry_run:
            raise FileNotFoundError(f"prediction corpus is missing: {paths['pred']}")
        review(paths, args)
    elif args.command == "all":
        package(paths, args)
        if not args.skip_smoke:
            make_smoke_source(paths, args)
            predict(paths, args, smoke=True)
        predict(paths, args)
        review(paths, args)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
