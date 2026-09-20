#!/usr/bin/env python
"""Cheap status view for an RF-DETR campaign run.

Reads only small receipts/logs.  It deliberately does not reopen the large
KWCoco train/validation manifests.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path

from driver import load_config


def _nonempty(row, key):
    value = row.get(key)
    return value not in (None, "")


def _read_metrics(path):
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _last_value(row, key, default=None):
    value = row.get(key) if row else None
    return default if value in (None, "") else value


def _docker_tail(image, n=120):
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", f"ancestor={image}", "--format", "{{.ID}}"],
            check=True, text=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, []
    ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if len(ids) != 1:
        return None, []
    cid = ids[0]
    try:
        proc = subprocess.run(
            ["docker", "logs", "--timestamps", "--tail", str(n), cid],
            check=False, text=True, capture_output=True,
        )
    except OSError:
        return cid, []
    return cid, (proc.stdout + proc.stderr).splitlines()


def inspect_once(config, round_index=0, use_docker=True):
    root = Path(config["paths"]["output_root"])
    policy = config["rfdetr"]
    run_name = str(policy["run_name"])
    workdir = root / "rounds" / f"round{round_index}" / "runs" / run_name
    metrics_path = workdir / "metrics.csv"
    pool_manifest_path = root / "pools" / "pool_manifest.json"

    train_tiles = validation_tiles = None
    if pool_manifest_path.is_file():
        pool = json.loads(pool_manifest_path.read_text())
        train = pool.get("train", {})
        vali = pool.get("validation", {})
        if isinstance(train.get("positive_tiles"), int) and isinstance(train.get("negative_tiles"), int):
            train_tiles = train["positive_tiles"] + train["negative_tiles"]
        if isinstance(vali.get("positive_tiles"), int) and isinstance(vali.get("negative_tiles"), int):
            validation_tiles = vali["positive_tiles"] + vali["negative_tiles"]

    num_gpus = int(policy["num_gpus"])
    train_global_batch = int(policy["batch_size_per_gpu"]) * num_gpus
    eval_global_batch = int(policy["validation_batch_size_per_gpu"]) * num_gpus
    train_steps = math.ceil(train_tiles / train_global_batch) if train_tiles else None
    validation_batches = math.ceil(validation_tiles / eval_global_batch) if validation_tiles else None

    rows = _read_metrics(metrics_path)
    latest = rows[-1] if rows else {}
    latest_val = next(
        (row for row in reversed(rows) if any(
            value not in (None, "") and key.startswith("val/")
            for key, value in row.items()
        )),
        None,
    )
    epoch = int(float(_last_value(latest, "epoch", -1))) if latest else None
    step = int(float(_last_value(latest, "step", -1))) if latest else None
    latest_val_epoch = (
        int(float(_last_value(latest_val, "epoch", -1)))
        if latest_val else None
    )

    # CSVLogger emits sparse training rows (normally every 50 optimizer steps).
    # Use step geometry as the primary signal so an old validation-start message
    # lingering in ``docker logs --tail`` cannot make an already-resumed training
    # epoch look like it is still validating.
    near_boundary = False
    if step is not None and train_steps:
        offset = (step + 1) % train_steps
        distance_to_boundary = min(offset, train_steps - offset)
        near_boundary = distance_to_boundary <= 55

    phase = "unknown"
    docker_id = None
    marker_lines = []
    validation_start_seen = False
    if use_docker:
        docker_id, log_lines = _docker_tail(str(policy["image"]))
        marker_lines = [
            line for line in log_lines
            if any(token in line.lower() for token in [
                "validation-loss", "validat", "map", "early stop",
                "checkpoint", "max_epochs", "finished",
            ])
        ]
        # RF-DETR emits this exact message from on_validation_epoch_start.  It is
        # corroborating evidence only; metrics/step position decide whether it is
        # still relevant to the current phase.
        validation_start_seen = any(
            "Skipping validation-loss computation" in line for line in log_lines
        )

    if step is not None and train_steps:
        if near_boundary and latest_val_epoch == epoch:
            phase = "validation_complete_or_checkpoint"
        elif near_boundary and validation_start_seen and latest_val_epoch != epoch:
            phase = "validation_or_metric_finalize"
        elif near_boundary and latest_val_epoch != epoch:
            phase = "likely_epoch_boundary_or_validation"
        else:
            phase = "training"
    elif validation_start_seen:
        phase = "validation_or_metric_finalize"

    print(f"run={run_name} round={round_index} workdir={workdir}")
    print(
        f"train: tiles={train_tiles} batch={policy['batch_size_per_gpu']}/gpu x {num_gpus} "
        f"= global {train_global_batch}; expected_steps/epoch~{train_steps}"
    )
    print(
        f"validation: tiles={validation_tiles} batch={policy['validation_batch_size_per_gpu']}/gpu "
        f"x {num_gpus} = global {eval_global_batch}; expected_batches~{validation_batches}"
    )
    print(f"metrics: {metrics_path} ({len(rows)} rows)")
    if latest:
        print(f"latest train-log row: epoch={epoch} step={step}")
    else:
        print("latest train-log row: none")
    if latest_val:
        val_keys = {
            key: value for key, value in latest_val.items()
            if value not in (None, "") and key.startswith("val/")
        }
        compact = {k: val_keys[k] for k in sorted(val_keys) if any(
            token in k for token in ["mAP_50_95", "mAP_50", "F1", "precision", "recall"]
        )}
        print(f"latest completed validation: epoch={latest_val_epoch} {compact}")
    else:
        print("latest completed validation: none yet")
    print(f"phase_guess={phase}")
    if docker_id:
        print(f"docker_container={docker_id}")
    if marker_lines:
        print("recent phase markers:")
        for line in marker_lines[-5:]:
            print(f"  {line}")
    return phase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--round-index", type=int, default=0)
    parser.add_argument("--no-docker", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=10.0)
    args = parser.parse_args()
    config = load_config(args.config)
    while True:
        inspect_once(config, round_index=args.round_index, use_docker=not args.no_docker)
        if not args.watch:
            break
        print("-" * 80, flush=True)
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
