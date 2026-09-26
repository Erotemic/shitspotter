#!/usr/bin/env python
"""Idempotent corrected-truth RF-DETR overnight orchestrator.

This intentionally starts at the artifact-derived campaign layer. Truth gathering,
split regeneration, and legacy cache migration should already be complete because
those operations define the inputs this script validates and consumes.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import driver


HERE = Path(__file__).resolve().parent


def _stamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _say(message):
    print(f"[{_stamp()}] {message}", flush=True)


def _load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _source_hashes(config, splits):
    return {split: driver.sha256_file(config["splits"][split]) for split in splits}


def _input_verification_current(config):
    path = Path(config["paths"]["output_root"]) / "input_verification.json"
    doc = _load_json(path)
    if not doc or doc.get("config_fingerprint") != driver.config_fingerprint(config):
        return False
    expected = _source_hashes(config, config["splits"])
    return all(
        doc.get("splits", {}).get(split, {}).get("manifest_sha256") == digest
        for split, digest in expected.items()
    )


def _census_current(config):
    path = Path(config["paths"]["output_root"]) / "census.json"
    doc = _load_json(path)
    if not doc or doc.get("config_fingerprint") != driver.config_fingerprint(config):
        return False
    expected = _source_hashes(config, config["splits"])
    return all(
        doc.get("splits", {}).get(split, {}).get("manifest_sha256") == digest
        for split, digest in expected.items()
    )


def _simulation_current(config):
    path = Path(config["paths"]["output_root"]) / "tile_policy_simulation.json"
    doc = _load_json(path)
    if not doc:
        return False
    if doc.get("simulation_policy_fingerprint") != driver.simulation_policy_fingerprint(config):
        return False
    return doc.get("source_manifest_sha256") == _source_hashes(
        config, ["train", "validation"]
    )


def _candidates_current(config):
    try:
        for split in ["train", "validation"]:
            driver._load_valid_candidate_index(config, split)
    except Exception:
        return False
    return True


def _pools_current(config):
    try:
        return bool(driver._pool_manifest_matches(config, details=False))
    except Exception:
        return False


def _run_driver(config_path, command, *, extra=()):
    argv = [
        sys.executable,
        str(HERE / "driver.py"),
        command,
        f"--config={Path(config_path).resolve()}",
        *extra,
    ]
    _say("RUN " + " ".join(map(str, argv)))
    subprocess.run(argv, check=True)


def _stage(name, current, action):
    if current():
        _say(f"SKIP {name}: valid durable artifact already exists")
        return
    _say(f"START {name}")
    action()
    if not current():
        raise RuntimeError(f"{name} completed without producing a valid durable artifact")
    _say(f"DONE {name}")


def _docker_state(name):
    proc = subprocess.run(
        ["docker", "inspect", "--format", "{{json .State}}", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if proc.returncode:
        return None
    return json.loads(proc.stdout)


def _safe_container_name(run_name, fingerprint):
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(run_name)).strip("-.")
    token = token or "round0"
    return f"shitspotter-rfdetr-{token}-{fingerprint[:10]}"


def _training_receipt_current(config, workdir):
    receipt = _load_json(workdir / "TRAINING_COMPLETE.json")
    if not receipt:
        return False
    if receipt.get("config_fingerprint") != driver.config_fingerprint(config):
        return False
    checkpoint = receipt.get("checkpoint")
    return bool(checkpoint and Path(checkpoint).is_file())


def _find_checkpoint(workdir):
    for name in ["checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last.ckpt"]:
        path = workdir / name
        if path.is_file():
            return path
    return None


def _follow_container(name):
    _say(f"FOLLOW docker logs for {name}")
    subprocess.run(["docker", "logs", "--follow", "--tail=100", name], check=False)
    state = _docker_state(name)
    if state is None:
        raise RuntimeError(f"training container disappeared: {name}")
    return state


def _train(config):
    workdir = driver._rfdetr_workdir(config, round_index=0)
    if _training_receipt_current(config, workdir):
        _say(f"SKIP training: {workdir / 'TRAINING_COMPLETE.json'} is valid")
        return

    cfg_path = workdir / "generated_configs" / "rfdetr_train.json"
    launcher = cfg_path.parent / "launch_rfdetr.py"
    if not cfg_path.is_file() or not launcher.is_file():
        raise FileNotFoundError("prepare must produce RF-DETR config and launcher")

    policy = config["rfdetr"]
    fingerprint = driver.config_fingerprint(config)
    name = _safe_container_name(policy.get("run_name", "round0"), fingerprint)
    state = _docker_state(name)

    if state is not None and state.get("Running"):
        _say(f"training container is already running: {name}")
        state = _follow_container(name)

    if state is not None and not state.get("Running"):
        exit_code = int(state.get("ExitCode", -1))
        if exit_code == 0:
            checkpoint = _find_checkpoint(workdir)
            if checkpoint is None:
                raise RuntimeError(
                    f"container {name} exited 0 but no checkpoint exists under {workdir}"
                )
            driver.atomic_json_dump({
                "schema_version": 1,
                "config_fingerprint": fingerprint,
                "container_name": name,
                "checkpoint": str(checkpoint.resolve()),
            }, workdir / "TRAINING_COMPLETE.json")
            _say(f"DONE training: {checkpoint}")
            return
        _say(f"previous training container exited {exit_code}; removing before resume")
        subprocess.run(["docker", "rm", name], check=True)
        state = None

    if state is None:
        kdk = Path(config["paths"]["kdk_repo"]).resolve()
        resume = workdir / "last.ckpt"
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "--gpus", "all",
            "--ipc=host",
            "--shm-size=64g",
            "-v", "/data:/data",
            "-v", f"{kdk}:{kdk}",
            "-w", str(kdk),
            str(policy["image"]),
            "python", "-m", "torch.distributed.run",
            f"--nproc_per_node={int(policy['num_gpus'])}",
            str(launcher),
            "--config", str(cfg_path),
        ]
        if resume.is_file():
            cmd += ["--resume", str(resume)]
            _say(f"RESUME training from {resume}")
        else:
            _say("START fresh training")
        _say("RUN " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        state = _follow_container(name)

    exit_code = int(state.get("ExitCode", -1))
    if exit_code != 0:
        raise RuntimeError(
            f"training container {name} exited with code {exit_code}; "
            "rerun this script to resume from last.ckpt if one exists"
        )
    checkpoint = _find_checkpoint(workdir)
    if checkpoint is None:
        raise RuntimeError(f"training completed but no checkpoint exists under {workdir}")
    driver.atomic_json_dump({
        "schema_version": 1,
        "config_fingerprint": fingerprint,
        "container_name": name,
        "checkpoint": str(checkpoint.resolve()),
    }, workdir / "TRAINING_COMPLETE.json")
    _say(f"DONE training: {checkpoint}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument(
        "--no-train", action="store_true",
        help="build/validate all CPU-side artifacts but do not start RF-DETR",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = driver.load_config(config_path)
    root = Path(config["paths"]["output_root"])
    root.mkdir(parents=True, exist_ok=True)

    _say(f"overnight campaign root: {root}")
    _say(f"config: {config_path}")

    _stage(
        "verify-inputs",
        lambda: _input_verification_current(config),
        lambda: _run_driver(config_path, "verify-inputs"),
    )
    _stage(
        "census",
        lambda: _census_current(config),
        lambda: _run_driver(config_path, "census"),
    )
    _stage(
        "simulate-policy",
        lambda: _simulation_current(config),
        lambda: _run_driver(config_path, "simulate-policy"),
    )
    _stage(
        "build-candidates",
        lambda: _candidates_current(config),
        lambda: _run_driver(config_path, "build-candidates"),
    )
    _stage(
        "build-pools",
        lambda: _pools_current(config),
        lambda: _run_driver(config_path, "build-pools"),
    )

    # prepare is cheap/idempotent after the RF-DETR export receipt exists and
    # regenerates the exact launcher/config used below.
    _say("START prepare")
    _run_driver(config_path, "prepare")
    _say("DONE prepare")

    if args.no_train:
        _say("STOP --no-train requested")
        return
    _train(config)

    driver.atomic_json_dump({
        "schema_version": 1,
        "config_fingerprint": driver.config_fingerprint(config),
        "completed_at": _stamp(),
    }, root / "OVERNIGHT_COMPLETE.json")
    _say(f"ALL DONE: {root / 'OVERNIGHT_COMPLETE.json'}")


if __name__ == "__main__":
    main()
