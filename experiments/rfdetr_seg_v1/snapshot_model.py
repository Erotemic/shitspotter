#!/usr/bin/env python
"""Create an immutable, race-checked snapshot of a live RF-DETR checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


DEFAULT_ROOT = Path("/data/users/jon.crall/shitspotter_rfdetr_v1")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_checkpoint(src: Path, dst: Path) -> None:
    """Prefer CoW/reflink, fall back to shutil without changing the source."""
    try:
        subprocess.run(
            ["cp", "--reflink=auto", "--preserve=mode,timestamps", str(src), str(dst)],
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        shutil.copy2(src, dst)


def copy_stable_file(src: Path, dst: Path) -> str:
    """Copy a small live metadata file only if its bytes stay unchanged."""
    before = sha256_file(src)
    shutil.copy2(src, dst)
    copied = sha256_file(dst)
    after = sha256_file(src)
    if not (before == copied == after):
        raise RuntimeError(
            f"metadata changed while snapshotting: {src}; no snapshot was published"
        )
    return copied


def snapshot_model(
    run_dpath: Path, name: str, checkpoint: str, expect_sha256: str | None = None
) -> Path:
    run_dpath = run_dpath.expanduser().resolve()
    src = run_dpath / checkpoint
    if not src.is_file():
        raise FileNotFoundError(src)
    snapshots = run_dpath / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    final = snapshots / name
    if final.exists():
        raise FileExistsError(
            f"snapshot already exists: {final}. Use a new immutable snapshot name."
        )

    # Hash before and after the copy. This detects both atomic replacement of
    # a live best checkpoint and in-place mutation while the snapshot is made.
    source_sha_before = sha256_file(src)
    if expect_sha256 is not None:
        expected = str(expect_sha256).strip().lower()
        if source_sha_before.lower() != expected:
            raise RuntimeError(
                "live checkpoint digest does not match --expect-sha256; refusing to "
                f"publish snapshot {name!r}: {source_sha_before} != {expected}"
            )
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}.tmp-", dir=snapshots))
    try:
        copied = tmp / src.name
        copy_checkpoint(src, copied)
        copied_sha = sha256_file(copied)
        source_sha_after = sha256_file(src)
        if not (source_sha_before == copied_sha == source_sha_after):
            raise RuntimeError(
                "live checkpoint changed while it was being snapshotted; no snapshot was "
                "published. Retry after the best-checkpoint write finishes."
            )

        copied_optional = []
        copied_metadata_sha256 = {}
        for rel in ["policy.json", "metrics.csv", "hparams.yaml"]:
            item = run_dpath / rel
            if item.is_file():
                copied_metadata_sha256[rel] = copy_stable_file(item, tmp / rel)
                copied_optional.append(rel)
        generated = run_dpath / "generated_configs"
        if generated.is_dir():
            shutil.copytree(generated, tmp / "generated_configs")
            copied_optional.append("generated_configs")
        datasets_json = run_dpath / "detector_prepared" / "datasets.json"
        if datasets_json.is_file():
            target = tmp / "detector_prepared" / "datasets.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(datasets_json, target)
            copied_optional.append("detector_prepared/datasets.json")

        snapshot_doc = {
            "schema": "shitspotter.rfdetr.snapshot.v1",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_run_dpath": str(run_dpath),
            "checkpoint_source_name": src.name,
            "checkpoint_sha256": copied_sha,
            "checkpoint_size_bytes": copied.stat().st_size,
            "expected_checkpoint_sha256": expect_sha256,
            "copied_metadata": copied_optional,
            "copied_metadata_sha256": copied_metadata_sha256,
        }
        (tmp / "SNAPSHOT.json").write_text(
            json.dumps(snapshot_doc, indent=2, sort_keys=True) + "\n"
        )
        checksum_files = [copied]
        checksum_files.extend(
            sorted(p for p in tmp.rglob("*") if p.is_file() and p.name not in {"SHA256SUMS", src.name})
        )
        with open(tmp / "SHA256SUMS", "w") as file:
            for path in checksum_files:
                file.write(f"{sha256_file(path)}  {path.relative_to(tmp)}\n")

        # The temporary directory is a sibling of the final snapshot, so this
        # publishes the already-copied checkpoint atomically without copying it
        # a second time.
        os.replace(tmp, final)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(final)
    print(f"checkpoint_sha256={source_sha_before}")
    return final


def main():
    root = Path(os.environ.get("SHITSPOTTER_RFDETR_ROOT", DEFAULT_ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dpath",
        default=str(root / "rounds" / "round0" / "runs" / "v3"),
        help="live RF-DETR workdir",
    )
    parser.add_argument("--name", required=True, help="immutable snapshot name, e.g. epoch2_best")
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best_ema.pth",
        help=(
            "checkpoint basename. During the active v3 EMA-only validation run, "
            "checkpoint_best_ema.pth is the live best; completed runs can use "
            "checkpoint_best_total.pth"
        ),
    )
    parser.add_argument(
        "--expect-sha256",
        help=(
            "optional known digest for the intended checkpoint (recommended when the "
            "snapshot name encodes an epoch); mismatches fail before copying"
        ),
    )
    args = parser.parse_args()
    snapshot_model(
        Path(args.run_dpath), args.name, args.checkpoint, args.expect_sha256
    )


if __name__ == "__main__":
    main()
