#!/usr/bin/env python
"""Rsync one immutable aiq-gpu model snapshot and verify its checksums."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256s(dst: Path) -> None:
    sums = dst / "SHA256SUMS"
    if not sums.is_file():
        raise FileNotFoundError(f"snapshot is missing checksum manifest: {sums}")
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        path = dst / rel
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {path}: {actual} != {expected}")
    snapshot_fpath = dst / "SNAPSHOT.json"
    if snapshot_fpath.is_file():
        snapshot = json.loads(snapshot_fpath.read_text())
        checkpoint = dst / snapshot["checkpoint_source_name"]
        actual = sha256_file(checkpoint)
        if actual != snapshot["checkpoint_sha256"]:
            raise RuntimeError("checkpoint digest disagrees with SNAPSHOT.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", default="jon.crall@aiq-gpu")
    parser.add_argument(
        "--snapshot-name",
        help="derive both remote and local paths from one immutable snapshot name",
    )
    parser.add_argument(
        "--remote-run",
        default=os.environ.get(
            "SHITSPOTTER_RFDETR_REMOTE_RUN",
            "/data/users/jon.crall/shitspotter_rfdetr_v1/rounds/round0/runs/v3",
        ),
    )
    parser.add_argument(
        "--remote-snapshot",
        help="absolute immutable snapshot directory on aiq-gpu",
    )
    parser.add_argument("--dst", help="local snapshot directory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    remote_snapshot = args.remote_snapshot
    dst_arg = args.dst
    if args.snapshot_name:
        remote_snapshot = remote_snapshot or (
            args.remote_run.rstrip("/") + "/snapshots/" + args.snapshot_name
        )
        dst_arg = dst_arg or str(
            Path("~/data/shitspotter_rfdetr").expanduser() / args.snapshot_name
        )
    if not remote_snapshot or not dst_arg:
        parser.error("provide --snapshot-name, or both --remote-snapshot and --dst")

    dst = Path(dst_arg).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync", "-a", "-P", "--protect-args",
        *( ["--dry-run"] if args.dry_run else [] ),
        f"{args.remote}:{remote_snapshot.rstrip('/')}/",
        f"{dst}/",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)
    if not args.dry_run:
        verify_sha256s(dst)
        print(f"verified snapshot: {dst}")


if __name__ == "__main__":
    main()
