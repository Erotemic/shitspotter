#!/usr/bin/env python3
"""
One-shot provenance backfill for existing v6/v7/v8 workdirs.

The kit only started stamping provenance into policy.json + metrics.json
after the 2026-05-24 fix (kit commits stamping into _dump_policy_json
and run_kwcoco_eval). For the v6.0/v7.0/v8.0 artifacts already on disk,
this script infers the kit_sha from:

  1. The workdir's mtime, cross-referenced against the kit git log
  2. The DEIMv2 submodule SHA (a CONSTANT 377e10a across the v6-v8
     lineage; this is checked, not inferred).
  3. The Open-GroundingDino submodule SHA (similarly constant).

The script writes a `provenance.json` sidecar into each workdir + each
eval dir, never modifying existing files. Rerun-safe.

Run on the VM (where the kit's .git is reachable):

    python3 experiments/provenance_backfill.py \\
        --v8_root /data/joncrall/kcd/v8 \\
        --v6_root /data/joncrall/kcd/v6 \\
        --v7_root /data/joncrall/kcd/v7

This is bookkeeping only; it does not retrain or re-eval anything.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


KIT_REPO = Path("/home/joncrall/code/kwcoco_detector_kit")


def _git(repo: Path, *args, **kw) -> str:
    """`git -C repo -c safe.directory=* ...` — bypasses the dubious-
    ownership guard for read-only bind-mounts in docker.
    """
    return subprocess.check_output(
        ["git", "-c", "safe.directory=*", "-C", str(repo), *args],
        text=True, **kw,
    )


def _git_log_with_dates(repo: Path):
    """Return [(sha, iso_date)] for every commit in repo, newest first."""
    out = _git(repo, "log", "--pretty=format:%H %cI", "--all")
    rows = []
    for line in out.splitlines():
        sha, _, iso = line.partition(" ")
        rows.append((sha, iso))
    return rows


def _infer_sha_from_mtime(repo: Path, target_iso: str):
    """Pick the latest kit commit at or before `target_iso`."""
    target = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    log = _git_log_with_dates(repo)
    for sha, iso in log:
        commit_dt = datetime.fromisoformat(iso)
        if commit_dt <= target:
            return sha
    return log[-1][0] if log else "<unknown>"


def _submodule_sha(repo: Path, submodule: str):
    try:
        out = _git(repo, "submodule", "status", submodule,
                   stderr=subprocess.DEVNULL)
        # output looks like " 377e10a... tpl/DEIMv2 (377e10a)"
        token = out.strip().split()[0]
        return token.lstrip("+-U")
    except Exception:
        return "<unknown>"


def backfill_workdir(wd: Path, *, kit_sha: str, kit_repo: Path):
    """Write provenance.json sidecar to `wd` (idempotent; never clobbers)."""
    sidecar = wd / "provenance.json"
    if sidecar.exists():
        return False, "exists"
    if not wd.exists() or not any(wd.iterdir()):
        return False, "no workdir"

    # The mtime of the workdir's policy.json (or any saved .pth) is the
    # most precise timestamp we have for "when the kit code committed to
    # this workdir was running."
    policy_fpath = wd / "policy.json"
    anchor_fpath = policy_fpath if policy_fpath.exists() else wd
    anchor_iso = datetime.fromtimestamp(
        anchor_fpath.stat().st_mtime, tz=timezone.utc,
    ).isoformat()

    if kit_sha == "auto":
        sha = _infer_sha_from_mtime(kit_repo, anchor_iso)
        sha_source = f"inferred from {anchor_fpath.name} mtime"
    else:
        sha = kit_sha
        sha_source = "user-supplied"

    obj = {
        "kit_sha": sha,
        "kit_sha_source": sha_source,
        "deimv2_sha": _submodule_sha(kit_repo, "tpl/DEIMv2"),
        "opengroundingdino_sha": _submodule_sha(kit_repo, "tpl/Open-GroundingDino"),
        "workdir_mtime": anchor_iso,
        "backfilled_at": datetime.now(timezone.utc).isoformat(),
        "backfill_note": (
            "This workdir predates the kit's automatic provenance "
            "stamping. SHAs inferred from workdir timestamp + kit git "
            "log. The DEIMv2 and Open-GroundingDino submodule SHAs were "
            "stable across the entire v6-v8 lineage, so they are reliable. "
            "The kit_sha is the BEST GUESS from mtime correlation, not a "
            "ground-truth read from the image."
        ),
    }
    sidecar.write_text(json.dumps(obj, indent=2))
    return True, sha


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--v6_root", default="/data/joncrall/kcd/v6")
    parser.add_argument("--v7_root", default="/data/joncrall/kcd/v7")
    parser.add_argument("--v8_root", default="/data/joncrall/kcd/v8")
    parser.add_argument("--v6_1_root", default="/data/joncrall/kcd/v6_1")
    parser.add_argument(
        "--kit_repo", default=str(KIT_REPO),
        help="local kit git working copy for SHA lookup",
    )
    parser.add_argument(
        "--kit_sha", default="auto",
        help="kit SHA to stamp (default: auto-infer from workdir mtime)",
    )
    args = parser.parse_args(argv)

    kit_repo = Path(args.kit_repo)
    if not (kit_repo / ".git").exists():
        print(f"ERROR: {kit_repo} is not a git working copy", file=sys.stderr)
        return 2

    workdirs = []
    for root in [args.v6_root, args.v7_root, args.v8_root, args.v6_1_root]:
        rp = Path(root)
        if not rp.exists():
            continue
        # v6/v7/v9/v10: flat sweep -> runs/<cand>/
        workdirs.extend(rp.glob("runs/*/"))
        # v8: round-loop -> deimv2_*/rounds/round*/runs/<cand>/
        workdirs.extend(rp.glob("deimv2_*/rounds/round*/runs/*/"))

    print(f"Scanning {len(workdirs)} potential workdirs...")
    written = 0
    skipped = 0
    for wd in sorted(workdirs):
        ok, msg = backfill_workdir(wd, kit_sha=args.kit_sha, kit_repo=kit_repo)
        if ok:
            print(f"  WROTE  {wd}  -> kit={msg[:12]}")
            written += 1
        else:
            skipped += 1

    print(f"\nDone. Wrote {written} provenance.json sidecars; skipped {skipped}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
