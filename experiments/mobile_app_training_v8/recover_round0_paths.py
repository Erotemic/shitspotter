#!/usr/bin/env python3
"""
One-shot recovery for v8 runs where round0 mining completed under the
pre-fix kit (commits <= 6450952). That mining wrote relative file_name
fields in hard_negs.kwcoco.zip that resolve against the WRONG directory
when round1's merge step reads them, causing the round1 training to die
on the first batch with FileNotFoundError.

This script rewrites file_name fields in hard_negs.kwcoco.zip to
absolute paths under the original neg-tile pool. The mining scores are
preserved -- only the path strings change.

Usage (inside the docker image, or with kwcoco installed locally):

    python3 experiments/mobile_app_training_v8/recover_round0_paths.py \\
        --v8_root /data/joncrall/kcd/v8 \\
        --cells "pico:416 n:640"

By default it patches every round*/hard_negs.kwcoco.zip under each cell
and the corresponding rounds/round*/train_round.kwcoco.zip (the round's
merged training kwcoco -- which round_loop will regenerate next run, but
patching here lets a user inspect the corrected file before re-running).

The fix lives upstream in kit commits b0db63c (merge.py) and a later
commit to mine.py; future v8/v9/v10 runs don't need this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _patch_kwcoco_zip_paths(fpath: Path, orig_pool_dpath: Path, *,
                             dry_run: bool = False) -> tuple[int, int]:
    """Open `fpath`, rewrite any relative file_name fields to abs paths
    under `orig_pool_dpath`, dump back to disk. Returns (n_rewritten,
    n_kept_abs).
    """
    import kwcoco
    dset = kwcoco.CocoDataset.coerce(str(fpath))
    n_rewritten = 0
    n_kept_abs = 0
    for img in dset.images().objs:
        fn = img.get("file_name", "")
        if not fn:
            continue
        if fn.startswith("/"):
            n_kept_abs += 1
            continue
        new = str((orig_pool_dpath / fn).resolve())
        img["file_name"] = new
        n_rewritten += 1
    print(
        f"  {fpath}: {n_rewritten} rewritten to abs, "
        f"{n_kept_abs} already abs"
    )
    if not dry_run and n_rewritten:
        # Force a fresh _build_index before dump -- modifying img dicts
        # in-place can invalidate kwcoco's image-name lookup.
        try:
            dset._build_index()
        except Exception:
            pass
        dset.dump()
    return n_rewritten, n_kept_abs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v8_root", default="/data/joncrall/kcd/v8",
                        help="path to v8 workspace root")
    parser.add_argument("--cells", default="pico:416 n:640",
                        help="space-separated <variant>:<size> entries")
    parser.add_argument("--orig_pool_dpath",
                        default="/data/joncrall/kcd/v8/data",
                        help="bundle dir of the original neg-tile pool "
                             "(where train_tiles_assets/ lives)")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args(argv)

    v8_root = Path(args.v8_root)
    orig_pool = Path(args.orig_pool_dpath)
    if not orig_pool.exists():
        print(f"ERROR: --orig_pool_dpath {orig_pool} does not exist",
              file=sys.stderr)
        return 2
    if not (orig_pool / "train_tiles_assets").exists():
        print(
            f"WARN: {orig_pool}/train_tiles_assets/ does not exist -- the "
            f"rewritten paths may still be wrong. Continuing anyway.",
            file=sys.stderr,
        )

    cells = [c for c in args.cells.split() if c]
    total_rewritten = 0
    for cell in cells:
        variant_short, size = cell.split(":")
        variant = f"deimv2_{variant_short}"
        workdir_tag = f"{variant}_{size}x{size}"
        cell_root = v8_root / workdir_tag
        if not cell_root.exists():
            print(f"[{cell}] skip: {cell_root} does not exist")
            continue
        print(f"[{cell}] scanning {cell_root}")
        for hard_neg in sorted(cell_root.glob("rounds/round*/hard_negs.kwcoco.zip")):
            n, _ = _patch_kwcoco_zip_paths(
                hard_neg, orig_pool, dry_run=args.dry_run,
            )
            total_rewritten += n
        for train_round in sorted(cell_root.glob("rounds/round*/train_round.kwcoco.zip")):
            n, _ = _patch_kwcoco_zip_paths(
                train_round, orig_pool, dry_run=args.dry_run,
            )
            total_rewritten += n

    print(
        f"\nTotal file_name rewrites: {total_rewritten} "
        f"({'DRY RUN' if args.dry_run else 'WRITTEN TO DISK'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
