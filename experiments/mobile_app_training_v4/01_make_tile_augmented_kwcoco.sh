#!/bin/bash
# Build the v4 tile-augmented training + validation kwcoco bundles.
#
# Inputs (read-only, on the DVC roots):
#   $V4_TRAIN_FPATH = train_imgs10671_b277c63d.kwcoco.zip   # v9 training split
#   $V4_VALI_FPATH  = vali_imgs1258_577e331c.kwcoco.zip
#
# Outputs (writable, under $V4_ROOT):
#   $V4_ROOT/data/train_tile_g${V4_TILE_GRID}.kwcoco.zip
#   $V4_ROOT/data/vali_tile_g${V4_TILE_GRID}.kwcoco.zip
#   $V4_ROOT/data/train_tile_g${V4_TILE_GRID}.simplified.kwcoco.zip
#   $V4_ROOT/data/vali_tile_g${V4_TILE_GRID}.simplified.kwcoco.zip
#   $V4_ROOT/data/train_tile_g${V4_TILE_GRID}.simplified.mscoco.json
#   $V4_ROOT/data/vali_tile_g${V4_TILE_GRID}.simplified.mscoco.json
#
# The simplified bundles use the same merge step as v9 so candidate eval
# numbers stay directly comparable to the canonical v9 metric.
#
# Re-running is cheap when nothing changes — every step is idempotent and
# checks for an existing output first. Set FORCE_TILE_REBUILD=1 to regenerate.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
# shellcheck source=experiments/mobile_app_training_v4/common.sh
source "$_v4_script_dpath/common.sh"
unset _v4_source

FORCE_TILE_REBUILD="${FORCE_TILE_REBUILD:-False}"
DATA_DPATH="$V4_ROOT/data"
mkdir -p "$DATA_DPATH"

# Tile-augmented bundles
TRAIN_TILE_FPATH="$DATA_DPATH/train_tile_g${V4_TILE_GRID}.kwcoco.zip"
VALI_TILE_FPATH="$DATA_DPATH/vali_tile_g${V4_TILE_GRID}.kwcoco.zip"

# Same after the v9 simplify merge
TRAIN_SIMPLIFIED_FPATH="$DATA_DPATH/train_tile_g${V4_TILE_GRID}.simplified.kwcoco.zip"
VALI_SIMPLIFIED_FPATH="$DATA_DPATH/vali_tile_g${V4_TILE_GRID}.simplified.kwcoco.zip"

# COCO json exports for DEIMv2 train.py
TRAIN_MSCOCO_FPATH="$DATA_DPATH/train_tile_g${V4_TILE_GRID}.simplified.mscoco.json"
VALI_MSCOCO_FPATH="$DATA_DPATH/vali_tile_g${V4_TILE_GRID}.simplified.mscoco.json"

echo "=== mobile_app_training_v4 / 01 tile-augmented kwcoco ==="
v4_print_env
printf '  %-32s %s\n' "TRAIN_TILE_FPATH"  "$TRAIN_TILE_FPATH"
printf '  %-32s %s\n' "VALI_TILE_FPATH"   "$VALI_TILE_FPATH"
printf '  %-32s %s\n' "TILE_GRID"         "$V4_TILE_GRID"
printf '  %-32s %s\n' "TILE_OVERLAP"      "$V4_TILE_OVERLAP"
printf '  %-32s %s\n' "TILE_OUTPUT_DIM"   "$V4_TILE_OUTPUT_DIM"
printf '  %-32s %s\n' "RESIZE_FULL_DIM"   "$V4_RESIZE_MAX_DIM"

echo
echo "=== Tile training split ==="
if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$TRAIN_TILE_FPATH" ]; then
    "$PYTHON_BIN" "$V4_DEV_DPATH/tile_kwcoco.py" \
        --src "$V4_TRAIN_FPATH" \
        --dst "$TRAIN_TILE_FPATH" \
        --full_dim "$V4_RESIZE_MAX_DIM" \
        --tile_grid "$V4_TILE_GRID" \
        --tile_overlap "$V4_TILE_OVERLAP" \
        --tile_output_dim "$V4_TILE_OUTPUT_DIM" \
        --keep_full "$V4_TILE_KEEP_FULL" \
        --output_ext "$V4_RESIZE_OUTPUT_EXT"
else
    echo "  reusing $TRAIN_TILE_FPATH"
fi

echo
echo "=== Tile validation split ==="
if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$VALI_TILE_FPATH" ]; then
    "$PYTHON_BIN" "$V4_DEV_DPATH/tile_kwcoco.py" \
        --src "$V4_VALI_FPATH" \
        --dst "$VALI_TILE_FPATH" \
        --full_dim "$V4_RESIZE_MAX_DIM" \
        --tile_grid "$V4_TILE_GRID" \
        --tile_overlap "$V4_TILE_OVERLAP" \
        --tile_output_dim "$V4_TILE_OUTPUT_DIM" \
        --keep_full "$V4_TILE_KEEP_FULL" \
        --output_ext "$V4_RESIZE_OUTPUT_EXT"
else
    echo "  reusing $VALI_TILE_FPATH"
fi

echo
echo "=== Simplify (v9-style cluster-level merge) ==="
# simplify_kwcoco currently has a hidden runtime dep on geowatch
# (`from geowatch.utils.util_kwimage import find_low_overlap_covering_boxes`).
# When geowatch isn't installed (e.g. lean smoke envs), fall back to a
# straight copy of the tile bundle so downstream stages still find a
# `.simplified.kwcoco.zip`. Set V4_FORCE_SIMPLIFY=1 to make the
# simplify failure fatal; the host-side prod env should always have
# geowatch.
#
# Per-call status is recorded into the build manifest so downstream
# steps (and the eligibility manifest) can tell whether each split
# was actually simplified or silently copied. simplify_status values:
#   simplified       - real simplify_kwcoco run succeeded
#   reused           - existing simplified file on disk, not regenerated
#   copied_fallback  - simplify failed, fell back to a straight cp
#   failed           - simplify failed AND V4_FORCE_SIMPLIFY=1 -> abort
TILE_BUILD_MANIFEST="$DATA_DPATH/tile_build_manifest.json"
declare -A V4_SIMPLIFY_STATUS=()
declare -A V4_SIMPLIFY_ERROR=()

_v4_simplify_or_copy() {
    local split="$1" src="$2" dst="$3"
    local err_log
    err_log=$(mktemp)
    if "$PYTHON_BIN" -m shitspotter.cli.simplify_kwcoco \
            --src "$src" --dst "$dst" \
            --minimum_instances "$V4_SIMPLIFY_MIN_INSTANCES" 2> >(tee "$err_log" >&2); then
        V4_SIMPLIFY_STATUS[$split]="simplified"
        rm -f "$err_log"
        return 0
    fi
    local err_msg
    err_msg=$(tail -3 "$err_log" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')
    rm -f "$err_log"
    if v4_is_truthy "${V4_FORCE_SIMPLIFY:-0}"; then
        echo "  ERROR: simplify failed and V4_FORCE_SIMPLIFY=1" >&2
        V4_SIMPLIFY_STATUS[$split]="failed"
        V4_SIMPLIFY_ERROR[$split]="$err_msg"
        return 1
    fi
    echo "  WARNING: simplify_kwcoco failed (likely missing geowatch dep);" >&2
    echo "           falling back to a straight copy. Pass V4_FORCE_SIMPLIFY=1" >&2
    echo "           to make this fatal." >&2
    cp "$src" "$dst"
    V4_SIMPLIFY_STATUS[$split]="copied_fallback"
    V4_SIMPLIFY_ERROR[$split]="$err_msg"
}

if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$TRAIN_SIMPLIFIED_FPATH" ]; then
    _v4_simplify_or_copy train "$TRAIN_TILE_FPATH" "$TRAIN_SIMPLIFIED_FPATH"
else
    echo "  reusing $TRAIN_SIMPLIFIED_FPATH"
    V4_SIMPLIFY_STATUS[train]="reused"
fi

if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$VALI_SIMPLIFIED_FPATH" ]; then
    _v4_simplify_or_copy vali "$VALI_TILE_FPATH" "$VALI_SIMPLIFIED_FPATH"
else
    echo "  reusing $VALI_SIMPLIFIED_FPATH"
    V4_SIMPLIFY_STATUS[vali]="reused"
fi

# Write the manifest. Stable JSON layout so the eligibility step
# (and tests) can read it back without inspecting filesystem state.
"$PYTHON_BIN" - <<PY > "$TILE_BUILD_MANIFEST"
import json, datetime
manifest = {
    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
    "v4_force_simplify": "${V4_FORCE_SIMPLIFY:-0}",
    "minimum_instances": int("$V4_SIMPLIFY_MIN_INSTANCES"),
    "tile_grid": int("$V4_TILE_GRID"),
    "tile_overlap": float("$V4_TILE_OVERLAP"),
    "tile_output_dim": int("$V4_TILE_OUTPUT_DIM"),
    "splits": {
        "train": {
            "source_kwcoco": "$V4_TRAIN_FPATH",
            "tile_kwcoco": "$TRAIN_TILE_FPATH",
            "simplified_kwcoco": "$TRAIN_SIMPLIFIED_FPATH",
            "simplify_status": "${V4_SIMPLIFY_STATUS[train]:-unknown}",
            "simplify_required": True,
            "simplify_error": "${V4_SIMPLIFY_ERROR[train]:-}",
        },
        "vali": {
            "source_kwcoco": "$V4_VALI_FPATH",
            "tile_kwcoco": "$VALI_TILE_FPATH",
            "simplified_kwcoco": "$VALI_SIMPLIFIED_FPATH",
            "simplify_status": "${V4_SIMPLIFY_STATUS[vali]:-unknown}",
            "simplify_required": True,
            "simplify_error": "${V4_SIMPLIFY_ERROR[vali]:-}",
        },
    },
}
print(json.dumps(manifest, indent=2))
PY

echo "  tile build manifest -> $TILE_BUILD_MANIFEST"
for split in train vali; do
    printf '    %-6s simplify_status=%s\n' "$split" "${V4_SIMPLIFY_STATUS[$split]}"
done

echo
echo "=== Export MSCOCO json for DEIMv2 train.py ==="
if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$TRAIN_MSCOCO_FPATH" ]; then
    "$PYTHON_BIN" - <<PY
from shitspotter.algo_foundation_v3.coco_adapter import _build_coco_export
_build_coco_export(
    src='$TRAIN_SIMPLIFIED_FPATH',
    dst='$TRAIN_MSCOCO_FPATH',
    category_name='poop',
    include_segmentations=False,
    category_id=0,
)
print('wrote $TRAIN_MSCOCO_FPATH')
PY
else
    echo "  reusing $TRAIN_MSCOCO_FPATH"
fi

if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$VALI_MSCOCO_FPATH" ]; then
    "$PYTHON_BIN" - <<PY
from shitspotter.algo_foundation_v3.coco_adapter import _build_coco_export
_build_coco_export(
    src='$VALI_SIMPLIFIED_FPATH',
    dst='$VALI_MSCOCO_FPATH',
    category_name='poop',
    include_segmentations=False,
    category_id=0,
)
print('wrote $VALI_MSCOCO_FPATH')
PY
else
    echo "  reusing $VALI_MSCOCO_FPATH"
fi

echo
echo "Done. Inputs ready for the v4 training scripts:"
printf '  %-32s %s\n' "train_mscoco" "$TRAIN_MSCOCO_FPATH"
printf '  %-32s %s\n' "vali_mscoco"  "$VALI_MSCOCO_FPATH"
