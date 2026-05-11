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
_v4_simplify_or_copy() {
    local src="$1" dst="$2"
    if "$PYTHON_BIN" -m shitspotter.cli.simplify_kwcoco \
            --src "$src" --dst "$dst" \
            --minimum_instances "$V4_SIMPLIFY_MIN_INSTANCES"; then
        return 0
    fi
    if v4_is_truthy "${V4_FORCE_SIMPLIFY:-0}"; then
        echo "  ERROR: simplify failed and V4_FORCE_SIMPLIFY=1" >&2
        return 1
    fi
    echo "  WARNING: simplify_kwcoco failed (likely missing geowatch dep);" >&2
    echo "           falling back to a straight copy. Pass V4_FORCE_SIMPLIFY=1" >&2
    echo "           to make this fatal." >&2
    cp "$src" "$dst"
}

if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$TRAIN_SIMPLIFIED_FPATH" ]; then
    _v4_simplify_or_copy "$TRAIN_TILE_FPATH" "$TRAIN_SIMPLIFIED_FPATH"
else
    echo "  reusing $TRAIN_SIMPLIFIED_FPATH"
fi

if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$VALI_SIMPLIFIED_FPATH" ]; then
    _v4_simplify_or_copy "$VALI_TILE_FPATH" "$VALI_SIMPLIFIED_FPATH"
else
    echo "  reusing $VALI_SIMPLIFIED_FPATH"
fi

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
