#!/bin/bash
# Build the v5 multi-scale tile bundles from the v9 train + vali splits.
#
# Outputs (under $V5_ROOT/data/):
#   train_tiles.kwcoco.zip    all tiles, both positive and negative
#   vali_tiles.kwcoco.zip
#   train_tiles_pos.kwcoco.zip    positive tiles only (training pool)
#   train_tiles_neg.kwcoco.zip    negative tiles only (mining pool)
#   tile_build_manifest.json      simplify-status-style sidecar

set -euo pipefail

_v5_source="${BASH_SOURCE[0]-}"
if [ -n "$_v5_source" ] && [ "$_v5_source" != "bash" ] && [ "$_v5_source" != "-bash" ]; then
    _v5_script_dpath="$(cd "$(dirname "$_v5_source")" && pwd)"
else
    _v5_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v5"
fi
# shellcheck source=experiments/mobile_app_training_v5/common.sh
source "$_v5_script_dpath/common.sh"
unset _v5_source _v5_script_dpath

FORCE_TILE_REBUILD="${V5_FORCE_TILE_REBUILD:-False}"
DATA_DPATH="$V5_ROOT/data"
mkdir -p "$DATA_DPATH"

TRAIN_TILES_FPATH="$DATA_DPATH/train_tiles.kwcoco.zip"
VALI_TILES_FPATH="$DATA_DPATH/vali_tiles.kwcoco.zip"
TRAIN_POS_FPATH="$DATA_DPATH/train_tiles_pos.kwcoco.zip"
TRAIN_NEG_FPATH="$DATA_DPATH/train_tiles_neg.kwcoco.zip"
VALI_POS_FPATH="$DATA_DPATH/vali_tiles_pos.kwcoco.zip"

echo "=== mobile_app_training_v5 / 01 multi-scale tile dataset ==="
v5_print_env
printf '  %-32s %s\n' "TRAIN_TILES_FPATH" "$TRAIN_TILES_FPATH"
printf '  %-32s %s\n' "VALI_TILES_FPATH"  "$VALI_TILES_FPATH"
printf '  %-32s %s\n' "TILE_SIZE"         "$V5_TILE_SIZE"
printf '  %-32s %s\n' "SOURCE_SCALES"     "$V5_SOURCE_SCALES"
printf '  %-32s %s\n' "STRIDE_FRAC"       "$V5_STRIDE_FRAC"
printf '  %-32s %s\n' "MIN_GT_AREA_FRAC"  "$V5_MIN_GT_AREA_FRAC"

# ---------------------------------------------------------------------------
# Train tiles — keep both positives and negatives so we can do hard-neg
# mining without re-extracting later.
# ---------------------------------------------------------------------------
echo
echo "=== Train tiles (multi-scale, fixed size, pos+neg) ==="
if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$TRAIN_TILES_FPATH" ]; then
    "$PYTHON_BIN" "$V5_DEV_DPATH/v5_tile.py" \
        --src "$V4_TRAIN_FPATH" \
        --dst "$TRAIN_TILES_FPATH" \
        --tile_size "$V5_TILE_SIZE" \
        --source_scales "$V5_SOURCE_SCALES" \
        --stride_frac "$V5_STRIDE_FRAC" \
        --min_gt_area_frac "$V5_MIN_GT_AREA_FRAC" \
        --min_kept_box_frac "$V5_MIN_KEPT_BOX_FRAC" \
        --min_source_scale_long_side "$V5_MIN_SOURCE_SCALE_LONG_SIDE" \
        --keep_negative True \
        --jpeg_quality "$V5_JPEG_QUALITY"
else
    echo "  reusing $TRAIN_TILES_FPATH"
fi

# ---------------------------------------------------------------------------
# Vali tiles — positives only suffices for validation loss tracking;
# generating negatives doesn't help because we don't mine on vali.
# ---------------------------------------------------------------------------
echo
echo "=== Vali tiles (multi-scale, fixed size, pos only) ==="
if v4_is_truthy "$FORCE_TILE_REBUILD" || [ ! -f "$VALI_TILES_FPATH" ]; then
    "$PYTHON_BIN" "$V5_DEV_DPATH/v5_tile.py" \
        --src "$V4_VALI_FPATH" \
        --dst "$VALI_TILES_FPATH" \
        --tile_size "$V5_TILE_SIZE" \
        --source_scales "$V5_SOURCE_SCALES" \
        --stride_frac "$V5_STRIDE_FRAC" \
        --min_gt_area_frac "$V5_MIN_GT_AREA_FRAC" \
        --min_kept_box_frac "$V5_MIN_KEPT_BOX_FRAC" \
        --min_source_scale_long_side "$V5_MIN_SOURCE_SCALE_LONG_SIDE" \
        --keep_negative False \
        --jpeg_quality "$V5_JPEG_QUALITY"
else
    echo "  reusing $VALI_TILES_FPATH"
fi

# ---------------------------------------------------------------------------
# Split train tiles into positive / negative kwcocos (no re-extraction;
# we just filter the existing bundle by tile_role).
# ---------------------------------------------------------------------------
echo
echo "=== Split train into positive / negative kwcocos ==="
"$PYTHON_BIN" - <<PY
import kwcoco
src = kwcoco.CocoDataset.coerce('$TRAIN_TILES_FPATH')
pos_gids = [img['id'] for img in src.images().objs if img.get('tile_role') == 'positive']
neg_gids = [img['id'] for img in src.images().objs if img.get('tile_role') == 'negative']
print(f'  source: {src.n_images} tiles ({len(pos_gids)} pos, {len(neg_gids)} neg)')
pos = src.subset(pos_gids); pos.fpath = '$TRAIN_POS_FPATH'; pos.dump()
neg = src.subset(neg_gids); neg.fpath = '$TRAIN_NEG_FPATH'; neg.dump()
print(f'  wrote $TRAIN_POS_FPATH ({pos.n_images} imgs, {pos.n_annots} anns)')
print(f'  wrote $TRAIN_NEG_FPATH ({neg.n_images} imgs)')

# Vali was already pos-only.
vali = kwcoco.CocoDataset.coerce('$VALI_TILES_FPATH')
vali.fpath = '$VALI_POS_FPATH'; vali.dump()
print(f'  wrote $VALI_POS_FPATH ({vali.n_images} imgs, {vali.n_annots} anns)')
PY

echo
echo "Done. Per-round merge happens in 02_train_round.sh."
printf '  %-32s %s\n' "train_pos"  "$TRAIN_POS_FPATH"
printf '  %-32s %s\n' "train_neg"  "$TRAIN_NEG_FPATH"
printf '  %-32s %s\n' "vali_pos"   "$VALI_POS_FPATH"
