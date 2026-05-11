#!/bin/bash
# Run a single training round.
#
# Round 0:  merge positives + random subsample of negatives, train.
# Round N>0:  merge positives + hard negs from round N-1, train from
#             the round N-1 checkpoint.
#
# Usage:
#     V5_ROUND=0 bash 02_train_round.sh
#     V5_ROUND=1 bash 02_train_round.sh
#
# The wrapper script run_round_loop.sh calls this in a loop with the
# mining step interleaved.

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

V5_ROUND="${V5_ROUND:-0}"
ROUND_DPATH="$V5_ROOT/rounds/round${V5_ROUND}"
mkdir -p "$ROUND_DPATH"

TRAIN_POS_FPATH="$V5_ROOT/data/train_tiles_pos.kwcoco.zip"
TRAIN_NEG_FPATH="$V5_ROOT/data/train_tiles_neg.kwcoco.zip"
VALI_POS_FPATH="$V5_ROOT/data/vali_tiles_pos.kwcoco.zip"
v4_require_path "$TRAIN_POS_FPATH"
v4_require_path "$VALI_POS_FPATH"

# ---------------------------------------------------------------------------
# Pick the negative-tile source for this round.
#   Round 0 -> the full train_tiles_neg.kwcoco.zip (random sampled by v5_merge)
#   Round N -> the hard_negs.kwcoco.zip mined in round N-1
# ---------------------------------------------------------------------------
if [ "$V5_ROUND" -eq 0 ]; then
    NEG_SOURCE="$TRAIN_NEG_FPATH"
    NEG_OVER_POS="$V5_ROUND0_NEG_OVER_POS"
    echo "=== Round 0: random subsample of negatives (ratio $NEG_OVER_POS) ==="
else
    PREV_ROUND=$(( V5_ROUND - 1 ))
    NEG_SOURCE="$V5_ROOT/rounds/round${PREV_ROUND}/hard_negs.kwcoco.zip"
    NEG_OVER_POS=0   # keep all mined hard negs
    echo "=== Round ${V5_ROUND}: hard negatives mined from round $PREV_ROUND ==="
    v4_require_path "$NEG_SOURCE"
fi

ROUND_TRAIN_FPATH="$ROUND_DPATH/train_round.kwcoco.zip"
ROUND_TRAIN_MSCOCO_FPATH="$ROUND_DPATH/train_round.simplified.mscoco.json"
ROUND_VALI_MSCOCO_FPATH="$ROUND_DPATH/vali.simplified.mscoco.json"

# ---------------------------------------------------------------------------
# Merge positives + selected negatives into the round training kwcoco.
# ---------------------------------------------------------------------------
echo
echo "=== Merging pos + neg into round training kwcoco ==="
"$PYTHON_BIN" "$V5_DEV_DPATH/v5_merge.py" \
    --pos_kwcoco "$TRAIN_POS_FPATH" \
    --neg_kwcoco "$NEG_SOURCE" \
    --dst "$ROUND_TRAIN_FPATH" \
    --neg_over_pos "$NEG_OVER_POS" \
    --seed "$V5_ROUND" \
    --round_index "$V5_ROUND"

# ---------------------------------------------------------------------------
# Export to MSCOCO json (DEIMv2's trainer needs that format). We re-use
# v4's coco_adapter.
# ---------------------------------------------------------------------------
echo
echo "=== Export round kwcoco -> MSCOCO json for DEIMv2 train.py ==="
"$PYTHON_BIN" - <<PY
from shitspotter.algo_foundation_v3.coco_adapter import _build_coco_export
_build_coco_export(
    src='$ROUND_TRAIN_FPATH',
    dst='$ROUND_TRAIN_MSCOCO_FPATH',
    category_name='poop',
    include_segmentations=False,
    category_id=0,
)
_build_coco_export(
    src='$VALI_POS_FPATH',
    dst='$ROUND_VALI_MSCOCO_FPATH',
    category_name='poop',
    include_segmentations=False,
    category_id=0,
)
print('wrote', '$ROUND_TRAIN_MSCOCO_FPATH')
print('wrote', '$ROUND_VALI_MSCOCO_FPATH')
PY

# ---------------------------------------------------------------------------
# Dispatch to v4's trainer.
#
# Set the v4 env vars so the v4 trainer reads the round's MSCOCO
# files rather than v4's own simplified bundle. The trick: v4's
# trainer derives the MSCOCO paths from V4_ROOT + V4_TILE_GRID.
# Rather than fighting that, we point V4_ROOT at the round dir and
# pre-place the expected files. Each round gets its own V4_ROOT-shaped
# subdir so v4's "already trained" short-circuit doesn't fire.
# ---------------------------------------------------------------------------
V4_ROUND_ROOT="$ROUND_DPATH/v4_root"
mkdir -p "$V4_ROUND_ROOT/data"
# v4's trainer looks for these exact filenames:
cp -u "$ROUND_TRAIN_MSCOCO_FPATH" \
    "$V4_ROUND_ROOT/data/train_tile_g2.simplified.mscoco.json"
cp -u "$ROUND_VALI_MSCOCO_FPATH" \
    "$V4_ROUND_ROOT/data/vali_tile_g2.simplified.mscoco.json"

V5_RUN_TAG="v5_round${V5_ROUND}"

# Resume-from-prev-round so we keep the learned weights across rounds.
if [ "$V5_ROUND" -gt 0 ]; then
    PREV_ROUND=$(( V5_ROUND - 1 ))
    PREV_WORKDIR="$V5_ROOT/rounds/round${PREV_ROUND}/v4_root/runs/${V5_VARIANT}_${V5_RUN_TAG/round${V5_ROUND}/round${PREV_ROUND}}_${V5_INPUT_HW// /x}"
    if [ -f "$PREV_WORKDIR/best_stg2.pth" ]; then
        export V4_DEIMV2_INIT_CKPT="$PREV_WORKDIR/best_stg2.pth"
        echo
        echo "=== Resuming from round $PREV_ROUND checkpoint ==="
        echo "  init_ckpt = $V4_DEIMV2_INIT_CKPT"
    fi
fi

echo
echo "=== Dispatching to v4 trainer for round $V5_ROUND ==="
V4_ROOT="$V4_ROUND_ROOT" \
V4_VARIANT="$V5_VARIANT" \
V4_INPUT_HW="$V5_INPUT_HW" \
V4_TRAIN_POLICY="fixed" \
V4_RUN_TAG="$V5_RUN_TAG" \
V4_NUM_EPOCHS="$V5_ROUND_EPOCHS" \
V4_TILE_GRID=2 V4_TILE_OVERLAP=0.20 V4_TILE_OUTPUT_DIM="$V5_TILE_SIZE" \
    bash "$V5_DEV_DPATH/../mobile_app_training_v4/_train_deimv2_variant.sh"

ROUND_WORKDIR="$V4_ROUND_ROOT/runs/${V5_VARIANT}_${V5_RUN_TAG}_${V5_INPUT_HW// /x}"
echo
echo "Round $V5_ROUND training done."
printf '  %-32s %s\n' "ROUND_WORKDIR" "$ROUND_WORKDIR"
printf '  %-32s %s\n' "best_stg2" "$ROUND_WORKDIR/best_stg2.pth"
