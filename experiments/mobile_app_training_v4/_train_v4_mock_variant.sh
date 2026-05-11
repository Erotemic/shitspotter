#!/bin/bash
# Library script — sourced by the per-variant 02_train_v4_mock_*.sh
# entrypoints AND dispatched to from 02_sweep.sh when V4_VARIANT starts
# with "v4_mock".
#
# Same contract as _train_deimv2_variant.sh:
#
#   in:  V4_VARIANT, V4_INPUT_HW, V4_TRAIN_BATCH, V4_VAL_BATCH,
#        V4_NUM_EPOCHS, V4_RUN_TAG, V4_TRAIN_POLICY, V4_*_KWCOCO,
#        V4_TILE_*, V4_ROOT, ...
#
#   out: $V4_ROOT/runs/<candidate_id>/best_stg2.pth
#        $V4_ROOT/runs/<candidate_id>/policy.json
#        $V4_ROOT/runs/<candidate_id>/generated_configs/train.yml
#
# The mock trainer is intentionally NOT distinguished from the DEIMv2
# trainer to the rest of the pipeline — the on-disk artifacts are
# identical so 03/04/manifest treat them the same.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
# shellcheck source=experiments/mobile_app_training_v4/common.sh
source "$_v4_script_dpath/common.sh"
unset _v4_source _v4_script_dpath

: "${V4_VARIANT:?V4_VARIANT must be set by the caller}"
: "${V4_INPUT_HW:?V4_INPUT_HW must be set by the caller (e.g. \"320 320\")}"
: "${V4_NUM_EPOCHS:?V4_NUM_EPOCHS must be set by the caller}"

V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-2}"
V4_VAL_BATCH="${V4_VAL_BATCH:-2}"
V4_NUM_QUERIES="${V4_NUM_QUERIES:-16}"
V4_TRAIN_POLICY="${V4_TRAIN_POLICY:-fixed}"
V4_LR="${V4_LR:-5e-2}"
V4_RUN_TAG="${V4_RUN_TAG:-tile_g${V4_TILE_GRID}_${V4_TRAIN_POLICY}}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

INPUT_H="${V4_INPUT_HW% *}"
INPUT_W="${V4_INPUT_HW#* }"

WORKDIR="$V4_ROOT/runs/${V4_VARIANT}_${V4_RUN_TAG}_${INPUT_H}x${INPUT_W}"
BEST_CKPT="$WORKDIR/best_stg2.pth"

DATA_DPATH="$V4_ROOT/data"
TRAIN_KWCOCO="${V4_MOCK_TRAIN_KWCOCO:-$DATA_DPATH/train_tile_g${V4_TILE_GRID}.simplified.kwcoco.zip}"
VALI_KWCOCO="${V4_MOCK_VALI_KWCOCO:-$DATA_DPATH/vali_tile_g${V4_TILE_GRID}.simplified.kwcoco.zip}"
v4_require_path "$TRAIN_KWCOCO"
v4_require_path "$VALI_KWCOCO"

echo "=== mobile_app_training_v4 / 02 train mock ${V4_VARIANT} ==="
v4_print_env
printf '  %-32s %s\n' "WORKDIR"          "$WORKDIR"
printf '  %-32s %s\n' "EXPORT_INPUT_HW"  "$V4_INPUT_HW"
printf '  %-32s %s\n' "TRAIN_KWCOCO"     "$TRAIN_KWCOCO"
printf '  %-32s %s\n' "VALI_KWCOCO"      "$VALI_KWCOCO"
printf '  %-32s %s\n' "TRAIN_POLICY"     "$V4_TRAIN_POLICY"
printf '  %-32s %s\n' "NUM_EPOCHS"       "$V4_NUM_EPOCHS"
printf '  %-32s %s\n' "BATCH"            "$V4_TRAIN_BATCH"
printf '  %-32s %s\n' "QUERIES"          "$V4_NUM_QUERIES"
printf '  %-32s %s\n' "LR"               "$V4_LR"

if [ "$FORCE_RETRAIN" != "1" ] && [ -f "$BEST_CKPT" ]; then
    echo "  already trained — found $BEST_CKPT"
    echo "  set FORCE_RETRAIN=1 to retrain"
    exit 0
fi
mkdir -p "$WORKDIR"

# Pass tile policy info through env so v4_mock can record it in
# policy.json identically to the DEIMv2 trainer.
export V4_VARIANT V4_RUN_TAG V4_TRAIN_POLICY
export V4_TILE_POLICY="tile_g${V4_TILE_GRID}_overlap${V4_TILE_OVERLAP}_out${V4_TILE_OUTPUT_DIM}"
export V4_TILE_GRID V4_TILE_OVERLAP V4_TILE_OUTPUT_DIM
export V4_CANDIDATE_ID="${V4_VARIANT}_${V4_RUN_TAG}_${INPUT_H}x${INPUT_W}"

"$PYTHON_BIN" "$V4_DEV_DPATH/v4_mock.py" train \
    --train_kwcoco "$TRAIN_KWCOCO" \
    --vali_kwcoco  "$VALI_KWCOCO" \
    --workdir      "$WORKDIR" \
    --input_h "$INPUT_H" --input_w "$INPUT_W" \
    --num_epochs "$V4_NUM_EPOCHS" \
    --batch_size "$V4_TRAIN_BATCH" \
    --num_queries "$V4_NUM_QUERIES" \
    --lr "$V4_LR"

echo
echo "Mock training complete:"
printf '  %-32s %s\n' "WORKDIR"   "$WORKDIR"
printf '  %-32s %s\n' "BEST_CKPT" "$BEST_CKPT"
echo
echo "Next: bash $V4_DEV_DPATH/03_export_onnx.sh $V4_VARIANT $V4_RUN_TAG $INPUT_H $INPUT_W"
