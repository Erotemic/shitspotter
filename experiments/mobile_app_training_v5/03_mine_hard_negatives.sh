#!/bin/bash
# Mine hard negatives from the trained model in $V5_ROUND.
# Outputs $V5_ROOT/rounds/round${V5_ROUND}/hard_negs.kwcoco.zip
# which becomes the input to round $V5_ROUND+1's merge.

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
V5_RUN_TAG="v5_round${V5_ROUND}"

WORKDIR="$ROUND_DPATH/v4_root/runs/${V5_VARIANT}_${V5_RUN_TAG}_${V5_INPUT_HW// /x}"
v4_require_path "$WORKDIR"
v4_require_path "$WORKDIR/best_stg2.pth"

TRAIN_NEG_FPATH="$V5_ROOT/data/train_tiles_neg.kwcoco.zip"
v4_require_path "$TRAIN_NEG_FPATH"

DST_FPATH="$ROUND_DPATH/hard_negs.kwcoco.zip"

echo "=== mobile_app_training_v5 / 03 mine hard negatives (round $V5_ROUND) ==="
printf '  %-32s %s\n' "ROUND_WORKDIR" "$WORKDIR"
printf '  %-32s %s\n' "neg_kwcoco"    "$TRAIN_NEG_FPATH"
printf '  %-32s %s\n' "dst"           "$DST_FPATH"
printf '  %-32s %s\n' "score_thresh"  "$V5_MINE_SCORE_THRESH"
printf '  %-32s %s\n' "max_hard"      "$V5_MAX_HARD_PER_ROUND"

V5_ROUND="$V5_ROUND" "$PYTHON_BIN" "$V5_DEV_DPATH/v5_mine.py" \
    --neg_kwcoco "$TRAIN_NEG_FPATH" \
    --workdir    "$WORKDIR" \
    --dst        "$DST_FPATH" \
    --score_thresh "$V5_MINE_SCORE_THRESH" \
    --max_hard_per_round "$V5_MAX_HARD_PER_ROUND" \
    --device "${V5_DEVICE:-cuda:0}"

echo
echo "Mined. Next round's training will use:"
printf '  %-32s %s\n' "neg_source_for_next_round" "$DST_FPATH"
