#!/bin/bash
# Evaluate the final round's checkpoint on the v9-canonical simplified
# test split via v4's eval script.

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

LAST_ROUND_IDX=$(( ${V5_NUM_ROUNDS:-3} - 1 ))
V5_ROUND="${1:-$LAST_ROUND_IDX}"
V5_RUN_TAG="v5_round${V5_ROUND}"
V4_ROUND_ROOT="$V5_ROOT/rounds/round${V5_ROUND}/v4_root"

echo "=== mobile_app_training_v5 / 06 eval on test (round $V5_ROUND) ==="

H="${V5_INPUT_HW% *}"
W="${V5_INPUT_HW#* }"
V4_ROOT="$V4_ROUND_ROOT" \
    bash "$V5_DEV_DPATH/../mobile_app_training_v4/04_eval_on_test.sh" \
    "$V5_VARIANT" "$V5_RUN_TAG" "$H" "$W"
