#!/bin/bash
# Export the final round's checkpoint to ONNX via v4's export script.

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

echo "=== mobile_app_training_v5 / 05 export (round $V5_ROUND) ==="

# v4's 03_export reads V4_ROOT to locate the workdir; we point it at
# the round's v4_root subdir so the rest of the export path is unchanged.
H="${V5_INPUT_HW% *}"
W="${V5_INPUT_HW#* }"
V4_ROOT="$V4_ROUND_ROOT" \
    bash "$V5_DEV_DPATH/../mobile_app_training_v4/03_export_onnx.sh" \
    "$V5_VARIANT" "$V5_RUN_TAG" "$H" "$W"
