#!/bin/bash
# Run the v5 round loop: round 0 -> mine -> round 1 -> mine -> ... -> round N-1.
# After the last round, mining is skipped (no next round to feed).

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

NUM_ROUNDS="${V5_NUM_ROUNDS:-3}"
LAST_ROUND_IDX=$(( NUM_ROUNDS - 1 ))

echo "=== mobile_app_training_v5 / round loop ==="
v5_print_env
printf '  %-32s %s\n' "NUM_ROUNDS" "$NUM_ROUNDS"
echo

for r in $(seq 0 "$LAST_ROUND_IDX"); do
    echo
    echo "##########################################################"
    echo "##  ROUND $r / $LAST_ROUND_IDX"
    echo "##########################################################"
    V5_ROUND="$r" bash "$V5_DEV_DPATH/02_train_round.sh"
    if [ "$r" -lt "$LAST_ROUND_IDX" ]; then
        V5_ROUND="$r" bash "$V5_DEV_DPATH/03_mine_hard_negatives.sh"
    fi
done

LAST_WORKDIR="$V5_ROOT/rounds/round${LAST_ROUND_IDX}/v4_root/runs/${V5_VARIANT}_v5_round${LAST_ROUND_IDX}_${V5_INPUT_HW// /x}"
echo
echo "=== Round loop complete ==="
printf '  %-32s %s\n' "final_workdir" "$LAST_WORKDIR"
printf '  %-32s %s\n' "final_ckpt"    "$LAST_WORKDIR/best_stg2.pth"
echo
echo "Next: export + eval"
echo "  bash $V5_DEV_DPATH/05_export_onnx.sh"
echo "  bash $V5_DEV_DPATH/06_eval_on_test.sh"
