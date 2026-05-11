#!/bin/bash
# v5 end-to-end:
#   00_setup -> 01_make_multiscale_tile_dataset -> run_round_loop
#   -> 05_export_onnx -> 06_eval_on_test
#
# Knobs (all optional, with sensible defaults from setup_env.sh):
#   V5_ROOT, V5_TILE_SIZE, V5_SOURCE_SCALES, V5_NUM_ROUNDS,
#   V5_ROUND_EPOCHS, V5_VARIANT, V5_INPUT_HW
#
# Each step short-circuits on existing outputs. Re-running after a
# crash is cheap: tile bundles persist, completed rounds persist.

set -euo pipefail

_v5_source="${BASH_SOURCE[0]-}"
if [ -n "$_v5_source" ] && [ "$_v5_source" != "bash" ] && [ "$_v5_source" != "-bash" ]; then
    _v5_dpath="$(cd "$(dirname "$_v5_source")" && pwd)"
else
    _v5_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v5"
fi
unset _v5_source

# shellcheck source=experiments/mobile_app_training_v5/setup_env.sh
source "$_v5_dpath/setup_env.sh"

print_step() {
    echo
    echo "=========================================================="
    echo "  [v5 run_all] $*"
    echo "=========================================================="
}

print_step "step 0 — env + deps (delegates to v4 setup)"
bash "$_v5_dpath/00_setup.sh"

print_step "step 1 — multi-scale tile dataset"
bash "$_v5_dpath/01_make_multiscale_tile_dataset.sh"

print_step "step 2 — round loop ($V5_NUM_ROUNDS rounds)"
bash "$_v5_dpath/run_round_loop.sh"

print_step "step 3 — export final round's ONNX"
bash "$_v5_dpath/05_export_onnx.sh"

print_step "step 4 — eval final round on v9 simplified test"
bash "$_v5_dpath/06_eval_on_test.sh" || true

echo
echo "=========================================================="
echo "  [v5 run_all] DONE"
echo "=========================================================="
echo "  Final round outputs under $V5_ROOT/rounds/round$(( ${V5_NUM_ROUNDS:-3} - 1 ))"
