#!/bin/bash
set -euo pipefail

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_DPATH="$(small_data_repo_dpath)"
DATA_DPATH="$(small_data_data_dpath)"
EXPT_DPATH="$(small_data_expt_dpath)"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-$EXPT_DPATH/small_data_tuning/dino_detector_benchmark}"
TRAIN_SRC="${TRAIN_SRC:-$(small_data_default_train_fpath)}"
VALI_SRC="${VALI_SRC:-$(small_data_default_vali_fpath)}"
TEST_SRC="${TEST_SRC:-$(small_data_default_test_fpath)}"
TRAIN_SIZES=(${TRAIN_SIZES:-128 256 512})
SEED="${SEED:-0}"
RESIZE_MAX_DIM="${RESIZE_MAX_DIM:-640}"
RESIZE_OUTPUT_EXT="${RESIZE_OUTPUT_EXT:-.jpg}"
SIMPLIFY_MINIMUM_INSTANCES="${SIMPLIFY_MINIMUM_INSTANCES:-1}"
OVERWRITE="${OVERWRITE:-False}"

ARGS=(
    "$PYTHON_BIN" "$REPO_DPATH/experiments/small-data-tuning/prepare_dino_detector_benchmark.py"
    --train_src "$TRAIN_SRC"
    --vali_src "$VALI_SRC"
    --test_src "$TEST_SRC"
    --out_root "$BENCHMARK_ROOT"
    --seed "$SEED"
    --resize_max_dim "$RESIZE_MAX_DIM"
    --resize_output_ext "$RESIZE_OUTPUT_EXT"
    --simplify_minimum_instances "$SIMPLIFY_MINIMUM_INSTANCES"
    --train_sizes "${TRAIN_SIZES[@]}"
)

case "$OVERWRITE" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        ARGS+=(--overwrite)
        ;;
esac

printf 'Prepare DINO detector benchmark\n'
printf '  %-24s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-24s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-24s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-24s %s\n' "BENCHMARK_ROOT" "$BENCHMARK_ROOT"
printf '  %-24s %s\n' "TRAIN_SRC" "$TRAIN_SRC"
printf '  %-24s %s\n' "VALI_SRC" "$VALI_SRC"
printf '  %-24s %s\n' "TEST_SRC" "$TEST_SRC"
printf '  %-24s %s\n' "TRAIN_SIZES" "${TRAIN_SIZES[*]}"
printf '  %-24s %s\n' "SEED" "$SEED"
printf '  %-24s %s\n' "RESIZE_MAX_DIM" "$RESIZE_MAX_DIM"
printf '  %-24s %s\n' "SIMPLIFY_MIN_INST" "$SIMPLIFY_MINIMUM_INSTANCES"

"${ARGS[@]}"
