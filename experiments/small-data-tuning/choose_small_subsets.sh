#!/bin/bash
set -euo pipefail

# Thin convenience wrapper around select_subsets.py. This is the "materialize
# the benchmark cohorts" entrypoint referenced by the README. It keeps the
# default paths and cohort sizes in one visible place so researchers do not need
# to remember a long Python invocation.

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_DPATH="$(small_data_repo_dpath)"
DATA_DPATH="$(small_data_data_dpath)"
EXPT_DPATH="$(small_data_expt_dpath)"
OUT_ROOT="${OUT_ROOT:-$EXPT_DPATH/small_data_tuning/cohorts}"

TRAIN_SRC="${TRAIN_SRC:-$(small_data_default_train_fpath)}"
VALI_SRC="${VALI_SRC:-$(small_data_default_vali_fpath)}"
TEST_SRC="${TEST_SRC:-$(small_data_default_test_fpath)}"

TRAIN_SIZES=(${TRAIN_SIZES:-128 256 512})
VALI_SIZE="${VALI_SIZE:-64}"
TEST_SIZE="${TEST_SIZE:-64}"
SELECTOR="${SELECTOR:-random}"
SEED="${SEED:-0}"
STRATIFY="${STRATIFY:-positive_negative}"
ABSOLUTE_PATHS="${ABSOLUTE_PATHS:-True}"
OVERWRITE="${OVERWRITE:-False}"

ARGS=(
    "$PYTHON_BIN" "$REPO_DPATH/experiments/small-data-tuning/select_subsets.py"
    --train_src "$TRAIN_SRC"
    --vali_src "$VALI_SRC"
    --test_src "$TEST_SRC"
    --out_root "$OUT_ROOT"
    --selector "$SELECTOR"
    --seed "$SEED"
    --stratify "$STRATIFY"
    --vali_size "$VALI_SIZE"
    --test_size "$TEST_SIZE"
    --train_sizes "${TRAIN_SIZES[@]}"
)

case "${ABSOLUTE_PATHS:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        ARGS+=(--absolute_paths)
        ;;
esac

case "${OVERWRITE:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        ARGS+=(--overwrite)
        ;;
esac

printf 'Small-data subset generation\n'
printf '  %-20s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-20s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-20s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-20s %s\n' "OUT_ROOT" "$OUT_ROOT"
printf '  %-20s %s\n' "TRAIN_SRC" "$TRAIN_SRC"
printf '  %-20s %s\n' "VALI_SRC" "$VALI_SRC"
printf '  %-20s %s\n' "TEST_SRC" "$TEST_SRC"
printf '  %-20s %s\n' "TRAIN_SIZES" "${TRAIN_SIZES[*]}"
printf '  %-20s %s\n' "VALI_SIZE" "$VALI_SIZE"
printf '  %-20s %s\n' "TEST_SIZE" "$TEST_SIZE"
printf '  %-20s %s\n' "SELECTOR" "$SELECTOR"
printf '  %-20s %s\n' "SEED" "$SEED"
printf '  %-20s %s\n' "STRATIFY" "$STRATIFY"

"${ARGS[@]}"
