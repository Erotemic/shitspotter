#!/bin/bash
set -euo pipefail

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPT_DPATH="$(small_data_expt_dpath)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$EXPT_DPATH/small_data_tuning/dino_detector_benchmark}"
ANALYSIS_DPATH="$BENCHMARK_ROOT/analysis"

bash "$_small_data_script_dpath/prepare_dino_detector_benchmark.sh"
bash "$_small_data_script_dpath/run_opengroundingdino_dino_detector_benchmark.sh"
bash "$_small_data_script_dpath/run_deimv2_dino_detector_benchmark.sh"
"$PYTHON_BIN" "$_small_data_script_dpath/analyze_dino_detector_benchmark.py" \
    --benchmark_root "$BENCHMARK_ROOT" \
    --out_dpath "$ANALYSIS_DPATH"

printf 'DINO detector benchmark complete\n'
printf '  %-22s %s\n' "BENCHMARK_ROOT" "$BENCHMARK_ROOT"
printf '  %-22s %s\n' "ANALYSIS_DPATH" "$ANALYSIS_DPATH"
printf '  %-22s %s\n' "SUMMARY_FPATH" "$ANALYSIS_DPATH/benchmark_summary.tsv"
printf '  %-22s %s\n' "PLOT_FPATH" "$ANALYSIS_DPATH/train_size_curve.png"
printf '  %-22s %s\n' "VALI_ASSETS_DPATH" "$BENCHMARK_ROOT/eval_sets/vali/vali_assets_maxdim640"
printf '  %-22s %s\n' "TEST_ASSETS_DPATH" "$BENCHMARK_ROOT/eval_sets/test/test_assets_maxdim640"
