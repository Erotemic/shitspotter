#!/bin/bash
set -euo pipefail

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
EXPT_DPATH="$(small_data_expt_dpath)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$EXPT_DPATH/small_data_tuning/dino_detector_benchmark}"
OUT_DPATH="${OUT_DPATH:-$BENCHMARK_ROOT/analysis}"
FOCUS_TRAIN_SIZES="${FOCUS_TRAIN_SIZES:-128 256 512 5747}"

ensure_writable_out_dpath() {
    local candidate="$1"
    mkdir -p "$candidate" 2>/dev/null || return 1
    local probe="$candidate/.peek_write_test"
    if : > "$probe" 2>/dev/null; then
        rm -f "$probe"
        return 0
    fi
    return 1
}

if ! ensure_writable_out_dpath "$OUT_DPATH"; then
    OUT_DPATH="${TMPDIR:-/tmp}/shitspotter_dino_detector_benchmark_analysis"
    mkdir -p "$OUT_DPATH"
fi

"$PYTHON_BIN" "$_small_data_script_dpath/analyze_dino_detector_benchmark.py" \
    --benchmark_root "$BENCHMARK_ROOT" \
    --out_dpath "$OUT_DPATH"

printf 'Filtered benchmark progress\n'
printf '  %-22s %s\n' "OUT_DPATH" "$OUT_DPATH"
printf '  %-22s %s\n' "SUMMARY_FPATH" "$OUT_DPATH/benchmark_summary.tsv"
printf '  %-22s %s\n' "PLOT_FPATH" "$OUT_DPATH/train_size_curve.png"
printf '  %-22s %s\n' "RUN_LINKS_DPATH" "$OUT_DPATH/run_links"
printf '  %-22s %s\n' "FOCUS_TRAIN_SIZES" "$FOCUS_TRAIN_SIZES"

"$PYTHON_BIN" - "$OUT_DPATH/benchmark_summary.tsv" $FOCUS_TRAIN_SIZES <<'PY'
import csv
import sys

summary_fpath = sys.argv[1]
focus_sizes = {int(arg) for arg in sys.argv[2:]}
rows = list(csv.DictReader(open(summary_fpath, 'r'), delimiter='\t'))
filtered = [row for row in rows if not focus_sizes or int(row['train_size']) in focus_sizes]

fieldnames = [
    'model_family',
    'config_tag',
    'status',
    'train_size',
    'train_batch_size',
    'selected_candidate_id',
    'poop_vali_ap',
    'poop_test_ap',
    'run_link_fpath',
]
writer = csv.DictWriter(sys.stdout, delimiter='\t', fieldnames=fieldnames)
writer.writeheader()
for row in filtered:
    writer.writerow({key: row.get(key, '') for key in fieldnames})
PY
