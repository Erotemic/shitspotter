#!/bin/bash
# Pareto sweep: walk a (variant, input_h, input_w) matrix end-to-end —
# train, export ONNX, run kwcoco eval, run desktop CPU latency bench.
#
# This is the recommended "best mobile detector" entrypoint. The
# per-variant 02_train_*.sh scripts remain useful when you want to
# investigate one cell in isolation, but the sweep is what builds the
# accuracy/latency frontier the eligibility manifest aggregates.
#
# Usage examples
# --------------
# Default escalating sweep — N at 320 first (cheapest), then Pico at
# the same size, then N at 416. S is treated as the quality reference,
# not part of the deploy sweep:
#
#     bash 02_sweep.sh
#
# Explicit matrix (one row per cell, "<variant> <h> <w>"):
#
#     V4_SWEEP_CELLS="
#       deimv2_n    320 320
#       deimv2_n    416 416
#       deimv2_pico 320 320
#       deimv2_n    512 512
#       deimv2_s    640 640
#     " bash 02_sweep.sh
#
# Skip the bench/eval steps and only do training:
#
#     V4_SWEEP_DO_EXPORT=0 V4_SWEEP_DO_EVAL=0 V4_SWEEP_DO_BENCH=0 \
#         bash 02_sweep.sh
#
# Continue past per-cell failures (default: stop on first error):
#
#     V4_SWEEP_KEEP_GOING=1 bash 02_sweep.sh
#
# Knobs you can override per-cell via env: V4_TRAIN_BATCH, V4_VAL_BATCH,
# V4_NUM_EPOCHS, V4_NUM_GPUS, FORCE_RETRAIN, FORCE_REPRED.

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

# Default Pareto sweep — one row per deployable candidate. Each row is
# "<variant> <export_h> <export_w> <train_policy>". The train policy is
# a string consumed by _train_deimv2_variant.sh; supported values:
#
#   fixed                    — single scale = export size
#   multiscale               — DEIMv2 ±25% band around base=max(H,W)
#   multiscale_<lo>_<hi>     — ±25% band targeting [lo, hi]
#   multiscale_<S>           — ±25% band centered on <S>
#
# The candidate identity is (variant, export_h, export_w, train_policy):
# distinct rows produce distinct workdirs, distinct ONNX exports, and
# distinct phone-app ModelSpec entries. Order is cheapest-first so you
# can stop early when a small candidate already passes the FPS gate.
#
# Lines starting with # are ignored.
DEFAULT_CELLS="
# variant      export_h export_w  train_policy
deimv2_n       320 320  multiscale_256_416
deimv2_n       416 416  multiscale_320_512
deimv2_pico    320 320  multiscale_256_416
deimv2_pico    416 416  multiscale_320_512
deimv2_n       512 512  multiscale_384_640
deimv2_pico    512 512  multiscale_384_640
deimv2_n       640 640  multiscale_512_768
deimv2_s       640 640  multiscale_512_768
"

V4_SWEEP_CELLS="${V4_SWEEP_CELLS:-$DEFAULT_CELLS}"
V4_SWEEP_DO_EXPORT="${V4_SWEEP_DO_EXPORT:-1}"
V4_SWEEP_DO_EVAL="${V4_SWEEP_DO_EVAL:-1}"
V4_SWEEP_DO_BENCH="${V4_SWEEP_DO_BENCH:-1}"
V4_SWEEP_BENCH_ITERS="${V4_SWEEP_BENCH_ITERS:-50}"
V4_SWEEP_KEEP_GOING="${V4_SWEEP_KEEP_GOING:-0}"
V4_BENCH_IMAGE="${V4_BENCH_IMAGE:-$SHITSPOTTER_DPATH/tpl/poop_models/dog.jpg}"

SWEEP_LOG_DPATH="$V4_ROOT/sweeps/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SWEEP_LOG_DPATH"
SWEEP_INDEX_FPATH="$SWEEP_LOG_DPATH/index.tsv"

echo "=== mobile_app_training_v4 / 02_sweep ==="
v4_print_env
printf '  %-32s %s\n' "SWEEP_LOG_DPATH"   "$SWEEP_LOG_DPATH"
printf '  %-32s %s\n' "DO_EXPORT"          "$V4_SWEEP_DO_EXPORT"
printf '  %-32s %s\n' "DO_EVAL"            "$V4_SWEEP_DO_EVAL"
printf '  %-32s %s\n' "DO_BENCH"           "$V4_SWEEP_DO_BENCH"
printf '  %-32s %s\n' "KEEP_GOING"         "$V4_SWEEP_KEEP_GOING"
echo
echo "Cells:"
echo "$V4_SWEEP_CELLS" | sed -e 's/^/    /'

# Re-do the sweep index header now that we know about the policy column
printf 'variant\texport_h\texport_w\ttrain_policy\tcandidate_id\trun_tag\tworkdir\tonnx_fpath\teval_dpath\tbench_json\tpolicy_json\tstatus\n' \
    > "$SWEEP_INDEX_FPATH"

# Parse cells into arrays. Accept either 3 or 4 columns; default the
# 4th (policy) to "multiscale" so old call sites keep working.
VARIANTS=(); INPUT_HS=(); INPUT_WS=(); POLICIES=()
while read -r line; do
    line="${line%%#*}"
    set -- $line
    if [ "$#" -eq 0 ]; then continue; fi
    if [ "$#" -eq 3 ]; then set -- "$1" "$2" "$3" "multiscale"; fi
    if [ "$#" -ne 4 ]; then
        echo "  bad cell row: '$line' — expected '<variant> <h> <w> [<policy>]'" >&2
        exit 1
    fi
    VARIANTS+=( "$1" ); INPUT_HS+=( "$2" ); INPUT_WS+=( "$3" ); POLICIES+=( "$4" )
done <<< "$V4_SWEEP_CELLS"

run_cell() {
    local variant="$1" h="$2" w="$3" policy="$4"
    local run_tag="tile_g${V4_TILE_GRID}_${policy}"
    local candidate_id="${variant}_${run_tag}_${h}x${w}"
    local cell_log="$SWEEP_LOG_DPATH/${candidate_id}.log"
    local workdir="$V4_ROOT/runs/${candidate_id}"
    local onnx_fpath="$workdir/export/${variant}_h${h}_w${w}.onnx"
    local eval_dpath="$V4_ROOT/eval/${candidate_id}"
    local policy_json="$workdir/policy.json"
    local bench_json=""

    echo
    echo "=========================================================="
    echo "  cell: $candidate_id  (policy=$policy, logs → $cell_log)"
    echo "=========================================================="

    (
        V4_VARIANT="$variant" \
        V4_INPUT_HW="$h $w" \
        V4_TRAIN_POLICY="$policy" \
        V4_RUN_TAG="$run_tag" \
        bash "$V4_DEV_DPATH/_train_deimv2_variant.sh" 2>&1 | tee -a "$cell_log"
    )

    if v4_is_truthy "$V4_SWEEP_DO_EXPORT"; then
        bash "$V4_DEV_DPATH/03_export_onnx.sh" "$variant" "$run_tag" "$h" "$w" \
            2>&1 | tee -a "$cell_log"
    fi

    if v4_is_truthy "$V4_SWEEP_DO_EVAL"; then
        bash "$V4_DEV_DPATH/04_eval_on_test.sh" "$variant" "$run_tag" "$h" "$w" \
            2>&1 | tee -a "$cell_log"
    fi

    if v4_is_truthy "$V4_SWEEP_DO_BENCH"; then
        if [ ! -f "$onnx_fpath" ]; then
            echo "  WARN: $onnx_fpath missing, skipping bench" | tee -a "$cell_log"
        else
            bench_json="$workdir/export/${variant}_h${h}_w${w}.bench.json"
            "$PYTHON_BIN" "$V4_DEV_DPATH/06_benchmark_onnx_desktop.py" \
                --onnx "$onnx_fpath" \
                --image "$V4_BENCH_IMAGE" \
                --warmup 5 --iters "$V4_SWEEP_BENCH_ITERS" \
                --dump_json "$bench_json" \
                2>&1 | tee -a "$cell_log"
        fi
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$variant" "$h" "$w" "$policy" "$candidate_id" "$run_tag" \
        "$workdir" "$onnx_fpath" "$eval_dpath" "${bench_json:-}" \
        "$policy_json" "ok" \
        >> "$SWEEP_INDEX_FPATH"
}

failures=0
for i in "${!VARIANTS[@]}"; do
    if run_cell "${VARIANTS[$i]}" "${INPUT_HS[$i]}" "${INPUT_WS[$i]}" "${POLICIES[$i]}"; then
        :
    else
        failures=$(( failures + 1 ))
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${VARIANTS[$i]}" "${INPUT_HS[$i]}" "${INPUT_WS[$i]}" "${POLICIES[$i]}" \
            "" "tile_g${V4_TILE_GRID}_${POLICIES[$i]}" "" "" "" "" "" "fail" \
            >> "$SWEEP_INDEX_FPATH"
        if ! v4_is_truthy "$V4_SWEEP_KEEP_GOING"; then
            echo
            echo "Cell ${VARIANTS[$i]}@${INPUT_HS[$i]}x${INPUT_WS[$i]} (${POLICIES[$i]}) failed; aborting."
            echo "Set V4_SWEEP_KEEP_GOING=1 to continue past failures."
            break
        fi
    fi
done

echo
echo "=== sweep summary ==="
column -t -s $'\t' "$SWEEP_INDEX_FPATH" || cat "$SWEEP_INDEX_FPATH"

echo
echo "Aggregate the frontier with:"
echo "  $PYTHON_BIN $V4_DEV_DPATH/eligibility_manifest.py \\"
echo "      --sweep_index $SWEEP_INDEX_FPATH \\"
echo "      --max_desktop_ms 80 \\"
echo "      --out $SWEEP_LOG_DPATH/manifest.tsv"

if [ "$failures" -gt 0 ]; then
    echo
    echo "Sweep finished with $failures failed cell(s)."
    [ "$V4_SWEEP_KEEP_GOING" = "1" ] || exit 1
fi
