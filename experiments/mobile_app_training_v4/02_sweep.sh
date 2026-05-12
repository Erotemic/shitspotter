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
# Graceful restart — by default every stage skips when its output is
# already on disk (best_stg2.pth / *.onnx / detect_metrics.json /
# *.bench.json). A re-run of `bash 02_sweep.sh` on the same V4_ROOT
# only does the missing work; cells that already finished get
# status=ok_resumed in the new sweep TSV. Force a specific stage to
# re-run with V4_SWEEP_FORCE_{TRAIN,EXPORT,EVAL,BENCH}=1.
#
# Retry only the cells that failed (or never ran) in a prior sweep:
#
#     V4_SWEEP_RETRY_FAILED=$V4_ROOT/sweeps/<UTC>/index.tsv \
#         V4_SWEEP_KEEP_GOING=1 bash 02_sweep.sh
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
# HGNetv2 variants (deimv2_n/pico) — encoder requires fixed input
# size, so each cell trains at its export resolution. Multi-resolution
# coverage for these comes from training across multiple cells, plus
# the tile augmentation that mixes 320..1280 px content per image.
deimv2_n       320 320  fixed
deimv2_n       416 416  fixed
deimv2_pico    320 320  fixed
deimv2_pico    416 416  fixed
deimv2_n       512 512  fixed
deimv2_pico    512 512  fixed
deimv2_n       640 640  fixed
# DINOv3-backed variants (deimv2_s/m/l/x) — encoder supports per-batch
# resize, so multi-scale jitter around the export size is fair game.
deimv2_s       640 640  multiscale_512_768
"

V4_SWEEP_CELLS="${V4_SWEEP_CELLS:-$DEFAULT_CELLS}"
V4_SWEEP_DO_EXPORT="${V4_SWEEP_DO_EXPORT:-1}"
V4_SWEEP_DO_EVAL="${V4_SWEEP_DO_EVAL:-1}"
V4_SWEEP_DO_BENCH="${V4_SWEEP_DO_BENCH:-1}"
V4_SWEEP_BENCH_ITERS="${V4_SWEEP_BENCH_ITERS:-50}"
V4_SWEEP_KEEP_GOING="${V4_SWEEP_KEEP_GOING:-0}"
V4_BENCH_IMAGE="${V4_BENCH_IMAGE:-$SHITSPOTTER_DPATH/tpl/YOLOX/assets/dog.jpg}"

# Graceful-restart knobs. By default the sweep is idempotent — each
# stage skips when its output already exists on disk. Flip these to 1
# to force the sweep to re-run that stage.
V4_SWEEP_FORCE_TRAIN="${V4_SWEEP_FORCE_TRAIN:-${FORCE_RETRAIN:-0}}"
V4_SWEEP_FORCE_EXPORT="${V4_SWEEP_FORCE_EXPORT:-${FORCE_REEXPORT:-0}}"
V4_SWEEP_FORCE_EVAL="${V4_SWEEP_FORCE_EVAL:-${FORCE_REEVAL:-0}}"
V4_SWEEP_FORCE_BENCH="${V4_SWEEP_FORCE_BENCH:-${FORCE_REBENCH:-0}}"

# Optional: drop cells already in `status=ok` (or `status=ok_resumed`)
# in a prior sweep TSV from the work list. Useful when restarting after
# fixing a bug that broke only some cells — pass the prior sweep's
# index.tsv and only the non-ok cells get queued.
V4_SWEEP_RETRY_FAILED="${V4_SWEEP_RETRY_FAILED:-}"

# Raise FD limit for the whole sweep — see _train_deimv2_variant.sh
# for the rationale. Doing it here too means bench/eval subprocesses
# inherit the higher limit, not just the trainer.
V4_FD_LIMIT="${V4_FD_LIMIT:-65536}"
if [ "$(ulimit -n 2>/dev/null || echo 0)" -lt "$V4_FD_LIMIT" ] 2>/dev/null; then
    ulimit -n "$V4_FD_LIMIT" 2>/dev/null || true
fi

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
printf '  %-32s %s\n' "FORCE_TRAIN"        "$V4_SWEEP_FORCE_TRAIN"
printf '  %-32s %s\n' "FORCE_EXPORT"       "$V4_SWEEP_FORCE_EXPORT"
printf '  %-32s %s\n' "FORCE_EVAL"         "$V4_SWEEP_FORCE_EVAL"
printf '  %-32s %s\n' "FORCE_BENCH"        "$V4_SWEEP_FORCE_BENCH"
printf '  %-32s %s\n' "RETRY_FAILED"       "${V4_SWEEP_RETRY_FAILED:-<all cells>}"
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

# Optional retry-only-failed filter: load a prior sweep's index.tsv and
# remove from this run any cell whose prior status starts with `ok`.
# Cells absent from the prior index are kept (they never ran).
if [ -n "$V4_SWEEP_RETRY_FAILED" ]; then
    if [ ! -f "$V4_SWEEP_RETRY_FAILED" ]; then
        echo "  V4_SWEEP_RETRY_FAILED points at a missing file: $V4_SWEEP_RETRY_FAILED" >&2
        exit 1
    fi
    # Read prior (candidate_id -> status) into env, then filter.
    declare -A _PRIOR_STATUS=()
    while IFS=$'\t' read -r v eh ew tp cid rt wd onnx ed bj pj status; do
        [ "$v" = "variant" ] && continue
        _PRIOR_STATUS["$cid"]="$status"
    done < "$V4_SWEEP_RETRY_FAILED"

    _NEW_V=(); _NEW_H=(); _NEW_W=(); _NEW_P=(); _SKIPPED_OK=0
    for i in "${!VARIANTS[@]}"; do
        v="${VARIANTS[$i]}"; h="${INPUT_HS[$i]}"; w="${INPUT_WS[$i]}"; p="${POLICIES[$i]}"
        cid="${v}_tile_g${V4_TILE_GRID}_${p}_${h}x${w}"
        st="${_PRIOR_STATUS[$cid]:-}"
        case "$st" in
            ok|ok_resumed)
                _SKIPPED_OK=$(( _SKIPPED_OK + 1 ))
                ;;
            *)
                _NEW_V+=( "$v" ); _NEW_H+=( "$h" ); _NEW_W+=( "$w" ); _NEW_P+=( "$p" )
                ;;
        esac
    done
    echo "  RETRY_FAILED: kept $((${#VARIANTS[@]} - _SKIPPED_OK)) of ${#VARIANTS[@]} cells from prior sweep"
    echo "                (skipped $_SKIPPED_OK already-ok cells from $V4_SWEEP_RETRY_FAILED)"
    VARIANTS=( "${_NEW_V[@]}" ); INPUT_HS=( "${_NEW_H[@]}" )
    INPUT_WS=( "${_NEW_W[@]}" ); POLICIES=( "${_NEW_P[@]}" )
    if [ "${#VARIANTS[@]}" -eq 0 ]; then
        echo "  nothing left to do — every cell in the prior sweep is already ok."
        echo "  Set V4_SWEEP_FORCE_<TRAIN|EXPORT|EVAL|BENCH>=1 to redo a stage anyway."
        exit 0
    fi
fi

# Make the printf-on-success path explicit — each stage records its own
# exit code and status is the worst of the enabled stages, not "ok by
# default". `set -o pipefail` (already on via common.sh) plus
# PIPESTATUS lets us see through the `tee` shell.
#
# Returns 0 only when every enabled stage really passed.
run_cell() {
    local variant="$1" h="$2" w="$3" policy="$4"
    local run_tag="tile_g${V4_TILE_GRID}_${policy}"
    local candidate_id="${variant}_${run_tag}_${h}x${w}"
    local cell_log="$SWEEP_LOG_DPATH/${candidate_id}.log"
    local workdir="$V4_ROOT/runs/${candidate_id}"
    local onnx_fpath="$workdir/export/${variant}_h${h}_w${w}.onnx"
    local eval_dpath="$V4_ROOT/eval/${candidate_id}"
    local policy_json="$workdir/policy.json"
    local bench_json="$workdir/export/${variant}_h${h}_w${w}.bench.json"
    local stage_status="ok"
    local fail_stage=""
    local _train_did=0 _export_did=0 _eval_did=0 _bench_did=0

    echo
    echo "=========================================================="
    echo "  cell: $candidate_id  (policy=$policy, logs -> $cell_log)"
    echo "=========================================================="

    # ---- 1. train ---------------------------------------------------------
    # Dispatch on variant prefix: v4_mock_* uses the tiny torch trainer;
    # everything else uses the DEIMv2 trainer.
    case "$variant" in
        v4_mock*) trainer="_train_v4_mock_variant.sh" ;;
        *)        trainer="_train_deimv2_variant.sh" ;;
    esac
    if [ -f "$workdir/best_stg2.pth" ] || [ -f "$workdir/best_stg1.pth" ] \
        && ! v4_is_truthy "$V4_SWEEP_FORCE_TRAIN"; then
        echo "  [skip train] $workdir already has a best_*.pth" | tee -a "$cell_log"
    else
        _train_did=1
        (
            set -o pipefail
            V4_VARIANT="$variant" \
            V4_INPUT_HW="$h $w" \
            V4_TRAIN_POLICY="$policy" \
            V4_RUN_TAG="$run_tag" \
            FORCE_RETRAIN="$V4_SWEEP_FORCE_TRAIN" \
            bash "$V4_DEV_DPATH/$trainer" 2>&1 | tee -a "$cell_log"
        )
        if [ "$?" -ne 0 ]; then
            stage_status="fail_train"; fail_stage="train"
        fi
    fi

    # ---- 2. export --------------------------------------------------------
    if [ "$stage_status" = "ok" ] && v4_is_truthy "$V4_SWEEP_DO_EXPORT"; then
        if [ -f "$onnx_fpath" ] && [ "$(stat -c %s "$onnx_fpath" 2>/dev/null || echo 0)" -ge 262144 ] \
            && ! v4_is_truthy "$V4_SWEEP_FORCE_EXPORT"; then
            echo "  [skip export] $onnx_fpath already present" | tee -a "$cell_log"
        else
            _export_did=1
            (
                set -o pipefail
                FORCE_REEXPORT="$V4_SWEEP_FORCE_EXPORT" \
                bash "$V4_DEV_DPATH/03_export_onnx.sh" "$variant" "$run_tag" "$h" "$w" \
                    2>&1 | tee -a "$cell_log"
            )
            if [ "$?" -ne 0 ]; then
                stage_status="fail_export"; fail_stage="export"
            elif [ ! -f "$onnx_fpath" ]; then
                stage_status="fail_export"; fail_stage="export(no .onnx produced)"
            fi
        fi
    fi

    # ---- 3. eval ----------------------------------------------------------
    if [ "$stage_status" = "ok" ] && v4_is_truthy "$V4_SWEEP_DO_EVAL"; then
        if [ -f "$eval_dpath/eval/detect_metrics.json" ] \
            && ! v4_is_truthy "$V4_SWEEP_FORCE_EVAL"; then
            echo "  [skip eval] $eval_dpath/eval/detect_metrics.json already present" | tee -a "$cell_log"
        else
            _eval_did=1
            (
                set -o pipefail
                FORCE_REEVAL="$V4_SWEEP_FORCE_EVAL" \
                bash "$V4_DEV_DPATH/04_eval_on_test.sh" "$variant" "$run_tag" "$h" "$w" \
                    2>&1 | tee -a "$cell_log"
            )
            if [ "$?" -ne 0 ]; then
                stage_status="fail_eval"; fail_stage="eval"
            elif [ ! -f "$eval_dpath/eval/detect_metrics.json" ]; then
                stage_status="fail_eval"; fail_stage="eval(no metrics.json)"
            fi
        fi
    fi

    # ---- 4. bench ---------------------------------------------------------
    if [ "$stage_status" = "ok" ] && v4_is_truthy "$V4_SWEEP_DO_BENCH"; then
        if [ ! -f "$onnx_fpath" ]; then
            echo "  bench skipped: $onnx_fpath missing" | tee -a "$cell_log"
            bench_json=""
        elif [ -f "$bench_json" ] && ! v4_is_truthy "$V4_SWEEP_FORCE_BENCH"; then
            echo "  [skip bench] $bench_json already present" | tee -a "$cell_log"
        else
            _bench_did=1
            (
                set -o pipefail
                "$PYTHON_BIN" "$V4_DEV_DPATH/06_benchmark_onnx_desktop.py" \
                    --onnx "$onnx_fpath" \
                    --image "$V4_BENCH_IMAGE" \
                    --warmup 5 --iters "$V4_SWEEP_BENCH_ITERS" \
                    --dump_json "$bench_json" \
                    2>&1 | tee -a "$cell_log"
            )
            if [ "$?" -ne 0 ]; then
                stage_status="fail_bench"; fail_stage="bench"
                bench_json=""
            fi
        fi
    fi

    # If every enabled stage was a no-op (everything already on disk),
    # mark the cell `ok_resumed` so retry-failed runs treat it the same
    # as `ok` next time.
    if [ "$stage_status" = "ok" ] \
        && [ "$_train_did"  = "0" ] && [ "$_export_did" = "0" ] \
        && [ "$_eval_did"   = "0" ] && [ "$_bench_did"  = "0" ]; then
        stage_status="ok_resumed"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$variant" "$h" "$w" "$policy" "$candidate_id" "$run_tag" \
        "$workdir" "$onnx_fpath" "$eval_dpath" "${bench_json:-}" \
        "$policy_json" "$stage_status" \
        >> "$SWEEP_INDEX_FPATH"

    case "$stage_status" in
        ok|ok_resumed) return 0 ;;
    esac
    echo "  -> cell $candidate_id FAILED at: $fail_stage" | tee -a "$cell_log"
    return 1
}

failures=0
for i in "${!VARIANTS[@]}"; do
    # Use a guarded form so a failing run_cell doesn't abort the parent
    # under `set -e`. We've already recorded a non-ok row inside run_cell.
    set +e
    run_cell "${VARIANTS[$i]}" "${INPUT_HS[$i]}" "${INPUT_WS[$i]}" "${POLICIES[$i]}"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        failures=$(( failures + 1 ))
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
