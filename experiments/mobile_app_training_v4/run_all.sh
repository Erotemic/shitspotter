#!/bin/bash
# One-shot end-to-end driver for mobile_app_training_v4.
#
# Sources setup_env.sh, then runs:
#   00_setup.sh                          env check + pretrained DEIMv2 weights
#   01_make_tile_augmented_kwcoco.sh     tile-augmented training bundle
#   02_sweep.sh                          full Pareto sweep (train/export/eval/bench)
#   eligibility_manifest.py              aggregate + print HOST_PROMISING winner
#
# Run on the host with a CUDA GPU. The VM is fine for steps 0/1 + the
# manifest aggregation, but 02_sweep.sh will be unusably slow without a
# GPU.
#
# Usage:
#     bash experiments/mobile_app_training_v4/run_all.sh
#
# Knobs you may want to set BEFORE invoking (everything has a default):
#
#     V4_ROOT=/scratch/v4              writable workspace (default ~/data/shitspotter_v4)
#     PYTHON_BIN=python3               python interpreter
#     V4_SWEEP_KEEP_GOING=1            don't abort the sweep on a failed cell
#     V4_SWEEP_CELLS="..."             override the candidate matrix
#                                        rows: <variant> <h> <w> <train_policy>
#     V4_MAX_DESKTOP_MS=80             desktop CPU mean ms gate for HOST_PROMISING
#     V4_PIXEL5_INDEX=/path.tsv        on-device bench TSV (lets the script print
#                                        the deploy-eligible winner too). Optional.
#     V4_RUN_ALL_SMOKE=1               smoke mode: run only the cheapest single
#                                        cell instead of the full sweep, to
#                                        verify the pipeline before committing
#                                        many GPU-hours.
#
# Re-running is safe — every step short-circuits on existing outputs.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
unset _v4_source

# shellcheck source=experiments/mobile_app_training_v4/setup_env.sh
source "$_v4_dpath/setup_env.sh"

V4_MAX_DESKTOP_MS="${V4_MAX_DESKTOP_MS:-80}"
V4_PIXEL5_INDEX="${V4_PIXEL5_INDEX:-}"
V4_RUN_ALL_SMOKE="${V4_RUN_ALL_SMOKE:-0}"

# In smoke mode, replace the default sweep matrix with a single cheap
# cell so the user can verify the whole pipeline end-to-end without
# spending hours on GPU.
if [ "$V4_RUN_ALL_SMOKE" = "1" ] && [ -z "${V4_SWEEP_CELLS:-}" ]; then
    export V4_SWEEP_CELLS="deimv2_pico 320 320 multiscale_256_416"
    export V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-2}"
    echo "[run_all] SMOKE mode: V4_SWEEP_CELLS=\"$V4_SWEEP_CELLS\""
    echo "[run_all] SMOKE mode: V4_NUM_EPOCHS=$V4_NUM_EPOCHS"
fi

print_step() {
    echo
    echo "=========================================================="
    echo "  [run_all] $*"
    echo "=========================================================="
}

print_step "step 0/3 — environment + pretrained DEIMv2 weights"
bash "$_v4_dpath/00_setup.sh"

print_step "step 1/3 — tile-augmented kwcoco bundle"
bash "$_v4_dpath/01_make_tile_augmented_kwcoco.sh"

print_step "step 2/3 — Pareto sweep (train + export + eval + bench)"
bash "$_v4_dpath/02_sweep.sh"

print_step "step 3/3 — aggregate eligibility manifest"
MANIFEST_TSV="$V4_ROOT/manifest.tsv"
MANIFEST_JSON="$V4_ROOT/manifest.json"
MANIFEST_ARGS=(
    --auto
    --max_desktop_ms "$V4_MAX_DESKTOP_MS"
    --out "$MANIFEST_TSV"
    --out_json "$MANIFEST_JSON"
)
if [ -n "$V4_PIXEL5_INDEX" ]; then
    MANIFEST_ARGS+=( --pixel5_index "$V4_PIXEL5_INDEX" )
fi
"$PYTHON_BIN" "$_v4_dpath/eligibility_manifest.py" "${MANIFEST_ARGS[@]}"

echo
echo "=========================================================="
echo "  [run_all] DONE"
echo "=========================================================="
echo "  Manifest TSV:  $MANIFEST_TSV"
echo "  Manifest JSON: $MANIFEST_JSON"
echo
if [ -z "$V4_PIXEL5_INDEX" ]; then
    cat <<EOF
Next steps:
  1. Sideload the host-promising winner ONNX onto a Pixel 5
     (see $_v4_dpath/07_register_in_phone_app.md).
  2. Capture a "<candidate_id>\\t<latency_ms>\\t<fps>" TSV from the
     on-device benchmark.
  3. Re-run the manifest aggregation with the on-device data:

       V4_PIXEL5_INDEX=/path/to/pixel5_bench.tsv \\
           bash $_v4_dpath/run_all.sh

     The sweep is idempotent, so this is cheap — only the manifest
     step actually re-runs.
EOF
fi
