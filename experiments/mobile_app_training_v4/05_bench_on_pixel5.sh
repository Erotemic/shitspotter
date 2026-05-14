#!/bin/bash
# Benchmark Pareto-front DEIMv2 ONNX candidates on a connected Pixel 5 and
# write $V4_ROOT/pixel5_bench.tsv for eligibility_manifest.py --pixel5_index.
#
# Usage (run on the workstation with the phone plugged in via USB):
#   source experiments/mobile_app_training_v4/setup_env.sh
#   bash experiments/mobile_app_training_v4/05_bench_on_pixel5.sh
#
# Prerequisites:
#   - Pixel 5 connected and USB debugging authorised (adb devices shows it)
#   - adb on PATH: source /data/tmp/shitspotter-app-toolchain/env.sh
#   - Android NDK at $ANDROID_NDK_HOME (set by the toolchain env)
#   - libonnxruntime.so in Gradle cache (already present after first build)
#
# How it works:
#   1. Compiles ort_bench.c with the NDK clang (arm64-v8a, API 29) against the
#      libonnxruntime.so from the ORT 1.19.2 Android AAR already in ~/.gradle.
#   2. Pushes the binary, the shared library, and the Pareto-front ONNXes to
#      /data/local/tmp/v4_bench/ on the device.
#   3. Runs "ort_bench <model> 50 5 nnapi" for each cell; falls back to CPU if
#      NNAPI is unavailable.
#   4. Writes $V4_ROOT/pixel5_bench.tsv (candidate_id TAB latency_ms TAB fps).
#
# After this script completes, re-run the eligibility manifest:
#   V4_ROOT=/data/joncrall/shitspotter_v4 \
#   python experiments/mobile_app_training_v4/eligibility_manifest.py \
#       --auto \
#       --pixel5_index "$V4_ROOT/pixel5_bench.tsv" \
#       --max_desktop_ms 80 --min_pixel5_fps 10 \
#       --out      "$V4_ROOT/manifest.tsv" \
#       --out_json "$V4_ROOT/manifest.json"
#
# Knobs:
#   BENCH_ITERS=50         inference iterations per model (after warmup)
#   BENCH_WARMUP=5         warmup iterations
#   BENCH_EP=nnapi         execution provider: nnapi or cpu
#   DEVICE_TMP=/data/local/tmp/v4_bench   workspace on device
#   FORCE_REBUILD=1        recompile ort_bench even if binary already exists
#   PARETO_ONLY=1          (default) only benchmark non-dominated Pareto cells

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

BENCH_ITERS="${BENCH_ITERS:-50}"
BENCH_WARMUP="${BENCH_WARMUP:-5}"
BENCH_EP="${BENCH_EP:-nnapi}"
DEVICE_TMP="${DEVICE_TMP:-/data/local/tmp/v4_bench}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

# Pareto-front cells (non-dominated on desktop latency × AP).
# Each row: variant h w train_policy
PARETO_CELLS=(
    "deimv2_pico 320 320 fixed"
    "deimv2_pico 416 416 fixed"
    "deimv2_n    512 512 fixed"
    "deimv2_n    640 640 fixed"
)

PIXEL5_TSV="$V4_ROOT/pixel5_bench.tsv"
ORT_BENCH_SRC="$V4_DEV_DPATH/ort_bench.c"
ORT_BENCH_BIN="$V4_ROOT/ort_bench_arm64"

# ORT 1.19.2 arm64 libs from the Gradle cache — locate via glob.
ORT_GRADLE_ROOT="${HOME}/.gradle/caches"
ORT_AAR_DIR=$(find "$ORT_GRADLE_ROOT" \
    -path "*/onnxruntime-android-1.19.2/jni/arm64-v8a" \
    -type d 2>/dev/null | head -1)
ORT_LIBORT="${ORT_AAR_DIR}/libonnxruntime.so"
ORT_HEADERS=$(dirname "$(dirname "$(dirname "$ORT_AAR_DIR")")")/headers

# ---------------------------------------------------------------------------
# 0. Toolchain + adb checks
# ---------------------------------------------------------------------------
if ! command -v adb >/dev/null 2>&1; then
    if [ -f /data/tmp/shitspotter-app-toolchain/env.sh ]; then
        # shellcheck disable=SC1091
        source /data/tmp/shitspotter-app-toolchain/env.sh
    fi
fi
if ! command -v adb >/dev/null 2>&1; then
    echo "ERROR: adb not on PATH." >&2
    echo "  source /data/tmp/shitspotter-app-toolchain/env.sh" >&2
    exit 1
fi
if [ "$(adb devices | grep -cE 'device$')" -eq 0 ]; then
    echo "ERROR: no Android device detected by adb. Plug in the Pixel 5 and" >&2
    echo "       authorise USB debugging (allow this computer on the phone)." >&2
    exit 1
fi
DEVICE_MODEL=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r' || echo "unknown")
ANDROID_VER=$(adb shell getprop ro.build.version.release 2>/dev/null | tr -d '\r' || echo "?")
ABI=$(adb shell getprop ro.product.cpu.abi 2>/dev/null | tr -d '\r' || echo "arm64-v8a")

echo "=== mobile_app_training_v4 / 05 pixel5 bench ==="
printf '  %-28s %s\n' "V4_ROOT"       "$V4_ROOT"
printf '  %-28s %s\n' "PIXEL5_TSV"    "$PIXEL5_TSV"
printf '  %-28s %s\n' "Device"        "$DEVICE_MODEL (Android $ANDROID_VER, ABI=$ABI)"
printf '  %-28s %s\n' "bench params"  "iters=$BENCH_ITERS warmup=$BENCH_WARMUP ep=$BENCH_EP"
echo

# ---------------------------------------------------------------------------
# 1. Compile ort_bench.c with Android NDK
# ---------------------------------------------------------------------------
NDK_CLANG=""
for _api in 29 28 26 21; do
    _cand="${ANDROID_NDK_HOME:-/data/tmp/shitspotter-app-toolchain/android-sdk/ndk/26.3.11579264}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${_api}-clang"
    if [ -f "$_cand" ]; then
        NDK_CLANG="$_cand"; break
    fi
done
if [ -z "$NDK_CLANG" ]; then
    echo "ERROR: aarch64 NDK clang not found. Set ANDROID_NDK_HOME." >&2
    exit 1
fi

if [ ! -f "$ORT_LIBORT" ]; then
    echo "ERROR: libonnxruntime.so not found at: $ORT_LIBORT" >&2
    echo "  Run './gradlew :composeApp:assembleDebug' from tpl/shitspotter-phone-app/ first." >&2
    exit 1
fi
if [ ! -f "$ORT_HEADERS/onnxruntime_c_api.h" ]; then
    echo "ERROR: ORT headers not found at: $ORT_HEADERS" >&2
    exit 1
fi

if [ ! -f "$ORT_BENCH_BIN" ] || v4_is_truthy "$FORCE_REBUILD"; then
    echo "  compiling ort_bench.c (arm64-v8a, API 29)..."
    "$NDK_CLANG" \
        -O2 -Wall \
        -I "$ORT_HEADERS" \
        -L "$(dirname "$ORT_LIBORT")" \
        -o "$ORT_BENCH_BIN" \
        "$ORT_BENCH_SRC" \
        -l onnxruntime \
        -Wl,-rpath,/data/local/tmp/v4_bench \
        -lm -lc
    echo "  -> $ORT_BENCH_BIN"
else
    echo "  ort_bench binary already built: $ORT_BENCH_BIN"
fi

# ---------------------------------------------------------------------------
# 2. Push binary + library + models to device
# ---------------------------------------------------------------------------
echo
echo "  setting up device workspace: $DEVICE_TMP"
adb shell mkdir -p "$DEVICE_TMP"

echo "  pushing ort_bench_arm64..."
adb push "$ORT_BENCH_BIN" "$DEVICE_TMP/ort_bench" >/dev/null
adb shell chmod +x "$DEVICE_TMP/ort_bench"

echo "  pushing libonnxruntime.so ($(du -h "$ORT_LIBORT" | cut -f1))..."
adb push "$ORT_LIBORT" "$DEVICE_TMP/libonnxruntime.so" >/dev/null

# ---------------------------------------------------------------------------
# 3. Run benchmarks, collect results
# ---------------------------------------------------------------------------
printf 'candidate_id\tlatency_ms\tfps\n' > "${PIXEL5_TSV}.tmp"

for cell in "${PARETO_CELLS[@]}"; do
    read -r variant h w policy <<< "$cell"
    run_tag="tile_g${V4_TILE_GRID}_${policy}"
    cid="${variant}_${run_tag}_${h}x${w}"
    onnx_name="${variant}_h${h}_w${w}.onnx"
    onnx_fpath="$V4_ROOT/runs/$cid/export/$onnx_name"

    echo
    echo "--- $cid ---"

    if [ ! -f "$onnx_fpath" ]; then
        echo "  [SKIP] ONNX not found: $onnx_fpath"
        continue
    fi

    echo "  pushing $onnx_name ($(du -h "$onnx_fpath" | cut -f1))..."
    adb push "$onnx_fpath" "$DEVICE_TMP/$onnx_name" >/dev/null

    # Try requested EP then fall back to CPU.
    mean_ms=""
    fps_val=""
    used_ep=""
    for try_ep in "$BENCH_EP" cpu; do
        echo "  running: ort_bench iters=$BENCH_ITERS warmup=$BENCH_WARMUP ep=$try_ep"
        RAW=$(adb shell \
            "LD_LIBRARY_PATH=$DEVICE_TMP \
             $DEVICE_TMP/ort_bench \
             $DEVICE_TMP/$onnx_name \
             $BENCH_ITERS $BENCH_WARMUP $try_ep 2>&1" || true)
        echo "  raw: $RAW"
        # Parse: mean_ms=<N> fps=<N>
        mean_ms=$(echo "$RAW" | grep -oE 'mean_ms=[0-9]+\.[0-9]+' | cut -d= -f2 || true)
        fps_val=$(echo "$RAW"  | grep -oE 'fps=[0-9]+\.[0-9]+'     | cut -d= -f2 || true)
        if [ -n "$mean_ms" ] && [ -n "$fps_val" ]; then
            used_ep="$try_ep"
            break
        fi
    done

    if [ -n "$mean_ms" ]; then
        echo "  -> mean_ms=$mean_ms fps=$fps_val (ep=$used_ep)"
        printf '%s\t%s\t%s\n' "$cid" "$mean_ms" "$fps_val" >> "${PIXEL5_TSV}.tmp"
    else
        echo "  ERROR: could not parse timing. Raw output:"
        echo "$RAW" | sed 's/^/    /'
        printf '%s\t%s\t%s\n' "$cid" "" "" >> "${PIXEL5_TSV}.tmp"
    fi
done

mv "${PIXEL5_TSV}.tmp" "$PIXEL5_TSV"

# ---------------------------------------------------------------------------
# 4. Print results + next step
# ---------------------------------------------------------------------------
echo
echo "=== pixel5_bench results ==="
cat "$PIXEL5_TSV"
echo
echo "  wrote $PIXEL5_TSV"
echo
echo "  Next — re-run the eligibility manifest with phone numbers:"
echo
echo "    V4_ROOT=$V4_ROOT \\"
echo "    $PYTHON_BIN experiments/mobile_app_training_v4/eligibility_manifest.py \\"
echo "        --auto \\"
echo "        --pixel5_index \"$PIXEL5_TSV\" \\"
echo "        --max_desktop_ms 80 --min_pixel5_fps 10 \\"
echo "        --out      \"$V4_ROOT/manifest.tsv\" \\"
echo "        --out_json \"$V4_ROOT/manifest.json\""
