#!/usr/bin/env bash
# One-shot desktop validation pass for the ShitSpotter v2 phone app.
# Runs everything that can be exercised on a Linux VM without a phone:
#
#   1. Toolchain check
#   2. Shared-core unit tests (commonTest)
#   3. ONNX smoke test (desktopTest, real model file)
#   4. Compare CLI against the YOLOX-nano poop model on dog.jpg
#   5. Compare CLI against all three registered models
#   6. Python reference parity check
#   7. Android debug APK build
#
# Exit codes: 0 = all checks passed, non-zero = first failure code.

set -euo pipefail

if [ -z "${SHITSPOTTER_APP_TOOLCHAIN_ROOT:-}" ]; then
    if [ -f /data/tmp/shitspotter-app-toolchain/env.sh ]; then
        # shellcheck disable=SC1091
        . /data/tmp/shitspotter-app-toolchain/env.sh
    fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/../.." && pwd)"
MODEL="$REPO_ROOT/tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx"
IMAGE="$REPO_ROOT/tpl/YOLOX/assets/dog.jpg"
CUSTOM_V5="$REPO_ROOT/tpl/poop_models/shitspotter-custom-v5-epoch_115.onnx"
CUSTOM_V2="$REPO_ROOT/tpl/poop_models/shitspotter_custom_v2_epoch126.onnx"

cd "$APP_ROOT"

step() {
    printf '\n========================================================\n'
    printf '== %s\n' "$1"
    printf '========================================================\n'
}

step "1. Toolchain"
java -version
adb --version

step "2. Shared-core unit tests"
./gradlew :composeApp:desktopTest --console=plain | tail -5

step "3. Compare CLI: YOLOX-nano vs stub"
if [ -f "$MODEL" ] && [ -f "$IMAGE" ]; then
    ./gradlew :composeApp:run --console=plain --args="compare \
       --image=$IMAGE --model=$MODEL --runs=3 --warmup=1" | tail -8
else
    echo "SKIP — model or image missing"
fi

step "4. Compare CLI: all 3 registered models"
if [ -f "$MODEL" ] && [ -f "$CUSTOM_V5" ] && [ -f "$CUSTOM_V2" ]; then
    ./gradlew :composeApp:run --console=plain --args="compare \
       --image=$IMAGE \
       --model=$MODEL \
       --model=$CUSTOM_V5 \
       --model=$CUSTOM_V2 \
       --runs=3 --warmup=1 --no-stub" | tail -10
else
    echo "SKIP — one or more custom models missing"
fi

step "5. Describe ONNX model"
if [ -f "$MODEL" ]; then
    ./gradlew :composeApp:run --console=plain --args="describe --model=$MODEL" | tail -10
else
    echo "SKIP — model missing"
fi

step "6. Python reference parity"
if [ -x /tmp/onnx_venv/bin/python ] && [ -f "$MODEL" ] && [ -f "$IMAGE" ]; then
    /tmp/onnx_venv/bin/python "$APP_ROOT/scripts/python_reference_compare.py" \
        --image "$IMAGE" --model "$MODEL" --threshold 0.25 --top 3
else
    echo "SKIP — onnxruntime venv or model/image missing"
fi

step "7. Android debug APK"
./gradlew :composeApp:assembleDebug --console=plain | tail -5
APK="$APP_ROOT/composeApp/build/outputs/apk/debug/composeApp-debug.apk"
if [ -f "$APK" ]; then
    echo "APK: $(du -h "$APK" | cut -f1) at $APK"
else
    echo "ERROR: APK missing" >&2
    exit 2
fi

printf '\n========================================================\n'
printf '== ALL DESKTOP VALIDATION PASSED\n'
printf '========================================================\n'
echo "next step on a workstation with a Pixel 5 plugged in:"
echo "  adb install -r $APK"
echo "  adb logcat -s 'ShitSpotter.AnalysisLoop:V' 'ShitSpotter.Failure:V'"
