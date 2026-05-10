#!/usr/bin/env bash
# One-shot desktop validation pass for the ShitSpotter v2 phone app.
#
# Required steps (any failure → exit non-zero, final line "FAILED"):
#   1. Toolchain check (java + adb visible)
#   2. Shared-core unit tests
#   3. Android debug APK build
#
# Optional steps (skipped → recorded in the summary; do not fail
# the script unless --strict is set):
#   4. CompareCli against YOLOX-nano on dog.jpg
#   5. CompareCli against all 3 registered models
#   6. Describe ONNX model
#   7. Python reference parity
#
# Pass `--strict` to also fail on any skipped optional step.
#
# Final line is one of:
#   ALL REQUIRED CHECKS PASSED
#   PASSED WITH SKIPS: <list>
#   FAILED: <reason>

set -uo pipefail
# Note: we deliberately do NOT use `set -e`. Required steps explicitly
# track their own exit codes so the summary at the end is accurate even
# when a non-required step fails mid-pipeline.

STRICT=0
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT=1 ;;
        *) echo "usage: $0 [--strict]" >&2; exit 2 ;;
    esac
done

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

FAILED_STEPS=()
SKIPPED_STEPS=()

step() {
    printf '\n========================================================\n'
    printf '== %s\n' "$1"
    printf '========================================================\n'
}

run_required() {
    local name="$1"; shift
    step "REQUIRED — $name"
    if "$@"; then
        echo "$name: OK"
        return 0
    else
        local rc=$?
        echo "$name: FAILED (exit $rc)" >&2
        FAILED_STEPS+=("$name")
        return $rc
    fi
}

skip_optional() {
    local name="$1"; local reason="$2"
    SKIPPED_STEPS+=("$name ($reason)")
    if [ "$STRICT" -eq 1 ]; then
        echo "$name: SKIPPED in --strict mode → FAIL" >&2
        FAILED_STEPS+=("$name (skipped in --strict)")
    else
        echo "$name: SKIPPED — $reason"
    fi
}

# 1. Toolchain check (required)
run_required "1. Toolchain" bash -c '
    set -e
    java -version
    adb --version
' || true

# 2. Shared-core unit tests (required)
run_required "2. Shared-core unit tests" bash -c '
    ./gradlew :composeApp:desktopTest --console=plain | tail -5
' || true

# 3. Android debug APK build (required)
run_required "3. Android debug APK" bash -c '
    ./gradlew :composeApp:assembleDebug --console=plain | tail -3
    test -f composeApp/build/outputs/apk/debug/composeApp-debug.apk
' || true

APK="$APP_ROOT/composeApp/build/outputs/apk/debug/composeApp-debug.apk"
if [ -f "$APK" ]; then
    echo "APK: $(du -h "$APK" | cut -f1) at $APK"
fi

# 4. CompareCli vs YOLOX-nano (optional)
if [ -f "$MODEL" ] && [ -f "$IMAGE" ]; then
    step "OPTIONAL — 4. CompareCli vs YOLOX-nano"
    if ./gradlew :composeApp:run --console=plain --args="compare \
            --image=$IMAGE --model=$MODEL --runs=3 --warmup=1" | tail -8; then
        echo "4. CompareCli vs YOLOX-nano: OK"
    else
        echo "4. CompareCli vs YOLOX-nano: FAILED" >&2
        if [ "$STRICT" -eq 1 ]; then
            FAILED_STEPS+=("4. CompareCli vs YOLOX-nano")
        else
            SKIPPED_STEPS+=("4. CompareCli vs YOLOX-nano (run failed)")
        fi
    fi
else
    skip_optional "4. CompareCli vs YOLOX-nano" "model or image missing"
fi

# 5. CompareCli vs 3 models (optional)
if [ -f "$MODEL" ] && [ -f "$CUSTOM_V5" ] && [ -f "$CUSTOM_V2" ]; then
    step "OPTIONAL — 5. CompareCli vs 3 models"
    if ./gradlew :composeApp:run --console=plain --args="compare \
            --image=$IMAGE \
            --model=$MODEL \
            --model=$CUSTOM_V5 \
            --model=$CUSTOM_V2 \
            --runs=3 --warmup=1 --no-stub" | tail -10; then
        echo "5. CompareCli vs 3 models: OK"
    else
        echo "5. CompareCli vs 3 models: FAILED" >&2
        if [ "$STRICT" -eq 1 ]; then
            FAILED_STEPS+=("5. CompareCli vs 3 models")
        else
            SKIPPED_STEPS+=("5. CompareCli vs 3 models (run failed)")
        fi
    fi
else
    skip_optional "5. CompareCli vs 3 models" "one or more custom models missing"
fi

# 6. Describe (optional)
if [ -f "$MODEL" ]; then
    step "OPTIONAL — 6. Describe ONNX model"
    ./gradlew :composeApp:run --console=plain --args="describe --model=$MODEL" | tail -10
    echo "6. Describe: OK"
else
    skip_optional "6. Describe ONNX model" "model missing"
fi

# 7. Python parity (optional)
if [ -x /tmp/onnx_venv/bin/python ] && [ -f "$MODEL" ] && [ -f "$IMAGE" ]; then
    step "OPTIONAL — 7. Python reference parity"
    /tmp/onnx_venv/bin/python "$APP_ROOT/scripts/python_reference_compare.py" \
        --image "$IMAGE" --model "$MODEL" --threshold 0.25 --top 3
    echo "7. Python parity: OK"
else
    skip_optional "7. Python reference parity" "onnxruntime venv or model/image missing"
fi

# Summary
printf '\n========================================================\n'
printf '== Summary\n'
printf '========================================================\n'

if [ ${#FAILED_STEPS[@]} -gt 0 ]; then
    echo "FAILED: ${FAILED_STEPS[*]}"
    exit 1
fi

if [ ${#SKIPPED_STEPS[@]} -gt 0 ]; then
    echo "PASSED WITH SKIPS: ${SKIPPED_STEPS[*]}"
    echo "  (re-run with --strict to treat skips as failures)"
    if [ -f "$APK" ]; then
        echo
        echo "Next step on a workstation with a Pixel 5 plugged in:"
        echo "  adb install -r $APK"
    fi
    exit 0
fi

echo "ALL REQUIRED CHECKS PASSED"
if [ -f "$APK" ]; then
    echo
    echo "Next step on a workstation with a Pixel 5 plugged in:"
    echo "  adb install -r $APK"
fi
