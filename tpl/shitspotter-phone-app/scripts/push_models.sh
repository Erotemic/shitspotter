#!/usr/bin/env bash
# Push ONNX models from tpl/poop_models/ to a connected Android device.
# Run this on the workstation that has the Pixel 5 plugged in.
#
# Usage:
#   scripts/push_models.sh              # push to debug app (default)
#   scripts/push_models.sh release      # push to release app
#   scripts/push_models.sh debug        # same as default
#
# The app looks for models in (priority order):
#   1. /sdcard/Android/data/<pkg>/files/models/   ← this script writes here
#   2. internal cache (previously copied from APK assets)
#   3. APK assets (bundled at build time)

set -euo pipefail

VARIANT="${1:-debug}"

if ! command -v adb >/dev/null 2>&1; then
    echo "ERROR: adb not on PATH" >&2
    echo "       source /data/tmp/shitspotter-app-toolchain/env.sh" >&2
    exit 1
fi

if [ "$(adb devices | grep -cE 'device$')" -eq 0 ]; then
    echo "ERROR: no Android device detected via 'adb devices'" >&2
    echo "       plug in the Pixel 5 and authorise USB debugging" >&2
    exit 1
fi

case "$VARIANT" in
    debug)   PKG="io.github.erotemic.shitspotter.debug" ;;
    release) PKG="io.github.erotemic.shitspotter" ;;
    *)       echo "ERROR: unknown variant '$VARIANT' (use debug or release)" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_SRC="$(cd "$SCRIPT_DIR/../../poop_models" && pwd)"
MODELS_DST="/sdcard/Android/data/$PKG/files/models"

if [ ! -d "$MODELS_SRC" ]; then
    echo "ERROR: model source dir not found: $MODELS_SRC" >&2
    echo "       expected tpl/poop_models/ relative to this script" >&2
    exit 1
fi

# Make sure the app has created the data dir (launch it first if not).
adb shell mkdir -p "$MODELS_DST"

# Push only the ONNX files the app's ModelRegistry knows about.
KNOWN=(
    yolox_nano_poop_cropped_only_best.onnx
    shitspotter-custom-v5-epoch_115.onnx
    shitspotter_custom_v2_epoch126.onnx
    shitspotter-simple-v3-run-v06.onnx
    deimv2_pico_h320_w320.onnx
    deimv2_pico_h416_w416.onnx
    deimv2_n_h512_w512.onnx
    deimv2_n_h640_w640.onnx
)

pushed=0
skipped=0
missing=0

for f in "${KNOWN[@]}"; do
    src="$MODELS_SRC/$f"
    if [ -f "$src" ]; then
        sz=$(du -h "$src" | cut -f1)
        echo "→ pushing $f ($sz)"
        adb push "$src" "$MODELS_DST/$f"
        pushed=$((pushed + 1))
    else
        echo "  skip $f (not in $MODELS_SRC)"
        missing=$((missing + 1))
    fi
done

echo
echo "done: $pushed pushed, $missing not found locally"
echo "destination: $MODELS_DST"
