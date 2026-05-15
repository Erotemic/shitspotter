#!/usr/bin/env bash
# Push ONNX models to a connected Android device.
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
#
# Model search order (first match wins):
#   tpl/poop_models/<file>
#   /data/joncrall/shitspotter_v4/runs/*/export/<file>
#   /data/joncrall/dvc-repos/shitspotter_dvc/models/**/<prefix>*.onnx

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
POOP_MODELS="$(cd "$SCRIPT_DIR/../../poop_models" 2>/dev/null && pwd)" || POOP_MODELS=""
RUNS_DIR="/data/joncrall/shitspotter_v4/runs"
DVC_MODELS="/data/joncrall/dvc-repos/shitspotter_dvc/models"
MODELS_DST="/sdcard/Android/data/$PKG/files/models"

# Find a model file by its canonical name. Checks multiple source locations.
find_model() {
    local name="$1"
    # 1. poop_models dir
    if [ -n "$POOP_MODELS" ] && [ -f "$POOP_MODELS/$name" ]; then
        echo "$POOP_MODELS/$name"; return
    fi
    # 2. shitspotter_v4 run export dirs
    if [ -d "$RUNS_DIR" ]; then
        local hit
        hit=$(find "$RUNS_DIR" -name "$name" -path "*/export/*" 2>/dev/null | head -1)
        if [ -n "$hit" ]; then echo "$hit"; return; fi
    fi
    # 3. DVC models dir — match on prefix (handles long checkpoint filenames)
    if [ -d "$DVC_MODELS" ]; then
        local prefix="${name%.onnx}"
        local hit
        hit=$(find "$DVC_MODELS" -name "${prefix}*.onnx" 2>/dev/null | head -1)
        if [ -n "$hit" ]; then echo "$hit"; return; fi
    fi
    echo ""
}

# Make sure the app data dir exists (launch the app first if not).
adb shell mkdir -p "$MODELS_DST"

# Migrate any models that are still in the old package's external dir.
OLD_DIRS=(
    "/sdcard/Android/data/io.kitware.shitspotter/files/models"
    "/sdcard/Android/data/io.github.erotemic.shitspotter/files/models"
)
for old in "${OLD_DIRS[@]}"; do
    if [ "$old" = "$MODELS_DST" ]; then continue; fi
    count=$(adb shell "ls '$old'/*.onnx 2>/dev/null | wc -l" | tr -d '[:space:]')
    if [ "${count:-0}" -gt 0 ]; then
        echo "→ migrating models from $old"
        adb shell "for f in '$old'/*.onnx; do dst='$MODELS_DST/\$(basename \"\$f\")'; [ -f \"\$dst\" ] || cp \"\$f\" \"\$dst\" && echo \"  cp \$(basename \$f)\"; done"
    fi
done

# All model files the app's ModelRegistry knows about.
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
missing=0

for f in "${KNOWN[@]}"; do
    src=$(find_model "$f")
    if [ -n "$src" ]; then
        sz=$(du -h "$src" | cut -f1)
        echo "→ pushing $f ($sz)"
        echo "  from: $src"
        adb push "$src" "$MODELS_DST/$f"
        pushed=$((pushed + 1))
    else
        echo "  skip $f (not found in any search path)"
        missing=$((missing + 1))
    fi
done

echo
echo "done: $pushed pushed, $missing not found"
echo "destination: $MODELS_DST"
