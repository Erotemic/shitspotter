#!/usr/bin/env bash
# Run the desktop backend-comparison harness for a small matrix of
# images / configurations and dump JSON reports under
# tpl/shitspotter-phone-app/docs/benchmarks/.
#
# This is the desktop-side regression check before the user does a
# Pixel 5 run. It is intentionally small (single-machine, JVM CPU only)
# — the real benchmark suite lives on the device.
#
# Usage:
#   scripts/bench_desktop.sh [/path/to/image1 [/path/to/image2 ...]]
#
# Defaults to running against tpl/YOLOX/assets/dog.jpg if no images are
# given.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$APP_ROOT/../.." && pwd)"
MODEL="$REPO_ROOT/tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx"
OUT_DIR="$APP_ROOT/docs/benchmarks"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found at $MODEL" >&2
    echo "       expected via tpl/poop_models/ submodule" >&2
    exit 1
fi

if [ "$#" -eq 0 ]; then
    set -- "$REPO_ROOT/tpl/YOLOX/assets/dog.jpg"
fi

mkdir -p "$OUT_DIR"

if [ -z "${SHITSPOTTER_APP_TOOLCHAIN_ROOT:-}" ]; then
    if [ -f /data/tmp/shitspotter-app-toolchain/env.sh ]; then
        # shellcheck disable=SC1091
        . /data/tmp/shitspotter-app-toolchain/env.sh
    fi
fi

DATE="$(date -u +%Y-%m-%dT%H%M%SZ)"

cd "$APP_ROOT"

for IMG in "$@"; do
    if [ ! -f "$IMG" ]; then
        echo "WARN: image missing: $IMG, skipping" >&2
        continue
    fi
    BASE="$(basename "$IMG" | sed 's/\.[^.]*$//')"
    REPORT="$OUT_DIR/${DATE}_desktop_${BASE}.json"
    echo "=== bench: $IMG ==="
    ./gradlew :composeApp:run --console=plain --args="compare \
       --image=$IMG \
       --model=$MODEL \
       --runs=10 --warmup=2 \
       --out=$REPORT" 2>&1 | tail -20
    echo "→ wrote $REPORT"
done
