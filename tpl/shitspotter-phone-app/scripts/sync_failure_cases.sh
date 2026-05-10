#!/usr/bin/env bash
# Pull failure-case captures off a connected Pixel 5 (or any Android
# device with the ShitSpotter v2 app installed) into a local dir,
# preserving timestamps. Run on the workstation that has the phone
# connected over USB; the VM has no USB passthrough.
#
# Usage:
#   scripts/sync_failure_cases.sh [destination_dir]
#
# Default destination: ./pulled_failure_cases/<DATE>/

set -euo pipefail

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

DEFAULT_DEST="./pulled_failure_cases/$(date -u +%Y%m%d_%H%M%S)"
DEST="${1:-$DEFAULT_DEST}"
SRC="/sdcard/Android/data/io.kitware.shitspotter/files/failure_cases"

mkdir -p "$DEST"
echo "→ pulling $SRC → $DEST"

# adb pull preserves the directory tree; we get $DEST/failure_cases/<ts>/...
if ! adb pull -a "$SRC" "$DEST" 2>&1; then
    echo "ERROR: adb pull failed; is the app installed?" >&2
    echo "       try 'adb shell run-as io.kitware.shitspotter ls files/'" >&2
    exit 1
fi

echo "→ pulled to $DEST/$(basename "$SRC")"
echo
echo "Tip: each captured case is a directory containing:"
echo "  image.jpg            (ARGB camera frame, rotated to display)"
echo "  metadata.json        (FailureCaseMetadata schema)"
echo "  detections.json      (List<Detection>)"
echo "  user_note.txt        (optional)"
