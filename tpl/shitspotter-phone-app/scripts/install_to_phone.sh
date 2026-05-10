#!/usr/bin/env bash
# Build the Android debug APK and install it on a connected phone.
# Run this on the workstation that has the Pixel 5 plugged in — the VM
# does not have USB passthrough.
#
# Usage:
#   scripts/install_to_phone.sh                # debug APK (default)
#   scripts/install_to_phone.sh release        # release APK (smaller)
#   scripts/install_to_phone.sh logcat         # tail ShitSpotter logs only

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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$APP_ROOT"

case "${1:-debug}" in
    debug)
        echo "→ building debug APK"
        ./gradlew :composeApp:assembleDebug
        APK="$APP_ROOT/composeApp/build/outputs/apk/debug/composeApp-debug.apk"
        ;;
    release)
        echo "→ building release APK (R8 + arm64-v8a only)"
        ./gradlew :composeApp:assembleRelease
        APK="$APP_ROOT/composeApp/build/outputs/apk/release/composeApp-release.apk"
        ;;
    logcat)
        exec adb logcat -s "ShitSpotter.AnalysisLoop:V" \
                          "ShitSpotter.Failure:V" \
                          "ShitSpotter.MainActivity:V" \
                          "ShitSpotter.Settings:V"
        ;;
    *)
        echo "usage: $0 [debug|release|logcat]" >&2
        exit 2
        ;;
esac

echo "→ installing $APK ($(du -h "$APK" | cut -f1))"
adb install -r "$APK"

echo
echo "→ to follow ShitSpotter logs:"
echo "    $0 logcat"
echo
echo "→ to pull failure-case captures back:"
echo "    $SCRIPT_DIR/sync_failure_cases.sh"
