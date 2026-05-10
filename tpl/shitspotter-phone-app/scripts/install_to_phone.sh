#!/usr/bin/env bash
# Build the Android APK, install it on a connected phone, and
# (optionally) launch + tail logs in one go. Run this on the
# workstation that has the Pixel 5 plugged in — the VM does not
# have USB passthrough.
#
# Usage:
#   scripts/install_to_phone.sh                  # alias for `run`
#   scripts/install_to_phone.sh run              # build debug + install + launch + tail
#   scripts/install_to_phone.sh run release      # same but release APK
#   scripts/install_to_phone.sh debug            # build debug + install, no launch / tail
#   scripts/install_to_phone.sh release          # build release + install, no launch / tail
#   scripts/install_to_phone.sh launch           # just launch the installed app + tail
#   scripts/install_to_phone.sh logcat           # tail ShitSpotter.* + crashes (no clear)
#
# Ctrl-C exits the tail cleanly.

set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
usage: install_to_phone.sh [run|debug|release|launch|logcat] [debug|release] [--run]

  run [variant]   — build + install + launch + tail   (default)
  debug [--run]   — build debug APK + install; with --run also launch + tail
  release [--run] — build release APK + install; with --run also launch + tail
  launch          — launch already-installed app + tail
  logcat          — tail ShitSpotter.* logs without restarting

For run, the optional second argument overrides the APK variant.
For debug/release, --run extends the install with launch + tail.
USAGE
}

case "${1:-}" in
    -h|--help|help) usage; exit 0 ;;
esac

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
PKG="io.kitware.shitspotter"
ACTIVITY="$PKG/io.kitware.shitspotter.android.MainActivity"
# Tags to show during tail. ":S" at the end silences every other tag.
TAIL_FILTER=(
    -v threadtime
    -s
    "ShitSpotter.AnalysisLoop:V"
    "ShitSpotter.BackendMgr:V"
    "ShitSpotter.MainActivity:V"
    "ShitSpotter.Failure:V"
    "ShitSpotter.Settings:V"
    "ShitSpotter.Desktop:V"
    "AndroidRuntime:E"
    "libc:F"
    "DEBUG:V"
)

cd "$APP_ROOT"

do_build() {
    local variant="${1:-debug}"
    case "$variant" in
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
        *)
            usage
            exit 2
            ;;
    esac
    if [ ! -f "$APK" ]; then
        echo "ERROR: APK not at $APK" >&2
        exit 1
    fi
}

do_install() {
    echo "→ installing $APK ($(du -h "$APK" | cut -f1))"
    adb install -r "$APK"
}

do_launch() {
    echo "→ clearing logcat buffer"
    adb logcat -c
    echo "→ stopping any previous instance"
    adb shell am force-stop "$PKG" || true
    echo "→ launching $ACTIVITY"
    adb shell am start -n "$ACTIVITY" >/dev/null
    sleep 0.5
}

do_tail() {
    echo "→ tailing logs (Ctrl-C to exit)"
    echo "  filter: ShitSpotter.* + AndroidRuntime:E + libc:F + DEBUG:V"
    echo
    exec adb logcat "${TAIL_FILTER[@]}"
}

# Check the rest of argv for a --run flag so debug/release can opt into
# the launch+tail tail without spelling out the longer `run release` form.
run_after_install=0
for arg in "${@:2}"; do
    case "$arg" in
        --run|--launch) run_after_install=1 ;;
    esac
done

case "${1:-run}" in
    run)
        # Variant defaults to "debug"; skip --run since it's the implicit mode.
        variant="${2:-debug}"
        case "$variant" in --run|--launch) variant=debug ;; esac
        do_build "$variant"
        do_install
        do_launch
        do_tail
        ;;
    debug)
        do_build debug
        do_install
        if [ "$run_after_install" -eq 1 ]; then
            do_launch
            do_tail
        else
            echo
            echo "→ done. To launch + tail: $0 launch  (or re-run with --run)"
        fi
        ;;
    release)
        do_build release
        do_install
        if [ "$run_after_install" -eq 1 ]; then
            do_launch
            do_tail
        else
            echo
            echo "→ done. To launch + tail: $0 launch  (or re-run with --run)"
        fi
        ;;
    launch)
        do_launch
        do_tail
        ;;
    logcat)
        echo "→ tailing existing logcat without clearing"
        exec adb logcat "${TAIL_FILTER[@]}"
        ;;
    *)
        usage
        exit 2
        ;;
esac
