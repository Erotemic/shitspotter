#!/bin/bash
# Shared environment + helpers for the mobile_app_training_v5 workflow.
# Mirrors v4's common.sh pattern.

set -euo pipefail

_v5_source="${BASH_SOURCE[0]-}"
if [ -n "$_v5_source" ] && [ "$_v5_source" != "bash" ] && [ "$_v5_source" != "-bash" ]; then
    _v5_script_dpath="$(cd "$(dirname "$_v5_source")" && pwd)"
else
    _v5_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v5"
fi
unset _v5_source

# shellcheck source=experiments/mobile_app_training_v5/setup_env.sh
source "$_v5_script_dpath/setup_env.sh" >/dev/null
unset _v5_script_dpath

# Reuse v4's helpers — they're loaded via the common.sh chain in
# v4's setup_env.sh. Helpers we'd otherwise duplicate:
#   v4_is_truthy, v4_require_path, v4_print_env
# We re-define a few v5-specific ones below.

v5_print_env() {
    printf '  %-32s %s\n' "V5_DEV_DPATH"      "$V5_DEV_DPATH"
    printf '  %-32s %s\n' "V5_ROOT"           "$V5_ROOT"
    printf '  %-32s %s\n' "V5_TILE_SIZE"      "$V5_TILE_SIZE"
    printf '  %-32s %s\n' "V5_SOURCE_SCALES"  "$V5_SOURCE_SCALES"
    printf '  %-32s %s\n' "V5_NUM_ROUNDS"     "$V5_NUM_ROUNDS"
    printf '  %-32s %s\n' "V5_VARIANT"        "$V5_VARIANT"
    printf '  %-32s %s\n' "V5_INPUT_HW"       "$V5_INPUT_HW"
    printf '  %-32s %s\n' "V4_TRAIN_FPATH"    "$V4_TRAIN_FPATH"
    printf '  %-32s %s\n' "V4_VALI_FPATH"     "$V4_VALI_FPATH"
}

# Idempotent makedir for known v5 layout.
v5_setup_dirs() {
    mkdir -p \
        "$V5_ROOT/data" \
        "$V5_ROOT/runs" \
        "$V5_ROOT/eval" \
        "$V5_ROOT/rounds"
}

# Pull v4 helpers explicitly in case the chain hasn't loaded them yet.
# (v4's common.sh defines these; if we re-source common.sh it'd add
# set -euo pipefail multiple times, which is harmless but redundant.)
if ! command -v v4_is_truthy >/dev/null 2>&1; then
    # shellcheck source=experiments/mobile_app_training_v4/common.sh
    source "$V5_DEV_DPATH/../mobile_app_training_v4/common.sh"
fi
