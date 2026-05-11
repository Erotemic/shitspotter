#!/bin/bash
# Shared environment + helpers for the mobile_app_training_v4 workflow.
#
# This file is sourced by every 0X_*.sh script. It enables `set -euo
# pipefail` (which is appropriate inside a script but NOT inside an
# interactive shell), then sources `setup_env.sh` to populate every env
# var the rest of the pipeline reads.
#
# If you want to set up the env vars in your own interactive shell
# *without* turning on pipefail, source `setup_env.sh` directly:
#
#     source experiments/mobile_app_training_v4/setup_env.sh
#
# Stage every derived artifact under $V4_ROOT (read-write). The DVC
# roots under /data/joncrall/dvc-repos/* are read-only — never write
# there.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate this script and load the canonical env exports
# ---------------------------------------------------------------------------
_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
unset _v4_source

# shellcheck source=experiments/mobile_app_training_v4/setup_env.sh
source "$_v4_script_dpath/setup_env.sh" >/dev/null
unset _v4_script_dpath

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
v4_is_truthy() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
}

v4_require_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
}

v4_canonical_existing_path() {
    local path="$1"
    v4_require_path "$path"
    (cd "$path" && pwd -P)
}

v4_variant_root() {
    # Per-variant working directory under V4_ROOT.
    local variant="$1"
    echo "$V4_ROOT/runs/$variant"
}

v4_variant_input_size() {
    # Default input H,W per variant — the script generators use this when
    # no V4_INPUT_SIZE override is in scope.
    local variant="$1"
    case "$variant" in
        deimv2_pico|deimv2_n) echo "320 320" ;;
        deimv2_s)             echo "640 640" ;;
        *)                    echo "640 640" ;;
    esac
}

v4_variant_repo_config() {
    # Path to the upstream DEIMv2 yaml config the variant inherits from.
    local variant="$1"
    case "$variant" in
        deimv2_atto)  echo "configs/deimv2/deimv2_hgnetv2_atto_coco.yml" ;;
        deimv2_femto) echo "configs/deimv2/deimv2_hgnetv2_femto_coco.yml" ;;
        deimv2_pico)  echo "configs/deimv2/deimv2_hgnetv2_pico_coco.yml" ;;
        deimv2_n)     echo "configs/deimv2/deimv2_hgnetv2_n_coco.yml" ;;
        deimv2_s)     echo "configs/deimv2/deimv2_dinov3_s_coco.yml" ;;
        deimv2_m)     echo "configs/deimv2/deimv2_dinov3_m_coco.yml" ;;
        *) echo "Unknown variant: $variant" >&2; return 1 ;;
    esac
}

v4_variant_init_ckpt_url() {
    # Google-Drive ID for the per-variant pretrained COCO detector.
    local variant="$1"
    case "$variant" in
        deimv2_atto)  echo "18sRJXX3FBUigmGJ1y5Oo_DPC5C3JCgYc" ;;
        deimv2_femto) echo "16hh6l9Oln9TJng4V0_HNf_Z7uYb7feds" ;;
        deimv2_pico)  echo "1PXpUxYSnQO-zJHtzrCPqQZ3KKatZwzFT" ;;
        deimv2_n)     echo "1G_Q80EVO4T7LZVPfHwZ3sT65FX5egp9K" ;;
        deimv2_s)     echo "1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL" ;;
        deimv2_m)     echo "1nPKDHrotusQ748O1cQXJfi5wdShq6bKp" ;;
        *) return 1 ;;
    esac
}

v4_print_env() {
    printf '  %-32s %s\n' "SHITSPOTTER_DPATH" "$SHITSPOTTER_DPATH"
    printf '  %-32s %s\n' "V4_DEV_DPATH"      "$V4_DEV_DPATH"
    printf '  %-32s %s\n' "V4_ROOT"           "$V4_ROOT"
    printf '  %-32s %s\n' "DVC_DATA_DPATH"    "$DVC_DATA_DPATH"
    printf '  %-32s %s\n' "DVC_EXPT_DPATH_RO" "$DVC_EXPT_DPATH_RO"
    printf '  %-32s %s\n' "V4_TRAIN_FPATH"    "$V4_TRAIN_FPATH"
    printf '  %-32s %s\n' "V4_VALI_FPATH"     "$V4_VALI_FPATH"
    printf '  %-32s %s\n' "V4_TEST_FPATH"     "$V4_TEST_FPATH"
    printf '  %-32s %s\n' "V9_PACKAGE_FPATH"  "$V9_PACKAGE_FPATH"
    printf '  %-32s %s\n' "DEIMV2_REPO"       "$SHITSPOTTER_DEIMV2_REPO_DPATH"
}
