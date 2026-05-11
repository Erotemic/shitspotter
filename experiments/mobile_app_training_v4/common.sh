#!/bin/bash
# Shared environment + helpers for the mobile_app_training_v4 workflow.
#
# All scripts source this file. It is cwd-independent: it derives the v4
# directory from BASH_SOURCE when possible, and falls back to
# $HOME/code/shitspotter/experiments/mobile_app_training_v4 when pasted into an
# interactive bash shell.
#
# Stage every derived artifact under $V4_ROOT (read-write). The DVC roots
# under /data/joncrall/dvc-repos/* are read-only — never write there.

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate this script and the shitspotter repo
# ---------------------------------------------------------------------------
_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
unset _v4_source

export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$(cd "$_v4_script_dpath/../.." && pwd)}"
export V4_DEV_DPATH="$_v4_script_dpath"
unset _v4_script_dpath

# ---------------------------------------------------------------------------
# Data + experiment roots
#
# The DVC roots are read-only. All v4-generated artifacts live under
# $V4_ROOT, which defaults to a writable scratch location off the user
# home so we never trigger DVC writes.
# ---------------------------------------------------------------------------
export DVC_DATA_DPATH="${DVC_DATA_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}"
export DVC_EXPT_DPATH_RO="${DVC_EXPT_DPATH_RO:-/data/joncrall/dvc-repos/shitspotter_expt_dvc}"

# Writable workspace. Override with V4_ROOT if you have more disk elsewhere.
export V4_ROOT="${V4_ROOT:-${HOME}/data/shitspotter_v4}"

# Canonical splits — locked to the v9 split so results are directly
# comparable to the v9 OpenGroundingDINO teacher.
export V4_TRAIN_FPATH="${V4_TRAIN_FPATH:-$DVC_DATA_DPATH/train_imgs10671_b277c63d.kwcoco.zip}"
export V4_VALI_FPATH="${V4_VALI_FPATH:-$DVC_DATA_DPATH/vali_imgs1258_577e331c.kwcoco.zip}"
export V4_TEST_FPATH="${V4_TEST_FPATH:-$DVC_DATA_DPATH/test_imgs121_d39956b1.kwcoco.zip}"

# ---------------------------------------------------------------------------
# Upstream submodule paths
# ---------------------------------------------------------------------------
export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/DEIMv2}"
export SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH="${SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH:-$SHITSPOTTER_DPATH/tpl/Open-GroundingDino}"

# ---------------------------------------------------------------------------
# Teacher (v9 OpenGroundingDINO + tuned SAM2)
#
# These are produced by experiments/foundation_detseg_v3/v9_*.sh. They are
# read-only inputs to the v4 distillation step.
# ---------------------------------------------------------------------------
export V9_PACKAGE_FPATH="${V9_PACKAGE_FPATH:-$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml}"
export V9_SELECTED_MANIFEST_FPATH="${V9_SELECTED_MANIFEST_FPATH:-$DVC_EXPT_DPATH_RO/foundation_detseg_v3/v9/selected_detector_checkpoint.yaml}"

# ---------------------------------------------------------------------------
# Common knobs
#
# RESIZE_MAX_DIM is the long-side max for the pre-resized training images.
# 640 matches the existing foundation v3 path; 1280 if you want the tile
# step to slice from full-resolution detail.
# ---------------------------------------------------------------------------
export V4_RESIZE_MAX_DIM="${V4_RESIZE_MAX_DIM:-1280}"
export V4_RESIZE_OUTPUT_EXT="${V4_RESIZE_OUTPUT_EXT:-.jpg}"
export V4_SIMPLIFY_MIN_INSTANCES="${V4_SIMPLIFY_MIN_INSTANCES:-100}"
export V4_TILE_GRID="${V4_TILE_GRID:-2}"          # NxN grid of overlapping tiles
export V4_TILE_OVERLAP="${V4_TILE_OVERLAP:-0.20}" # fraction overlap between adjacent tiles
export V4_TILE_OUTPUT_DIM="${V4_TILE_OUTPUT_DIM:-640}"
export V4_TILE_KEEP_FULL="${V4_TILE_KEEP_FULL:-1}" # keep the original full image too

# Default short-list of variants to walk through. Override per-script.
export V4_DEFAULT_VARIANTS="${V4_DEFAULT_VARIANTS:-deimv2_n deimv2_pico deimv2_s}"

# ---------------------------------------------------------------------------
# Python entrypoint
# ---------------------------------------------------------------------------
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$SHITSPOTTER_DPATH${PYTHONPATH:+:$PYTHONPATH}"
# DEIMv2 train.py needs to be importable
export PYTHONPATH="$SHITSPOTTER_DEIMV2_REPO_DPATH:$PYTHONPATH"

# Helpful for fine-tuning DEIMv2 backbone init files
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

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
    # Populated for the variants we ship in v4. The download script
    # uses these to bring in pretrained weights so fine-tuning starts
    # from a strong COCO baseline rather than random init.
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

mkdir -p "$V4_ROOT"
