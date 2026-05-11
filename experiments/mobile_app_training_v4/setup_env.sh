# shellcheck shell=bash
# Source this file once per interactive shell to set up every env var the
# mobile_app_training_v4 scripts read. Safe to source repeatedly.
#
# Usage:
#     source experiments/mobile_app_training_v4/setup_env.sh
#
# Then any of the 0X_*.sh / 0X_*.py scripts can be invoked without
# re-exporting paths.
#
# Override anything by exporting it BEFORE sourcing this file. Example:
#     V4_ROOT=/scratch/v4 source experiments/mobile_app_training_v4/setup_env.sh
#
# Unlike common.sh, this file does not enable `set -euo pipefail`, so it
# is safe to source into an interactive shell.

# ---------------------------------------------------------------------------
# Locate the v4 dev dir and the shitspotter repo
# ---------------------------------------------------------------------------
_v4_setup_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_setup_source" ] && [ "$_v4_setup_source" != "bash" ] && [ "$_v4_setup_source" != "-bash" ]; then
    _v4_setup_dpath="$(cd "$(dirname "$_v4_setup_source")" && pwd)"
else
    _v4_setup_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
unset _v4_setup_source

export V4_DEV_DPATH="$_v4_setup_dpath"
export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$(cd "$_v4_setup_dpath/../.." && pwd)}"
unset _v4_setup_dpath

# ---------------------------------------------------------------------------
# Data + experiment roots (DVC roots are read-only)
# ---------------------------------------------------------------------------
export DVC_DATA_DPATH="${DVC_DATA_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}"
export DVC_EXPT_DPATH_RO="${DVC_EXPT_DPATH_RO:-/data/joncrall/dvc-repos/shitspotter_expt_dvc}"

# Writable workspace for v4 artifacts.
export V4_ROOT="${V4_ROOT:-${HOME}/data/shitspotter_v4}"

# ---------------------------------------------------------------------------
# Canonical splits — locked to the v9 split for direct comparability.
# ---------------------------------------------------------------------------
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
# ---------------------------------------------------------------------------
export V9_PACKAGE_FPATH="${V9_PACKAGE_FPATH:-$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3/packages/v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml}"
export V9_SELECTED_MANIFEST_FPATH="${V9_SELECTED_MANIFEST_FPATH:-$DVC_EXPT_DPATH_RO/foundation_detseg_v3/v9/selected_detector_checkpoint.yaml}"

# ---------------------------------------------------------------------------
# Common knobs — safe defaults, override before sourcing if desired.
# ---------------------------------------------------------------------------
export V4_RESIZE_MAX_DIM="${V4_RESIZE_MAX_DIM:-1280}"
export V4_RESIZE_OUTPUT_EXT="${V4_RESIZE_OUTPUT_EXT:-.jpg}"
export V4_SIMPLIFY_MIN_INSTANCES="${V4_SIMPLIFY_MIN_INSTANCES:-100}"
export V4_TILE_GRID="${V4_TILE_GRID:-2}"
export V4_TILE_OVERLAP="${V4_TILE_OVERLAP:-0.20}"
export V4_TILE_OUTPUT_DIM="${V4_TILE_OUTPUT_DIM:-640}"
export V4_TILE_KEEP_FULL="${V4_TILE_KEEP_FULL:-1}"
export V4_DEFAULT_VARIANTS="${V4_DEFAULT_VARIANTS:-deimv2_n deimv2_pico deimv2_s}"

# ---------------------------------------------------------------------------
# Python entrypoint
# ---------------------------------------------------------------------------
export PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="$SHITSPOTTER_DPATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH="$SHITSPOTTER_DEIMV2_REPO_DPATH:$PYTHONPATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

# Pin to GPU 0 by default. The host has 2x 3090s but GPU 1 sits on a
# 2x PCIe link, so multi-GPU all-reduce gets bottlenecked by the slow
# peer and ends up slower than single-GPU on GPU 0. Override with
# CUDA_VISIBLE_DEVICES=0,1 (and V4_NUM_GPUS=2) only if you have a
# bandwidth-matched setup. Override with CUDA_VISIBLE_DEVICES=1 to
# move the single-GPU run to the other card.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$V4_ROOT"

# Print a concise summary so the user can confirm what got set.
echo "mobile_app_training_v4 environment ready"
echo "  V4_ROOT             = $V4_ROOT"
echo "  V4_DEV_DPATH        = $V4_DEV_DPATH"
echo "  SHITSPOTTER_DPATH   = $SHITSPOTTER_DPATH"
echo "  DVC_DATA_DPATH      = $DVC_DATA_DPATH (read-only)"
echo "  V4_TRAIN_FPATH      = $V4_TRAIN_FPATH"
echo "  PYTHON_BIN          = $PYTHON_BIN"
