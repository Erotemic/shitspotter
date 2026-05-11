# shellcheck shell=bash
# Source this once per interactive shell to set up every env var the
# mobile_app_training_v5 scripts read.
#
# Usage:
#     source experiments/mobile_app_training_v5/setup_env.sh
#
# v5 shares the heavy deps (DEIMv2, geowatch, onnx trio) with v4, so we
# source v4's setup_env.sh first and then layer the v5-specific knobs
# on top. Override any v4 var by exporting it BEFORE sourcing this.

# ---------------------------------------------------------------------------
# Locate v5 + chain through v4
# ---------------------------------------------------------------------------
_v5_setup_source="${BASH_SOURCE[0]-}"
if [ -n "$_v5_setup_source" ] && [ "$_v5_setup_source" != "bash" ] && [ "$_v5_setup_source" != "-bash" ]; then
    _v5_setup_dpath="$(cd "$(dirname "$_v5_setup_source")" && pwd)"
else
    _v5_setup_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v5"
fi
unset _v5_setup_source

export V5_DEV_DPATH="$_v5_setup_dpath"
unset _v5_setup_dpath

# Chain through v4's setup so we inherit:
#   PYTHON_BIN, PYTHONPATH (idempotent), SHITSPOTTER_DPATH,
#   SHITSPOTTER_DEIMV2_REPO_DPATH, DVC paths, CUDA_VISIBLE_DEVICES, etc.
# shellcheck source=experiments/mobile_app_training_v4/setup_env.sh
source "$V5_DEV_DPATH/../mobile_app_training_v4/setup_env.sh"

# ---------------------------------------------------------------------------
# v5-specific knobs
# ---------------------------------------------------------------------------
export V5_ROOT="${V5_ROOT:-${HOME}/data/shitspotter_v5}"

# Tile extractor defaults
export V5_TILE_SIZE="${V5_TILE_SIZE:-320}"
export V5_SOURCE_SCALES="${V5_SOURCE_SCALES:-1.0,0.66,0.40,0.25}"
export V5_STRIDE_FRAC="${V5_STRIDE_FRAC:-0.5}"
export V5_MIN_GT_AREA_FRAC="${V5_MIN_GT_AREA_FRAC:-0.005}"
export V5_MIN_KEPT_BOX_FRAC="${V5_MIN_KEPT_BOX_FRAC:-0.30}"
export V5_MIN_SOURCE_SCALE_LONG_SIDE="${V5_MIN_SOURCE_SCALE_LONG_SIDE:-64}"
export V5_JPEG_QUALITY="${V5_JPEG_QUALITY:-90}"

# Round loop defaults
export V5_NUM_ROUNDS="${V5_NUM_ROUNDS:-3}"
export V5_ROUND0_NEG_OVER_POS="${V5_ROUND0_NEG_OVER_POS:-3.0}"
export V5_MINE_SCORE_THRESH="${V5_MINE_SCORE_THRESH:-0.30}"
export V5_MAX_HARD_PER_ROUND="${V5_MAX_HARD_PER_ROUND:-5000}"

# Per-round training knobs (forwarded to v4's trainer)
export V5_ROUND_EPOCHS="${V5_ROUND_EPOCHS:-20}"
export V5_VARIANT="${V5_VARIANT:-deimv2_n}"
export V5_INPUT_HW="${V5_INPUT_HW:-320 320}"

mkdir -p "$V5_ROOT"

echo "mobile_app_training_v5 environment ready"
echo "  V5_ROOT             = $V5_ROOT"
echo "  V5_DEV_DPATH        = $V5_DEV_DPATH"
echo "  V5_TILE_SIZE        = $V5_TILE_SIZE"
echo "  V5_SOURCE_SCALES    = $V5_SOURCE_SCALES"
echo "  V5_NUM_ROUNDS       = $V5_NUM_ROUNDS"
echo "  V5_VARIANT          = $V5_VARIANT"
echo "  inherits v4: V4_ROOT=$V4_ROOT, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
