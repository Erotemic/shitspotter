#!/bin/bash
set -euo pipefail

FOUNDATION_V3_DEV_DPATH="${FOUNDATION_V3_DEV_DPATH:-${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}/experiments/foundation_detseg_v3}"
_foundation_v3_source="${BASH_SOURCE[0]-}"
if [ -n "$_foundation_v3_source" ] && [ "$_foundation_v3_source" != "bash" ] && [ "$_foundation_v3_source" != "-bash" ]; then
    _foundation_v3_script_dpath="$(cd "$(dirname "$_foundation_v3_source")" && pwd)"
else
    _foundation_v3_script_dpath="$FOUNDATION_V3_DEV_DPATH"
fi
# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$_foundation_v3_script_dpath/common.sh"
unset _foundation_v3_source
unset _foundation_v3_script_dpath

TRAIN_FPATH="${TRAIN_FPATH:-${FOUNDATION_V3_TRAIN_KWCOCO_FPATH:?Set FOUNDATION_V3_TRAIN_KWCOCO_FPATH or install geowatch_dvc}}"
VALI_FPATH="${VALI_FPATH:-${FOUNDATION_V3_VALI_KWCOCO_FPATH:?Set FOUNDATION_V3_VALI_KWCOCO_FPATH or install geowatch_dvc}}"
WORKDIR="${WORKDIR:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m}"
VARIANT="${VARIANT:-deimv2_m}"
DEIMV2_INIT_CKPT="${DEIMV2_INIT_CKPT:-}"
USE_AMP="${USE_AMP:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
DEIMV2_CONFIG_OVERRIDES="${DEIMV2_CONFIG_OVERRIDES:-}"
ENABLE_RESIZE_PREPROCESS="${ENABLE_RESIZE_PREPROCESS:-True}"
FORCE_RESIZE_PREPROCESS="${FORCE_RESIZE_PREPROCESS:-False}"
RESIZE_MAX_DIM="${RESIZE_MAX_DIM:-640}"
RESIZE_OUTPUT_EXT="${RESIZE_OUTPUT_EXT:-.jpg}"
ENABLE_SIMPLIFY_PREPROCESS="${ENABLE_SIMPLIFY_PREPROCESS:-False}"
FORCE_SIMPLIFY_PREPROCESS="${FORCE_SIMPLIFY_PREPROCESS:-False}"
SIMPLIFY_MINIMUM_INSTANCES="${SIMPLIFY_MINIMUM_INSTANCES:-100}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

_foundation_v3_truthy() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if _foundation_v3_truthy "$ENABLE_RESIZE_PREPROCESS"; then
    PREPROC_DPATH="$WORKDIR/preprocessed_kwcoco"
    mkdir -p "$PREPROC_DPATH"
    PREPROC_TRAIN_FPATH="$PREPROC_DPATH/train_maxdim${RESIZE_MAX_DIM}.kwcoco.zip"
    PREPROC_VALI_FPATH="$PREPROC_DPATH/vali_maxdim${RESIZE_MAX_DIM}.kwcoco.zip"

    if _foundation_v3_truthy "$FORCE_RESIZE_PREPROCESS" || [ ! -f "$PREPROC_TRAIN_FPATH" ]; then
        python -m shitspotter.cli.resize_kwcoco \
            --src "$TRAIN_FPATH" \
            --dst "$PREPROC_TRAIN_FPATH" \
            --max_dim "$RESIZE_MAX_DIM" \
            --asset_dname "train_assets_maxdim${RESIZE_MAX_DIM}" \
            --output_ext "$RESIZE_OUTPUT_EXT"
    fi

    if _foundation_v3_truthy "$FORCE_RESIZE_PREPROCESS" || [ ! -f "$PREPROC_VALI_FPATH" ]; then
        python -m shitspotter.cli.resize_kwcoco \
            --src "$VALI_FPATH" \
            --dst "$PREPROC_VALI_FPATH" \
            --max_dim "$RESIZE_MAX_DIM" \
            --asset_dname "vali_assets_maxdim${RESIZE_MAX_DIM}" \
            --output_ext "$RESIZE_OUTPUT_EXT"
    fi

    TRAIN_FPATH="$PREPROC_TRAIN_FPATH"
    VALI_FPATH="$PREPROC_VALI_FPATH"
fi

if _foundation_v3_truthy "$ENABLE_SIMPLIFY_PREPROCESS"; then
    PREPROC_DPATH="$WORKDIR/preprocessed_kwcoco"
    mkdir -p "$PREPROC_DPATH"
    SIMPLIFY_TRAIN_FPATH="$PREPROC_DPATH/$(basename "${TRAIN_FPATH%.*}").simplified.kwcoco.zip"
    SIMPLIFY_VALI_FPATH="$PREPROC_DPATH/$(basename "${VALI_FPATH%.*}").simplified.kwcoco.zip"

    if _foundation_v3_truthy "$FORCE_SIMPLIFY_PREPROCESS" || [ ! -f "$SIMPLIFY_TRAIN_FPATH" ]; then
        python -m shitspotter.cli.simplify_kwcoco \
            --src "$TRAIN_FPATH" \
            --dst "$SIMPLIFY_TRAIN_FPATH" \
            --minimum_instances "$SIMPLIFY_MINIMUM_INSTANCES"
    fi

    if _foundation_v3_truthy "$FORCE_SIMPLIFY_PREPROCESS" || [ ! -f "$SIMPLIFY_VALI_FPATH" ]; then
        python -m shitspotter.cli.simplify_kwcoco \
            --src "$VALI_FPATH" \
            --dst "$SIMPLIFY_VALI_FPATH" \
            --minimum_instances "$SIMPLIFY_MINIMUM_INSTANCES"
    fi

    TRAIN_FPATH="$SIMPLIFY_TRAIN_FPATH"
    VALI_FPATH="$SIMPLIFY_VALI_FPATH"
fi

if [ -z "$DEIMV2_CONFIG_OVERRIDES" ]; then
    read -r BACKBONE_LR MAIN_LR <<EOF
$(python - <<PY
train_batch = int("$TRAIN_BATCH_SIZE")
base_batch = 32.0
main_lr = 5e-4 * (train_batch / base_batch)
backbone_lr = 2.5e-5 * (train_batch / base_batch)
print(f"{backbone_lr:.10f} {main_lr:.10f}")
PY
)
EOF
    DEIMV2_CONFIG_OVERRIDES="$(cat <<EOF
use_amp: ${USE_AMP}
train_dataloader:
  total_batch_size: ${TRAIN_BATCH_SIZE}
  num_workers: ${TRAIN_NUM_WORKERS}
val_dataloader:
  total_batch_size: ${VAL_BATCH_SIZE}
  num_workers: ${VAL_NUM_WORKERS}
optimizer:
  params:
    - params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: ${BACKBONE_LR}
    - params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: ${BACKBONE_LR}
      weight_decay: 0.0
    - params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0
  lr: ${MAIN_LR}
EOF
)"
fi

ARGS=(
    python -m shitspotter.algo_foundation_v3.cli_train detector
    --train_kwcoco "$TRAIN_FPATH"
    --vali_kwcoco "$VALI_FPATH"
    --workdir "$WORKDIR"
    --variant "$VARIANT"
    --use_amp "$USE_AMP"
    --config_overrides "$DEIMV2_CONFIG_OVERRIDES"
)

if [ -n "$DEIMV2_INIT_CKPT" ]; then
    ARGS+=(--init_checkpoint_fpath "$DEIMV2_INIT_CKPT")
fi

"${ARGS[@]}"
