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
SAM2_WORKDIR="${SAM2_WORKDIR:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/sam2.1_hiera_base_plus}"
SAM2_VARIANT="${SAM2_VARIANT:-sam2.1_hiera_base_plus}"
SAM2_INIT_CKPT="${SAM2_INIT_CKPT:-$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt}"
SAM2_PACKAGE_OUT="${SAM2_PACKAGE_OUT:-}"
SAM2_METADATA_NAME="${SAM2_METADATA_NAME:-foundation_v3_${SAM2_VARIANT}_tuned}"
DEIMV2_TRAINED_CKPT="${DEIMV2_TRAINED_CKPT:-}"
SAM2_RESOLUTION="${SAM2_RESOLUTION:-1024}"
SAM2_TRAIN_BATCH_SIZE="${SAM2_TRAIN_BATCH_SIZE:-1}"
SAM2_NUM_TRAIN_WORKERS="${SAM2_NUM_TRAIN_WORKERS:-8}"
SAM2_NUM_EPOCHS="${SAM2_NUM_EPOCHS:-20}"
SAM2_NUM_GPUS="${SAM2_NUM_GPUS:-1}"
SAM2_BASE_LR="${SAM2_BASE_LR:-0.000005}"
SAM2_VISION_LR="${SAM2_VISION_LR:-0.000003}"
SAM2_MAX_NUM_OBJECTS="${SAM2_MAX_NUM_OBJECTS:-8}"
SAM2_MULTIPLIER="${SAM2_MULTIPLIER:-1}"
SAM2_CHECKPOINT_SAVE_FREQ="${SAM2_CHECKPOINT_SAVE_FREQ:-1}"
SAM2_CATEGORY_NAMES="${SAM2_CATEGORY_NAMES:-}"
SAM2_CONFIG_OVERRIDES="${SAM2_CONFIG_OVERRIDES:-}"

ARGS=(
    python -m shitspotter.algo_foundation_v3.cli_train segmenter
    --train_kwcoco "$TRAIN_FPATH"
    --vali_kwcoco "$VALI_FPATH"
    --workdir "$SAM2_WORKDIR"
    --variant "$SAM2_VARIANT"
    --checkpoint_fpath "$SAM2_INIT_CKPT"
    --metadata_name "$SAM2_METADATA_NAME"
    --resolution "$SAM2_RESOLUTION"
    --train_batch_size "$SAM2_TRAIN_BATCH_SIZE"
    --num_train_workers "$SAM2_NUM_TRAIN_WORKERS"
    --num_epochs "$SAM2_NUM_EPOCHS"
    --num_gpus "$SAM2_NUM_GPUS"
    --base_lr "$SAM2_BASE_LR"
    --vision_lr "$SAM2_VISION_LR"
    --max_num_objects "$SAM2_MAX_NUM_OBJECTS"
    --multiplier "$SAM2_MULTIPLIER"
    --checkpoint_save_freq "$SAM2_CHECKPOINT_SAVE_FREQ"
)

if [ -n "$SAM2_CATEGORY_NAMES" ]; then
    ARGS+=(--category_names "$SAM2_CATEGORY_NAMES")
fi

if [ -n "$SAM2_CONFIG_OVERRIDES" ]; then
    ARGS+=(--config_overrides "$SAM2_CONFIG_OVERRIDES")
fi

if [ -n "$SAM2_PACKAGE_OUT" ]; then
    ARGS+=(--package_out "$SAM2_PACKAGE_OUT")
fi

if [ -n "$DEIMV2_TRAINED_CKPT" ]; then
    ARGS+=(--detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT")
fi

"${ARGS[@]}"
