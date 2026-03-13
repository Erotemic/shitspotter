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
WORKDIR="${WORKDIR:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/sam2.1_hiera_base_plus}"
VARIANT="${VARIANT:-sam2.1_hiera_base_plus}"
SAM2_INIT_CKPT="${SAM2_INIT_CKPT:-$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt}"
PACKAGE_OUT="${PACKAGE_OUT:-}"
METADATA_NAME="${METADATA_NAME:-foundation_v3_${VARIANT}_tuned}"
DEIMV2_TRAINED_CKPT="${DEIMV2_TRAINED_CKPT:-}"
RESOLUTION="${RESOLUTION:-1024}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
NUM_TRAIN_WORKERS="${NUM_TRAIN_WORKERS:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
NUM_GPUS="${NUM_GPUS:-1}"
BASE_LR="${BASE_LR:-0.000005}"
VISION_LR="${VISION_LR:-0.000003}"
MAX_NUM_OBJECTS="${MAX_NUM_OBJECTS:-8}"
MULTIPLIER="${MULTIPLIER:-1}"
CHECKPOINT_SAVE_FREQ="${CHECKPOINT_SAVE_FREQ:-1}"
CATEGORY_NAMES="${CATEGORY_NAMES:-}"
SAM2_CONFIG_OVERRIDES="${SAM2_CONFIG_OVERRIDES:-}"

ARGS=(
    python -m shitspotter.algo_foundation_v3.cli_train segmenter
    --train_kwcoco "$TRAIN_FPATH"
    --vali_kwcoco "$VALI_FPATH"
    --workdir "$WORKDIR"
    --variant "$VARIANT"
    --checkpoint_fpath "$SAM2_INIT_CKPT"
    --metadata_name "$METADATA_NAME"
    --resolution "$RESOLUTION"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --num_train_workers "$NUM_TRAIN_WORKERS"
    --num_epochs "$NUM_EPOCHS"
    --num_gpus "$NUM_GPUS"
    --base_lr "$BASE_LR"
    --vision_lr "$VISION_LR"
    --max_num_objects "$MAX_NUM_OBJECTS"
    --multiplier "$MULTIPLIER"
    --checkpoint_save_freq "$CHECKPOINT_SAVE_FREQ"
)

if [ -n "$CATEGORY_NAMES" ]; then
    ARGS+=(--category_names "$CATEGORY_NAMES")
fi

if [ -n "$SAM2_CONFIG_OVERRIDES" ]; then
    ARGS+=(--config_overrides "$SAM2_CONFIG_OVERRIDES")
fi

if [ -n "$PACKAGE_OUT" ]; then
    ARGS+=(--package_out "$PACKAGE_OUT")
fi

if [ -n "$DEIMV2_TRAINED_CKPT" ]; then
    ARGS+=(--detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT")
fi

"${ARGS[@]}"
