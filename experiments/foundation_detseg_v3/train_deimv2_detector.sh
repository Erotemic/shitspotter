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
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
DEIMV2_CONFIG_OVERRIDES="${DEIMV2_CONFIG_OVERRIDES:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

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
