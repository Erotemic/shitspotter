#!/bin/bash
set -euo pipefail

DVC_DATA_DPATH="${DVC_DATA_DPATH:-$(geowatch_dvc --tags="shitspotter_data")}"
DVC_EXPT_DPATH="${DVC_EXPT_DPATH:-$(geowatch_dvc --tags="shitspotter_expt")}"

TRAIN_FPATH="${TRAIN_FPATH:-$DVC_DATA_DPATH/train.kwcoco.zip}"
VALI_FPATH="${VALI_FPATH:-$DVC_DATA_DPATH/vali.kwcoco.zip}"
WORKDIR="${WORKDIR:-$DVC_EXPT_DPATH/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m}"
VARIANT="${VARIANT:-deimv2_m}"
DEIMV2_INIT_CKPT="${DEIMV2_INIT_CKPT:-}"

ARGS=(
    python -m shitspotter.algo_foundation_v3.cli_train detector
    --train_kwcoco "$TRAIN_FPATH"
    --vali_kwcoco "$VALI_FPATH"
    --workdir "$WORKDIR"
    --variant "$VARIANT"
)

if [ -n "$DEIMV2_INIT_CKPT" ]; then
    ARGS+=(--init_checkpoint_fpath "$DEIMV2_INIT_CKPT")
fi

"${ARGS[@]}"
