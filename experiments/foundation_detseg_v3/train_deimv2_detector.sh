#!/bin/bash
set -euo pipefail

# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_FPATH="${TRAIN_FPATH:-${DVC_DATA_DPATH:?Set DVC_DATA_DPATH or install geowatch_dvc}/train.kwcoco.zip}"
VALI_FPATH="${VALI_FPATH:-${DVC_DATA_DPATH:?Set DVC_DATA_DPATH or install geowatch_dvc}/vali.kwcoco.zip}"
WORKDIR="${WORKDIR:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/deimv2_m}"
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
