#!/bin/bash
set -euo pipefail

# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TRAIN_FPATH="${TRAIN_FPATH:-${DVC_DATA_DPATH:?Set DVC_DATA_DPATH or install geowatch_dvc}/train.kwcoco.zip}"
VALI_FPATH="${VALI_FPATH:-${DVC_DATA_DPATH:?Set DVC_DATA_DPATH or install geowatch_dvc}/vali.kwcoco.zip}"
WORKDIR="${WORKDIR:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/training/$HOSTNAME/$USER/ShitSpotter/runs/foundation_detseg_v3/maskdino_r50}"
MASKDINO_INIT_CKPT="${MASKDINO_INIT_CKPT:-}"

ARGS=(
    python -m shitspotter.algo_foundation_v3.cli_train baseline-maskdino
    --train_kwcoco "$TRAIN_FPATH"
    --vali_kwcoco "$VALI_FPATH"
    --workdir "$WORKDIR"
    --variant maskdino_r50
)

if [ -n "$MASKDINO_INIT_CKPT" ]; then
    ARGS+=(--init_checkpoint_fpath "$MASKDINO_INIT_CKPT")
fi

"${ARGS[@]}"
