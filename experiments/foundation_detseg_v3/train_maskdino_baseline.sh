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
