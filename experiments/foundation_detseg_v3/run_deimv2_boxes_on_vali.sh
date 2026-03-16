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

VALI_FPATH="${VALI_FPATH:-${FOUNDATION_V3_VALI_KWCOCO_FPATH:?Set FOUNDATION_V3_VALI_KWCOCO_FPATH or install geowatch_dvc}}"
EVAL_PATH="${EVAL_PATH:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/_foundation_detseg_v3/vali_boxes}"
PRED_FPATH="${PRED_FPATH:-$EVAL_PATH/pred_boxes.kwcoco.zip}"
METRICS_DPATH="${METRICS_DPATH:-$EVAL_PATH/eval}"
METRICS_FPATH="${METRICS_FPATH:-$METRICS_DPATH/detect_metrics.json}"
CONFUSION_FPATH="${CONFUSION_FPATH:-$METRICS_DPATH/confusion.kwcoco.zip}"
PACKAGE_FPATH="${PACKAGE_FPATH:?Set PACKAGE_FPATH to a deimv2_sam2 package yaml}"

mkdir -p "$EVAL_PATH" "$METRICS_DPATH"

python -m shitspotter.algo_foundation_v3.cli_predict_boxes \
    --src="$VALI_FPATH" \
    --package_fpath="$PACKAGE_FPATH" \
    --dst="$PRED_FPATH"

python -m kwcoco eval \
    --true_dataset="$VALI_FPATH" \
    --pred_dataset="$PRED_FPATH" \
    --out_dpath="$METRICS_DPATH" \
    --out_fpath="$METRICS_FPATH" \
    --confusion_fpath="$CONFUSION_FPATH" \
    --draw=False \
    --iou_thresh=0.5
