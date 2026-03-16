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
EVAL_PATH="${EVAL_PATH:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/_foundation_detseg_v3/vali}"
PRED_FPATH="${PRED_FPATH:-$EVAL_PATH/pred.kwcoco.zip}"
METRICS_DPATH="${METRICS_DPATH:-$EVAL_PATH/eval}"
METRICS_FPATH="${METRICS_FPATH:-$METRICS_DPATH/detect_metrics.json}"
CONFUSION_FPATH="${CONFUSION_FPATH:-$METRICS_DPATH/confusion.kwcoco.zip}"

PACKAGE_FPATH="${PACKAGE_FPATH:-$FOUNDATION_V3_PACKAGE_DPATH/deimv2_sam2_default.yaml}"
DEIMV2_CKPT="${DEIMV2_CKPT:-}"
SAM2_CKPT="${SAM2_CKPT:-}"

if [ ! -f "$PACKAGE_FPATH" ] && [ -n "$DEIMV2_CKPT" ] && [ -n "$SAM2_CKPT" ]; then
    python -m shitspotter.algo_foundation_v3.cli_package build "$PACKAGE_FPATH" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$DEIMV2_CKPT" \
        --segmenter_checkpoint_fpath "$SAM2_CKPT" \
        --metadata_name deimv2_sam2_vali
fi

mkdir -p "$EVAL_PATH" "$METRICS_DPATH"

python -m shitspotter.algo_foundation_v3.cli_predict \
    --src="$VALI_FPATH" \
    --package_fpath="$PACKAGE_FPATH" \
    --create_labelme=0 \
    --dst="$PRED_FPATH"

python -m kwcoco eval \
    --true_dataset="$VALI_FPATH" \
    --pred_dataset="$PRED_FPATH" \
    --out_dpath="$METRICS_DPATH" \
    --out_fpath="$METRICS_FPATH" \
    --confusion_fpath="$CONFUSION_FPATH" \
    --draw=False \
    --iou_thresh=0.5
