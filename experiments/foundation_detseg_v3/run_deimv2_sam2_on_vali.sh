#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DVC_DATA_DPATH="${DVC_DATA_DPATH:-$(geowatch_dvc --tags="shitspotter_data")}"
DVC_EXPT_DPATH="${DVC_EXPT_DPATH:-$(geowatch_dvc --tags="shitspotter_expt")}"
VALI_FPATH="${VALI_FPATH:-$DVC_DATA_DPATH/vali.kwcoco.zip}"
EVAL_PATH="${EVAL_PATH:-$DVC_EXPT_DPATH/_foundation_detseg_v3/vali}"

PACKAGE_FPATH="${PACKAGE_FPATH:-$HERE/packages/deimv2_sam2_default.yaml}"
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

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.foundation_v3_evaluation_pipeline()'
        matrix:
            foundation_v3_pred.src:
                - '$VALI_FPATH'
            foundation_v3_pred.package_fpath:
                - '$PACKAGE_FPATH'
            foundation_v3_pred.create_labelme: 0
            detection_evaluation.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0," --tmux_workers=1 \
    --backend=tmux --skip_existing=1 \
    --run=1
