#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DVC_DATA_DPATH="${DVC_DATA_DPATH:-$(geowatch_dvc --tags="shitspotter_data")}"
DVC_EXPT_DPATH="${DVC_EXPT_DPATH:-$(geowatch_dvc --tags="shitspotter_expt")}"
TEST_FPATH="${TEST_FPATH:-$DVC_DATA_DPATH/test.kwcoco.zip}"
EVAL_PATH="${EVAL_PATH:-$DVC_EXPT_DPATH/_foundation_detseg_v3/test}"
PACKAGE_FPATH="${PACKAGE_FPATH:-$HERE/packages/deimv2_sam2_default.yaml}"

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.foundation_v3_evaluation_pipeline()'
        matrix:
            foundation_v3_pred.src:
                - '$TEST_FPATH'
            foundation_v3_pred.package_fpath:
                - '$PACKAGE_FPATH'
            foundation_v3_pred.create_labelme: 0
            detection_evaluation.__enabled__: 1
    " \
    --root_dpath="$EVAL_PATH" \
    --devices="0," --tmux_workers=1 \
    --backend=tmux --skip_existing=1 \
    --run=1
