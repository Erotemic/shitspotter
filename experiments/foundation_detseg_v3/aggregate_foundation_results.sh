#!/bin/bash
set -euo pipefail

# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_DPATH="${TARGET_DPATH:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/_foundation_detseg_v3}"
OUTPUT_DPATH="${OUTPUT_DPATH:-$TARGET_DPATH/full_aggregate}"

python -m geowatch.mlops.aggregate \
    --pipeline='shitspotter.pipelines.foundation_v3_evaluation_pipeline()' \
    --target "
        - $TARGET_DPATH
    " \
    --output_dpath="$OUTPUT_DPATH" \
    --resource_report=0 \
    --io_workers=0 \
    --eval_nodes="
        - detection_evaluation
    " \
    --stdout_report="
        top_k: 10
        per_group: null
        macro_analysis: 0
        analyze: 0
        print_models: True
        reference_region: null
        concise: 1
        show_csv: 0
    "
