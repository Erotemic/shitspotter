#!/bin/bash
set -euo pipefail

COHORT_DPATH="${COHORT_DPATH:?Set COHORT_DPATH to a directory of phone images}"
PACKAGE_FPATH="${PACKAGE_FPATH:?Set PACKAGE_FPATH to a foundation model package yaml}"

python -m shitspotter.algo_foundation_v3.cli_predict \
    --src "$COHORT_DPATH" \
    --package_fpath "$PACKAGE_FPATH" \
    --create_labelme True
