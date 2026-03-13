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

COHORT_DPATH="${COHORT_DPATH:?Set COHORT_DPATH to a directory of phone images}"
PACKAGE_FPATH="${PACKAGE_FPATH:?Set PACKAGE_FPATH to a foundation model package yaml}"

python -m shitspotter.algo_foundation_v3.cli_predict \
    --src "$COHORT_DPATH" \
    --package_fpath "$PACKAGE_FPATH" \
    --create_labelme True
