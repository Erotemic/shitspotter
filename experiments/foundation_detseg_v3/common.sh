#!/bin/bash
set -euo pipefail

FOUNDATION_V3_SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOUNDATION_V3_ROOT_DIR="$(cd "$FOUNDATION_V3_SCRIPT_DPATH/../.." && pwd)"
FOUNDATION_V3_PACKAGE_DPATH="$FOUNDATION_V3_SCRIPT_DPATH/packages"

export FOUNDATION_V3_SCRIPT_DPATH
export FOUNDATION_V3_ROOT_DIR
export FOUNDATION_V3_PACKAGE_DPATH

export PYTHONPATH="$FOUNDATION_V3_ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$FOUNDATION_V3_ROOT_DIR/tpl/DEIMv2}"
export SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$FOUNDATION_V3_ROOT_DIR/tpl/segment-anything-2}"
export SHITSPOTTER_MASKDINO_REPO_DPATH="${SHITSPOTTER_MASKDINO_REPO_DPATH:-$FOUNDATION_V3_ROOT_DIR/tpl/MaskDINO}"

if command -v geowatch_dvc >/dev/null 2>&1; then
    if [ -z "${DVC_DATA_DPATH:-}" ] && _dvc_data="$(geowatch_dvc --tags="shitspotter_data" 2>/dev/null)"; then
        export DVC_DATA_DPATH="$_dvc_data"
    fi
    if [ -z "${DVC_EXPT_DPATH:-}" ] && _dvc_expt="$(geowatch_dvc --tags="shitspotter_expt" 2>/dev/null)"; then
        export DVC_EXPT_DPATH="$_dvc_expt"
    fi
fi
