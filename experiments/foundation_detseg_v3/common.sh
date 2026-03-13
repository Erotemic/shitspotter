#!/bin/bash
set -euo pipefail

export SHITSPOTTER_DPATH="${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}"
export FOUNDATION_V3_DEV_DPATH="${FOUNDATION_V3_DEV_DPATH:-$SHITSPOTTER_DPATH/experiments/foundation_detseg_v3}"

_foundation_v3_source="${BASH_SOURCE[0]-}"
if [ -n "$_foundation_v3_source" ] && [ "$_foundation_v3_source" != "bash" ] && [ "$_foundation_v3_source" != "-bash" ]; then
    FOUNDATION_V3_SCRIPT_DPATH="$(cd "$(dirname "$_foundation_v3_source")" && pwd)"
else
    FOUNDATION_V3_SCRIPT_DPATH="$FOUNDATION_V3_DEV_DPATH"
fi
FOUNDATION_V3_ROOT_DIR="$(cd "$FOUNDATION_V3_SCRIPT_DPATH/../.." && pwd)"
FOUNDATION_V3_PACKAGE_DPATH="$FOUNDATION_V3_SCRIPT_DPATH/packages"
unset _foundation_v3_source

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

if [ -n "${DVC_DATA_DPATH:-}" ]; then
    export FOUNDATION_V3_MODEL_DPATH="${FOUNDATION_V3_MODEL_DPATH:-$DVC_DATA_DPATH/models/foundation_detseg_v3}"
else
    export FOUNDATION_V3_MODEL_DPATH="${FOUNDATION_V3_MODEL_DPATH:-$FOUNDATION_V3_ROOT_DIR/.cache/foundation_detseg_v3}"
fi
