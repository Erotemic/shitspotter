#!/bin/bash
set -euo pipefail

# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_checkout() {
    local repo_dpath="$1"
    local submodule_relpath="$2"
    local repo_url="$3"
    if [ -d "$repo_dpath" ]; then
        return 0
    fi
    mkdir -p "$(dirname "$repo_dpath")"
    if [ -d "$FOUNDATION_V3_ROOT_DIR/.git" ] && [ "$repo_dpath" = "$FOUNDATION_V3_ROOT_DIR/$submodule_relpath" ]; then
        git -C "$FOUNDATION_V3_ROOT_DIR" submodule update --init --recursive --depth 1 "$submodule_relpath"
    fi
    if [ ! -d "$repo_dpath" ]; then
        git clone --depth 1 "$repo_url" "$repo_dpath"
    fi
}

ensure_checkout "$SHITSPOTTER_DEIMV2_REPO_DPATH" "tpl/DEIMv2" "https://github.com/Intellindust-AI-Lab/DEIMv2.git"
ensure_checkout "$SHITSPOTTER_SAM2_REPO_DPATH" "tpl/segment-anything-2" "https://github.com/fal-ai/segment-anything-2.git"
ensure_checkout "$SHITSPOTTER_MASKDINO_REPO_DPATH" "tpl/MaskDINO" "https://github.com/IDEA-Research/MaskDINO.git"

python -m pip install -r "$FOUNDATION_V3_ROOT_DIR/requirements/runtime.txt" -r "$FOUNDATION_V3_ROOT_DIR/requirements/tests.txt"
python -m pip install -e "$FOUNDATION_V3_ROOT_DIR"
python -m pip install kwcoco kwimage kwutil

python -m pip install -r "$SHITSPOTTER_DEIMV2_REPO_DPATH/requirements.txt"
python -m pip install -e "$SHITSPOTTER_SAM2_REPO_DPATH"
python -m pip install -r "$SHITSPOTTER_MASKDINO_REPO_DPATH/requirements.txt"

cat <<EOF
Environment setup complete.

Repo paths:
  SHITSPOTTER_DEIMV2_REPO_DPATH=$SHITSPOTTER_DEIMV2_REPO_DPATH
  SHITSPOTTER_SAM2_REPO_DPATH=$SHITSPOTTER_SAM2_REPO_DPATH
  SHITSPOTTER_MASKDINO_REPO_DPATH=$SHITSPOTTER_MASKDINO_REPO_DPATH

Notes:
  - Default external repos now live under $FOUNDATION_V3_ROOT_DIR/tpl as git submodules.
  - MaskDINO still requires a compatible Detectron2 install and its CUDA ops.
  - SAM2 may require a recent torch build and optional CUDA extension support.
  - geowatch aggregation may require extra GDAL/OSGeo system packages.
EOF
