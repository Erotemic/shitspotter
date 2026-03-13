#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export SHITSPOTTER_DEIMV2_REPO_DPATH="${SHITSPOTTER_DEIMV2_REPO_DPATH:-$HOME/code/DEIMv2}"
export SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$HOME/code/segment-anything-2}"
export SHITSPOTTER_MASKDINO_REPO_DPATH="${SHITSPOTTER_MASKDINO_REPO_DPATH:-$HOME/code/MaskDINO}"

python -m pip install -r "$ROOT_DIR/requirements/runtime.txt" -r "$ROOT_DIR/requirements/tests.txt"
python -m pip install -e "$ROOT_DIR"
python -m pip install kwcoco kwimage kwutil

if [ ! -d "$SHITSPOTTER_DEIMV2_REPO_DPATH" ]; then
    git clone --depth 1 https://github.com/Intellindust-AI-Lab/DEIMv2.git "$SHITSPOTTER_DEIMV2_REPO_DPATH"
fi
if [ ! -d "$SHITSPOTTER_SAM2_REPO_DPATH" ]; then
    git clone --depth 1 https://github.com/fal-ai/segment-anything-2.git "$SHITSPOTTER_SAM2_REPO_DPATH"
fi
if [ ! -d "$SHITSPOTTER_MASKDINO_REPO_DPATH" ]; then
    git clone --depth 1 https://github.com/IDEA-Research/MaskDINO.git "$SHITSPOTTER_MASKDINO_REPO_DPATH"
fi

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
  - MaskDINO still requires a compatible Detectron2 install and its CUDA ops.
  - SAM2 may require a recent torch build and optional CUDA extension support.
  - geowatch aggregation may require extra GDAL/OSGeo system packages.
EOF
