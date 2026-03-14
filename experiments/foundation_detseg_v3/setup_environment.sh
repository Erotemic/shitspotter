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

install_deimv2_requirements_without_torch_pins() {
    local req_fpath="$1"
    local filtered_req_fpath
    filtered_req_fpath="$(mktemp)"
    grep -vE '^(torch|torchvision)([[:space:]]*[<>=!~].*)?$' "$req_fpath" > "$filtered_req_fpath"
    python -m pip install -r "$filtered_req_fpath"
    rm -f "$filtered_req_fpath"
}

install_maskdino_requirements_preserve_opencv_stack() {
    local req_fpath="$1"
    local filtered_req_fpath
    filtered_req_fpath="$(mktemp)"
    grep -vE '^opencv-python([[:space:]]*[<>=!~].*)?$' "$req_fpath" > "$filtered_req_fpath"
    python -m pip install -r "$filtered_req_fpath"
    rm -f "$filtered_req_fpath"
    if ! python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("cv2") is not None else 1)
PY
    then
        python -m pip install opencv-python-headless
    fi
}

python -m pip install -r "$FOUNDATION_V3_ROOT_DIR/requirements/runtime.txt" -r "$FOUNDATION_V3_ROOT_DIR/requirements/tests.txt"
python -m pip install -e "$FOUNDATION_V3_ROOT_DIR"
python -m pip install kwcoco kwimage kwutil huggingface_hub gdown pycocotools

install_deimv2_requirements_without_torch_pins "$SHITSPOTTER_DEIMV2_REPO_DPATH/requirements.txt"
python -m pip install -e "$SHITSPOTTER_SAM2_REPO_DPATH"
python -m pip install tensordict submitit iopath fvcore pandas scikit-image tensorboard
install_maskdino_requirements_preserve_opencv_stack "$SHITSPOTTER_MASKDINO_REPO_DPATH/requirements.txt"

cat <<EOF
Environment setup complete.

Repo paths:
  SHITSPOTTER_DEIMV2_REPO_DPATH=$SHITSPOTTER_DEIMV2_REPO_DPATH
  SHITSPOTTER_SAM2_REPO_DPATH=$SHITSPOTTER_SAM2_REPO_DPATH
  SHITSPOTTER_MASKDINO_REPO_DPATH=$SHITSPOTTER_MASKDINO_REPO_DPATH

Notes:
  - Default external repos now live under $FOUNDATION_V3_ROOT_DIR/tpl as git submodules.
  - DEIMv2 upstream pins torch==2.5.1 and torchvision==0.20.1, but this setup script intentionally preserves your existing torch stack and only installs the other DEIMv2 deps.
  - MaskDINO upstream lists opencv-python, but this setup script preserves your existing cv2 provider and only installs opencv-python-headless if cv2 is missing entirely.
  - MaskDINO still requires a compatible Detectron2 install and its CUDA ops.
  - SAM2 may require a recent torch build and optional CUDA extension support.
  - SAM2 fine-tuning also needs training-side extras such as tensordict, submitit, iopath, and fvcore; this setup script now installs them explicitly.
  - Run experiments/foundation_detseg_v3/download_foundation_assets.sh next if you want the default DEIMv2 and SAM2 weights placed in the expected locations.
  - geowatch aggregation may require extra GDAL/OSGeo system packages.
EOF
