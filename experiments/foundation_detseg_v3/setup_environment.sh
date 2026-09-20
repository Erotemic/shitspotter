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

ensure_kdk_checkout() {
    local repo_dpath="$1"
    local submodule_relpath="$2"
    if [ -d "$repo_dpath" ]; then
        return 0
    fi
    if [ ! -d "$KWCOCO_DETECTOR_KIT_DPATH/.git" ]; then
        echo "KWCoco Detector Kit checkout not found: $KWCOCO_DETECTOR_KIT_DPATH" >&2
        echo "Set KWCOCO_DETECTOR_KIT_DPATH to the KDK repository root." >&2
        return 1
    fi
    git -C "$KWCOCO_DETECTOR_KIT_DPATH" submodule update --init --recursive "$submodule_relpath"
    if [ ! -d "$repo_dpath" ]; then
        echo "KDK backend checkout was not created: $repo_dpath" >&2
        return 1
    fi
}

python_pkg_install() {
    if command -v uv >/dev/null 2>&1; then
        uv pip install "$@"
    else
        python -m pip install "$@"
    fi
}

ensure_kdk_checkout "$SHITSPOTTER_DEIMV2_REPO_DPATH" "tpl/DEIMv2"
ensure_kdk_checkout "$SHITSPOTTER_SAM2_REPO_DPATH" "tpl/segment-anything-2"
ensure_kdk_checkout "$SHITSPOTTER_MASKDINO_REPO_DPATH" "tpl/MaskDINO"
ensure_kdk_checkout "$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH" "tpl/Open-GroundingDino"

install_deimv2_requirements_without_torch_pins() {
    local req_fpath="$1"
    local filtered_req_fpath
    filtered_req_fpath="$(mktemp)"
    grep -vE '^(torch|torchvision)([[:space:]]*[<>=!~].*)?$' "$req_fpath" > "$filtered_req_fpath"
    python_pkg_install -r "$filtered_req_fpath"
    rm -f "$filtered_req_fpath"
}

install_maskdino_requirements_preserve_opencv_stack() {
    local req_fpath="$1"
    local filtered_req_fpath
    filtered_req_fpath="$(mktemp)"
    grep -vE '^opencv-python([[:space:]]*[<>=!~].*)?$' "$req_fpath" > "$filtered_req_fpath"
    python_pkg_install -r "$filtered_req_fpath"
    rm -f "$filtered_req_fpath"
    if ! python - <<'PY_EOF'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("cv2") is not None else 1)
PY_EOF
    then
        python_pkg_install opencv-python-headless
    fi
}

python_pkg_install -r "$FOUNDATION_V3_ROOT_DIR/requirements/runtime.txt" -r "$FOUNDATION_V3_ROOT_DIR/requirements/tests.txt"
python_pkg_install -e "$FOUNDATION_V3_ROOT_DIR"
python_pkg_install kwcoco kwimage kwutil huggingface_hub gdown pycocotools

install_deimv2_requirements_without_torch_pins "$SHITSPOTTER_DEIMV2_REPO_DPATH/requirements.txt"
python_pkg_install -e "$SHITSPOTTER_SAM2_REPO_DPATH"
python_pkg_install tensordict submitit iopath fvcore pandas scikit-image tensorboard
install_maskdino_requirements_preserve_opencv_stack "$SHITSPOTTER_MASKDINO_REPO_DPATH/requirements.txt"

cat <<REPORT_EOF
Environment setup complete.

Canonical backend owner:
  KWCOCO_DETECTOR_KIT_DPATH=$KWCOCO_DETECTOR_KIT_DPATH

Repo paths:
  SHITSPOTTER_DEIMV2_REPO_DPATH=$SHITSPOTTER_DEIMV2_REPO_DPATH
  SHITSPOTTER_SAM2_REPO_DPATH=$SHITSPOTTER_SAM2_REPO_DPATH
  SHITSPOTTER_MASKDINO_REPO_DPATH=$SHITSPOTTER_MASKDINO_REPO_DPATH
  SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH=$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH

Notes:
  - Generic detector / segmenter source is initialized from KDK's tpl/ submodules; ShitSpotter no longer owns duplicate upstream submodules.
  - DEIMv2 upstream pins torch==2.5.1 and torchvision==0.20.1, but this setup script intentionally preserves your existing torch stack and only installs the other DEIMv2 deps.
  - MaskDINO upstream lists opencv-python, but this setup script preserves your existing cv2 provider and only installs opencv-python-headless if cv2 is missing entirely.
  - MaskDINO still requires a compatible Detectron2 install and its CUDA ops.
  - SAM2 may require a recent torch build and optional CUDA extension support.
  - SAM2 fine-tuning also needs training-side extras such as tensordict, submitit, iopath, and fvcore; this setup script installs them explicitly.
  - Run experiments/foundation_detseg_v3/download_foundation_assets.sh next if you want the default DEIMv2 and SAM2 weights placed in the expected locations.
  - geowatch aggregation may require extra GDAL/OSGeo system packages.
REPORT_EOF
