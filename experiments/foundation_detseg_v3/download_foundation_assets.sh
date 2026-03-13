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

ensure_python_module() {
    local module_name="$1"
    local package_name="$2"
    if ! python - <<PY >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$module_name") is not None else 1)
PY
    then
        python -m pip install "$package_name"
    fi
}

download_gdrive_if_missing() {
    local file_id="$1"
    local dst_fpath="$2"
    if [ -f "$dst_fpath" ]; then
        echo "Already have $dst_fpath"
        return 0
    fi
    mkdir -p "$(dirname "$dst_fpath")"
    python -m gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view?usp=sharing" -O "$dst_fpath"
}

ensure_python_module gdown gdown
ensure_python_module huggingface_hub huggingface_hub

mkdir -p "$FOUNDATION_V3_MODEL_DPATH/deimv2"
mkdir -p "$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts"
mkdir -p "$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints"

# DEIMv2 DINOv3-distilled backbone initialization files used by the S/M configs.
download_gdrive_if_missing "1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/vitt_distill.pt"
download_gdrive_if_missing "1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/vittplus_distill.pt"

# Optional full DEIMv2 detector checkpoints for immediate inference or tuning.
download_gdrive_if_missing "1MDOh8UXD39DNSew6rDzGFp1tAVpSGJdL" \
    "$FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_s_coco.pth"
download_gdrive_if_missing "1nPKDHrotusQ748O1cQXJfi5wdShq6bKp" \
    "$FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_m_coco.pth"

python - <<'PY'
import os
from huggingface_hub import hf_hub_download

download_plan = [
    ('facebook/sam2.1-hiera-base-plus', 'sam2.1_hiera_base_plus.pt'),
    ('facebook/sam2.1-hiera-large', 'sam2.1_hiera_large.pt'),
]
local_dir = os.path.expanduser(os.environ['SHITSPOTTER_SAM2_REPO_DPATH'])
local_dir = os.path.join(local_dir, 'checkpoints')
os.makedirs(local_dir, exist_ok=True)
for repo_id, filename in download_plan:
    path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(f'Downloaded {repo_id}:{filename} -> {path}')
PY

cat <<EOF
Foundation assets are ready.

Downloaded files:
  - $SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/vitt_distill.pt
  - $SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts/vittplus_distill.pt
  - $FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_s_coco.pth
  - $FOUNDATION_V3_MODEL_DPATH/deimv2/deimv2_dinov3_m_coco.pth
  - $SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt
  - $SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_large.pt

Notes:
  - The DEIMv2 repo expects vitt_distill.pt / vittplus_distill.pt under its local ckpts directory.
  - Reusable detector checkpoints are stored under FOUNDATION_V3_MODEL_DPATH=$FOUNDATION_V3_MODEL_DPATH
  - SAM2 checkpoints are stored under the local SAM2 repo checkpoints directory.
EOF
