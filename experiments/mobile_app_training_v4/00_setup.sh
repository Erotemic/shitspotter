#!/bin/bash
# Sanity-check the v4 environment: print resolved paths, verify required
# inputs exist, optionally download pretrained DEIMv2 detector checkpoints,
# and report GPU availability.
#
# This script is safe to re-run. It is read-mostly and only writes into
# $V4_ROOT/pretrained when downloading.

set -euo pipefail

_v4_source="${BASH_SOURCE[0]-}"
if [ -n "$_v4_source" ] && [ "$_v4_source" != "bash" ] && [ "$_v4_source" != "-bash" ]; then
    _v4_script_dpath="$(cd "$(dirname "$_v4_source")" && pwd)"
else
    _v4_script_dpath="${HOME}/code/shitspotter/experiments/mobile_app_training_v4"
fi
# shellcheck source=experiments/mobile_app_training_v4/common.sh
source "$_v4_script_dpath/common.sh"
unset _v4_source _v4_script_dpath

DOWNLOAD_PRETRAINED="${DOWNLOAD_PRETRAINED:-True}"
PRETRAINED_DPATH="$V4_ROOT/pretrained/deimv2"
DEIMV2_CKPTS_DPATH="$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts"
mkdir -p "$PRETRAINED_DPATH" "$DEIMV2_CKPTS_DPATH"

echo "=== mobile_app_training_v4 environment ==="
v4_print_env

echo
echo "=== Required inputs ==="
for required in \
    "$SHITSPOTTER_DPATH" \
    "$DVC_DATA_DPATH" \
    "$V4_TRAIN_FPATH" \
    "$V4_VALI_FPATH" \
    "$V4_TEST_FPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH/train.py" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH/tools/deployment/export_onnx.py"; do
    v4_require_path "$required"
done
echo "  all required paths exist"

echo
echo "=== Teacher (v9 OpenGroundingDINO) ==="
if [ -e "$V9_PACKAGE_FPATH" ]; then
    echo "  package: $V9_PACKAGE_FPATH"
else
    echo "  WARNING: v9 package not found at $V9_PACKAGE_FPATH"
    echo "    The distillation step will be skipped or run with --no-teacher."
fi
if [ -e "$V9_SELECTED_MANIFEST_FPATH" ]; then
    echo "  manifest: $V9_SELECTED_MANIFEST_FPATH"
fi

echo
echo "=== Pretrained DEIMv2 detector checkpoints ==="
if v4_is_truthy "$DOWNLOAD_PRETRAINED"; then
    # The `if ! "$PYTHON_BIN" - <<PY` form is the only one that survives
    # `set -euo pipefail`. The earlier `$?`-check pattern aborted the
    # script before reaching the install branch when gdown was missing.
    if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('gdown') is not None else 1)
PY
    then
        echo "  installing gdown into the active Python env"
        "$PYTHON_BIN" -m pip install gdown
    fi

    for variant in $V4_DEFAULT_VARIANTS; do
        gid="$(v4_variant_init_ckpt_url "$variant" || true)"
        if [ -z "$gid" ]; then
            echo "  no pretrained URL for $variant — skipping"
            continue
        fi
        dst="$PRETRAINED_DPATH/${variant}_coco.pth"
        if [ -f "$dst" ]; then
            echo "  reusing $dst"
        else
            echo "  downloading $variant pretrained -> $dst"
            "$PYTHON_BIN" -m gdown --fuzzy \
                "https://drive.google.com/file/d/${gid}/view?usp=sharing" \
                -O "$dst"
        fi
    done

    # The DINOv3 distilled backbone init files are required by deimv2_s/m
    # configs even when fine-tuning from a full detector checkpoint.
    if [[ " $V4_DEFAULT_VARIANTS " == *" deimv2_s "* ]] || \
       [[ " $V4_DEFAULT_VARIANTS " == *" deimv2_m "* ]]; then
        for entry in \
            "1YMTq_woOLjAcZnHSYNTsNg7f0ahj5LPs vitt_distill.pt" \
            "1COHfjzq5KfnEaXTluVGEOMdhpuVcG6Jt vittplus_distill.pt"; do
            gid="${entry% *}"
            name="${entry##* }"
            dst="$DEIMV2_CKPTS_DPATH/$name"
            if [ -f "$dst" ]; then
                echo "  reusing $dst"
            else
                echo "  downloading backbone init -> $dst"
                "$PYTHON_BIN" -m gdown --fuzzy \
                    "https://drive.google.com/file/d/${gid}/view?usp=sharing" \
                    -O "$dst"
            fi
        done
    fi
else
    echo "  DOWNLOAD_PRETRAINED=False — skipping pretrained download"
fi

echo
echo "=== GPU report ==="
if "$PYTHON_BIN" -c 'import torch' >/dev/null 2>&1; then
    "$PYTHON_BIN" - <<'PY'
import torch
n = torch.cuda.device_count()
print(f'  torch={torch.__version__} cuda_available={torch.cuda.is_available()} num_gpus={n}')
for i in range(n):
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f'    gpu[{i}] = {name}  free={free/1e9:.1f}G  total={total/1e9:.1f}G')
PY
else
    echo "  WARNING: torch not importable from $PYTHON_BIN"
fi

echo
echo "Setup complete. Next steps:"
echo "  1. bash $V4_DEV_DPATH/01_make_tile_augmented_kwcoco.sh"
echo "  2. bash $V4_DEV_DPATH/02_train_deimv2_n.sh           # primary phone candidate"
echo "  3. bash $V4_DEV_DPATH/02_train_deimv2_pico.sh        # speed fallback"
echo "  4. bash $V4_DEV_DPATH/02_train_deimv2_s.sh           # quality reference"
