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

    # onnxscript is required by torch.onnx.export on torch >= 2.5 even
    # when the user passes dynamo=False — the import happens at
    # function-call time inside torch.onnx.__init__. Block here so
    # both the v4_mock export AND the DEIMv2 tools/deployment/
    # export_onnx.py paths just work later in the sweep.
    if ! "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('onnxscript') is not None else 1)
PY
    then
        echo "  installing onnxscript (required by torch.onnx.export on torch>=2.5)"
        "$PYTHON_BIN" -m pip install onnxscript
    fi

    # Min size guard: Drive serves a tiny HTML "quota exceeded" / "needs
    # confirmation" page on failure. Without this guard the next run
    # would short-circuit and skip the redownload because a file exists.
    # 1 MiB threshold is well below the smallest real DEIMv2 detector
    # checkpoint (~5 MiB for Pico) and well above any error page.
    _v4_min_download_bytes=$(( 1024 * 1024 ))

    _v4_have_real_download() {
        local p="$1"
        [ -f "$p" ] || return 1
        local sz
        sz=$(stat -c '%s' "$p" 2>/dev/null || stat -f '%z' "$p" 2>/dev/null || echo 0)
        [ "$sz" -ge "$_v4_min_download_bytes" ]
    }

    # gdown 6.x dropped --fuzzy (the URL parser now accepts every form
    # natively). Pass the bare file ID, which has worked in every gdown
    # version. For 6.x compatibility we don't pass --fuzzy.
    _v4_gdown() {
        local gid="$1" dst="$2"
        "$PYTHON_BIN" -m gdown "$gid" -O "$dst"
    }

    for variant in $V4_DEFAULT_VARIANTS; do
        gid="$(v4_variant_init_ckpt_url "$variant" || true)"
        if [ -z "$gid" ]; then
            echo "  no pretrained URL for $variant — skipping"
            continue
        fi
        dst="$PRETRAINED_DPATH/${variant}_coco.pth"
        if _v4_have_real_download "$dst"; then
            echo "  reusing $dst"
        else
            if [ -f "$dst" ]; then
                echo "  $dst exists but is too small ($(stat -c '%s' "$dst" 2>/dev/null) bytes) — re-downloading"
                rm -f "$dst"
            fi
            echo "  downloading $variant pretrained -> $dst"
            _v4_gdown "$gid" "$dst"
            if ! _v4_have_real_download "$dst"; then
                echo "  ERROR: download of $variant looks bogus (under 1 MiB)" >&2
                echo "  Drive may be throttling. Retry, or download manually:" >&2
                echo "    https://drive.google.com/file/d/${gid}/view" >&2
                exit 1
            fi
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
            if _v4_have_real_download "$dst"; then
                echo "  reusing $dst"
            else
                if [ -f "$dst" ]; then
                    echo "  $dst exists but is too small — re-downloading"
                    rm -f "$dst"
                fi
                echo "  downloading backbone init -> $dst"
                _v4_gdown "$gid" "$dst"
                if ! _v4_have_real_download "$dst"; then
                    echo "  ERROR: download of $name looks bogus (under 1 MiB)" >&2
                    echo "  Retry, or download manually:" >&2
                    echo "    https://drive.google.com/file/d/${gid}/view" >&2
                    exit 1
                fi
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
