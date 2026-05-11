#!/bin/bash
# Sanity-check the v4 environment: print resolved paths, verify required
# inputs exist, optionally download pretrained DEIMv2 detector checkpoints,
# install missing python deps, and report GPU availability.
#
# This script is safe to re-run. Read-mostly; only writes into
# $V4_ROOT/pretrained when downloading.
#
# Knobs:
#   DOWNLOAD_PRETRAINED=True/False  (default True) — fetch the DEIMv2
#       pretrained .pth files via gdown.
#   INSTALL_ONNX_DEPS=True/False    (default True) — install onnxscript +
#       onnx + onnxruntime when missing.
#   INSTALL_DEIMV2_DEPS=True/False  (default True) — install everything in
#       tpl/DEIMv2/requirements.txt that isn't already importable.

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
INSTALL_ONNX_DEPS="${INSTALL_ONNX_DEPS:-True}"
INSTALL_DEIMV2_DEPS="${INSTALL_DEIMV2_DEPS:-True}"
PRETRAINED_DPATH="$V4_ROOT/pretrained/deimv2"
DEIMV2_CKPTS_DPATH="$SHITSPOTTER_DEIMV2_REPO_DPATH/ckpts"
mkdir -p "$PRETRAINED_DPATH" "$DEIMV2_CKPTS_DPATH"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Returns 0 if the named Python module is importable, 1 otherwise.
# Module name is the importable name (foo.bar style), not the PyPI dist name.
v4_have_pymodule() {
    local mod="$1"
    "$PYTHON_BIN" - <<PY 2>/dev/null
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec("$mod") is not None else 1)
PY
}

# Min-size guard: gdown silently writes Drive's "quota exceeded" HTML
# stub when a download fails. Threshold (1 MiB) is below the smallest
# real DEIMv2 detector checkpoint (~5 MiB for Pico) and well above any
# error page.
_v4_min_download_bytes=$(( 1024 * 1024 ))

v4_have_real_download() {
    local p="$1"
    [ -f "$p" ] || return 1
    local sz
    sz=$(stat -c '%s' "$p" 2>/dev/null || stat -f '%z' "$p" 2>/dev/null || echo 0)
    [ "$sz" -ge "$_v4_min_download_bytes" ]
}

# gdown 6.x dropped --fuzzy. Pass the bare file ID, which works in
# every gdown version we've tested.
v4_gdown() {
    local gid="$1" dst="$2"
    "$PYTHON_BIN" -m gdown "$gid" -O "$dst"
}

# ---------------------------------------------------------------------------
# Banner + path checks
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# gdown + ONNX trio + DEIMv2 runtime deps
#
# These are all "what does the host need installed before the pipeline
# can run" — handled here so the first sweep cell doesn't crash 30s
# into a fresh setup with `ModuleNotFoundError`. Each block is
# independent and only invokes pip when something is actually missing.
# ---------------------------------------------------------------------------

echo
echo "=== Python deps ==="

# gdown: needed for the pretrained-checkpoint downloads below.
if v4_is_truthy "$DOWNLOAD_PRETRAINED"; then
    if ! v4_have_pymodule gdown; then
        echo "  installing gdown"
        "$PYTHON_BIN" -m pip install gdown
    fi
fi

# ONNX trio:
#   onnxscript    required by torch.onnx.export on torch >= 2.5 (the
#                 import happens inside torch.onnx.__init__, even when
#                 dynamo=False)
#   onnx          required at export time to actually serialise the graph
#   onnxruntime   required by 05_desktop_onnx_parity.py and
#                 06_benchmark_onnx_desktop.py
if v4_is_truthy "$INSTALL_ONNX_DEPS"; then
    _v4_missing_onnx_pkgs=()
    # onnxsim is used by DEIMv2's tools/deployment/export_onnx.py's
    # --simplify step; without it the export wrapper skips simplify
    # with a warning but the unsimplified .onnx is still produced.
    for pkg in onnxscript onnx onnxruntime onnxsim; do
        if ! v4_have_pymodule "$pkg"; then
            _v4_missing_onnx_pkgs+=( "$pkg" )
        fi
    done
    if [ "${#_v4_missing_onnx_pkgs[@]}" -gt 0 ]; then
        echo "  installing ONNX deps: ${_v4_missing_onnx_pkgs[*]}"
        "$PYTHON_BIN" -m pip install "${_v4_missing_onnx_pkgs[@]}"
    else
        echo "  ONNX deps already present"
    fi
fi

# DEIMv2 runtime deps from tpl/DEIMv2/requirements.txt (faster_coco_eval,
# calflops, transformers, tensorboard, scipy, PyYAML, ...). These aren't
# shitspotter deps; declaring them here means the sweep's first DEIMv2
# cell doesn't crash with ModuleNotFoundError partway through training.
if v4_is_truthy "$INSTALL_DEIMV2_DEPS"; then
    DEIMV2_REQ_FPATH="$SHITSPOTTER_DEIMV2_REPO_DPATH/requirements.txt"
    if [ ! -f "$DEIMV2_REQ_FPATH" ]; then
        echo "  WARNING: $DEIMV2_REQ_FPATH not found — DEIMv2 submodule not initialised?"
    else
        _v4_missing_deimv2_pkgs=()
        # Probe each declared package; only invoke pip if any are missing.
        # Map PyPI dist name -> Python module name where they differ
        # (faster-coco-eval -> faster_coco_eval, PyYAML -> yaml, etc).
        while IFS= read -r line; do
            line="${line%%#*}"
            pkg=$(echo "$line" | sed -E 's/[[:space:]]+//g; s/[><=!~].*$//')
            [ -z "$pkg" ] && continue
            case "$pkg" in
                PyYAML) mod=yaml ;;
                *)      mod=$(echo "$pkg" | tr '[:upper:]-' '[:lower:]_') ;;
            esac
            if ! v4_have_pymodule "$mod"; then
                _v4_missing_deimv2_pkgs+=( "$line" )
            fi
        done < "$DEIMV2_REQ_FPATH"

        if [ "${#_v4_missing_deimv2_pkgs[@]}" -gt 0 ]; then
            echo "  installing DEIMv2 deps: ${_v4_missing_deimv2_pkgs[*]}"
            "$PYTHON_BIN" -m pip install "${_v4_missing_deimv2_pkgs[@]}"
        else
            echo "  DEIMv2 deps already present"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Pretrained DEIMv2 detector checkpoints
# ---------------------------------------------------------------------------

echo
echo "=== Pretrained DEIMv2 detector checkpoints ==="
if v4_is_truthy "$DOWNLOAD_PRETRAINED"; then
    for variant in $V4_DEFAULT_VARIANTS; do
        gid="$(v4_variant_init_ckpt_url "$variant" || true)"
        if [ -z "$gid" ]; then
            echo "  no pretrained URL for $variant — skipping"
            continue
        fi
        dst="$PRETRAINED_DPATH/${variant}_coco.pth"
        if v4_have_real_download "$dst"; then
            echo "  reusing $dst"
        else
            if [ -f "$dst" ]; then
                echo "  $dst exists but is too small ($(stat -c '%s' "$dst" 2>/dev/null) bytes) — re-downloading"
                rm -f "$dst"
            fi
            echo "  downloading $variant pretrained -> $dst"
            v4_gdown "$gid" "$dst"
            if ! v4_have_real_download "$dst"; then
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
            if v4_have_real_download "$dst"; then
                echo "  reusing $dst"
            else
                if [ -f "$dst" ]; then
                    echo "  $dst exists but is too small — re-downloading"
                    rm -f "$dst"
                fi
                echo "  downloading backbone init -> $dst"
                v4_gdown "$gid" "$dst"
                if ! v4_have_real_download "$dst"; then
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

# ---------------------------------------------------------------------------
# GPU report
# ---------------------------------------------------------------------------

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
echo "  2. bash $V4_DEV_DPATH/02_sweep.sh                       # full Pareto sweep"
echo "    or:"
echo "     bash $V4_DEV_DPATH/02_train_deimv2_n.sh              # primary phone candidate"
echo "     bash $V4_DEV_DPATH/02_train_deimv2_pico.sh           # speed fallback"
echo "     bash $V4_DEV_DPATH/02_train_deimv2_s.sh              # quality reference"
