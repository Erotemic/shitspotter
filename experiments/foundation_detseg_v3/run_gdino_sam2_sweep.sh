#!/bin/bash
# Evaluate OpenGroundingDINO detector + SAM2 segmenter combinations and measure
# combined (pixel-level) AP on validation and test splits.
#
# This script runs the following candidate configurations:
#
#   gdino_zeroshot  — best DINOv2 detector (train5747) + pretrained SAM2 (no fine-tuning)
#   gdino_tuned     — best DINOv2 detector (train5747) + v5's fine-tuned SAM2
#   v5_baseline     — v5 DEIMv2 detector + v5's fine-tuned SAM2 (for comparison)
#
# REQUIRED environment variables
# --------------------------------
# GDINO_CKPT      path to the best OpenGroundingDINO checkpoint
#                 Typically: $DVC_EXPT_DPATH/small_data_tuning/dino_detector_benchmark/
#                            train5747_<hash>/batch8_lr1p25/<run_dir>/checkpoint0006.pth
#                 Find it with:
#                   find $DVC_EXPT_DPATH/small_data_tuning -name 'checkpoint0006.pth' \
#                        -path '*/batch8_lr1p25/*' -path '*/train5747*'
#
# GDINO_CFG       path to the SLConfig written by that training run
#                 Typically the same directory: <run_dir>/config_cfg.py
#
# OPTIONAL environment variables
# --------------------------------
# TUNED_SAM2_CKPT   fine-tuned SAM2 checkpoint (default: v5's checkpoint)
# V5_PACKAGE_FPATH  path to the v5 package yaml (default: _checkpoint_select/packages/v5_checkpoint0004.yaml)
# SWEEP_ROOT        output directory for packages and eval results
# RUN_V5_BASELINE   set to "false" to skip the v5 DEIMv2 baseline
# RUN_TEST          set to "false" to skip test-split evaluation

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
# FOUNDATION_V3_SCRIPT_DPATH is now set by common.sh

PYTHON_BIN="${PYTHON_BIN:-python}"

# ---- required inputs --------------------------------------------------------
GDINO_CKPT="${GDINO_CKPT:?Set GDINO_CKPT to the best OpenGroundingDINO checkpoint path}"
GDINO_CFG="${GDINO_CFG:?Set GDINO_CFG to the config_cfg.py path from the same training run}"

# ---- data paths -------------------------------------------------------------
VALI_FPATH="${VALI_FPATH:-${FOUNDATION_V3_VALI_KWCOCO_FPATH:?Set FOUNDATION_V3_VALI_KWCOCO_FPATH or install geowatch_dvc}}"
TEST_FPATH="${TEST_FPATH:-${FOUNDATION_V3_TEST_KWCOCO_FPATH:?Set FOUNDATION_V3_TEST_KWCOCO_FPATH or install geowatch_dvc}}"

# ---- segmenter checkpoints --------------------------------------------------
REPO_DPATH="${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}"
EXPT_DPATH="${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}"

ZEROSHOT_SAM2_CKPT="${ZEROSHOT_SAM2_CKPT:-$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt}"
TUNED_SAM2_CKPT="${TUNED_SAM2_CKPT:-$EXPT_DPATH/foundation_detseg_v3/v5/train_segmenter_sam2_1_hiera_base_plus/checkpoints/checkpoint.pt}"

# ---- v5 baseline package ----------------------------------------------------
V5_PACKAGE_FPATH="${V5_PACKAGE_FPATH:-$FOUNDATION_V3_SCRIPT_DPATH/_checkpoint_select/packages/v5_checkpoint0004.yaml}"

# ---- output root ------------------------------------------------------------
DEFAULT_SWEEP_ROOT="$EXPT_DPATH/foundation_detseg_v3/gdino_sam2_sweep"
SWEEP_ROOT="${SWEEP_ROOT:-$DEFAULT_SWEEP_ROOT}"
mkdir -p "$SWEEP_ROOT/packages" "$SWEEP_ROOT/evals"

# ---- flags ------------------------------------------------------------------
RUN_V5_BASELINE="${RUN_V5_BASELINE:-true}"
RUN_TEST="${RUN_TEST:-true}"

# ---- helpers ----------------------------------------------------------------

have_metrics() {
    local dpath="$1"
    [ -f "$dpath/eval/detect_metrics.json" ]
}

read_ap() {
    local metrics_fpath="$1"
    "$PYTHON_BIN" - "$metrics_fpath" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1]).read())
def find_ap(node):
    if isinstance(node, dict):
        if 'nocls_measures' in node:
            v = node['nocls_measures'].get('ap', None)
            if v is not None:
                return v
        for v in node.values():
            found = find_ap(v)
            if found is not None:
                return found
    elif isinstance(node, list):
        for v in node:
            found = find_ap(v)
            if found is not None:
                return found
    return None
ap = find_ap(data)
if ap is None:
    raise KeyError('Could not find nocls_measures.ap')
print(f'{float(ap):.6f}')
PY
}

build_gdino_package() {
    local package_fpath="$1"
    local gdino_ckpt="$2"
    local gdino_cfg="$3"
    local sam2_ckpt="$4"
    local metadata_name="$5"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend opengroundingdino_sam2 \
        --detector_preset opengroundingdino_shitspotter \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$gdino_ckpt" \
        --detector_config_fpath "$gdino_cfg" \
        --segmenter_checkpoint_fpath "$sam2_ckpt" \
        --metadata_name "$metadata_name" >/dev/null
}

run_combined_eval() {
    local package_fpath="$1"
    local src_fpath="$2"
    local out_dpath="$3"
    mkdir -p "$out_dpath/eval"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict \
        --src="$src_fpath" \
        --package_fpath="$package_fpath" \
        --create_labelme=0 \
        --dst="$out_dpath/pred.kwcoco.zip"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset="$src_fpath" \
        --pred_dataset="$out_dpath/pred.kwcoco.zip" \
        --out_dpath="$out_dpath/eval" \
        --out_fpath="$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath="$out_dpath/eval/confusion.kwcoco.zip" \
        --draw=False \
        --iou_thresh=0.5
}

print_result() {
    local label="$1"
    local vali_ap="$2"
    local test_ap="${3:-N/A}"
    printf '  %-32s  vali_ap=%-10s  test_ap=%s\n' "$label" "$vali_ap" "$test_ap"
}

# ---- sanity checks ----------------------------------------------------------

if [ ! -f "$GDINO_CKPT" ]; then
    echo "ERROR: GDINO_CKPT not found: $GDINO_CKPT" >&2
    exit 1
fi
if [ ! -f "$GDINO_CFG" ]; then
    echo "ERROR: GDINO_CFG not found: $GDINO_CFG" >&2
    exit 1
fi
if [ ! -f "$ZEROSHOT_SAM2_CKPT" ]; then
    echo "WARNING: zero-shot SAM2 checkpoint not found: $ZEROSHOT_SAM2_CKPT" >&2
    echo "  Skipping gdino_zeroshot candidate." >&2
    SKIP_ZEROSHOT=true
else
    SKIP_ZEROSHOT=false
fi
if [ ! -f "$TUNED_SAM2_CKPT" ]; then
    echo "WARNING: tuned SAM2 checkpoint not found: $TUNED_SAM2_CKPT" >&2
    echo "  Skipping gdino_tuned candidate." >&2
    SKIP_TUNED=true
else
    SKIP_TUNED=false
fi

echo "=== gdino_sam2_sweep ==="
printf '  %-28s %s\n' "GDINO_CKPT"     "$GDINO_CKPT"
printf '  %-28s %s\n' "GDINO_CFG"      "$GDINO_CFG"
printf '  %-28s %s\n' "ZEROSHOT_SAM2"  "$ZEROSHOT_SAM2_CKPT"
printf '  %-28s %s\n' "TUNED_SAM2"     "$TUNED_SAM2_CKPT"
printf '  %-28s %s\n' "SWEEP_ROOT"     "$SWEEP_ROOT"
echo

# ---- candidate 1: gdino + zero-shot SAM2 ------------------------------------

if [ "$SKIP_ZEROSHOT" = "false" ]; then
    echo "--- gdino_zeroshot ---"
    pkg="$SWEEP_ROOT/packages/gdino_zeroshot.yaml"
    build_gdino_package "$pkg" "$GDINO_CKPT" "$GDINO_CFG" "$ZEROSHOT_SAM2_CKPT" "gdino_zeroshot"

    vali_dpath="$SWEEP_ROOT/evals/gdino_zeroshot/vali"
    if have_metrics "$vali_dpath"; then
        echo "  Reusing vali metrics"
    else
        run_combined_eval "$pkg" "$VALI_FPATH" "$vali_dpath"
    fi
    gdino_zeroshot_vali_ap="$(read_ap "$vali_dpath/eval/detect_metrics.json")"

    if [ "${RUN_TEST,,}" != "false" ]; then
        test_dpath="$SWEEP_ROOT/evals/gdino_zeroshot/test"
        if have_metrics "$test_dpath"; then
            echo "  Reusing test metrics"
        else
            run_combined_eval "$pkg" "$TEST_FPATH" "$test_dpath"
        fi
        gdino_zeroshot_test_ap="$(read_ap "$test_dpath/eval/detect_metrics.json")"
    else
        gdino_zeroshot_test_ap="N/A"
    fi
    print_result "gdino_zeroshot" "$gdino_zeroshot_vali_ap" "$gdino_zeroshot_test_ap"
fi

# ---- candidate 2: gdino + tuned SAM2 ----------------------------------------

if [ "$SKIP_TUNED" = "false" ]; then
    echo "--- gdino_tuned ---"
    pkg="$SWEEP_ROOT/packages/gdino_tuned.yaml"
    build_gdino_package "$pkg" "$GDINO_CKPT" "$GDINO_CFG" "$TUNED_SAM2_CKPT" "gdino_tuned"

    vali_dpath="$SWEEP_ROOT/evals/gdino_tuned/vali"
    if have_metrics "$vali_dpath"; then
        echo "  Reusing vali metrics"
    else
        run_combined_eval "$pkg" "$VALI_FPATH" "$vali_dpath"
    fi
    gdino_tuned_vali_ap="$(read_ap "$vali_dpath/eval/detect_metrics.json")"

    if [ "${RUN_TEST,,}" != "false" ]; then
        test_dpath="$SWEEP_ROOT/evals/gdino_tuned/test"
        if have_metrics "$test_dpath"; then
            echo "  Reusing test metrics"
        else
            run_combined_eval "$pkg" "$TEST_FPATH" "$test_dpath"
        fi
        gdino_tuned_test_ap="$(read_ap "$test_dpath/eval/detect_metrics.json")"
    else
        gdino_tuned_test_ap="N/A"
    fi
    print_result "gdino_tuned" "$gdino_tuned_vali_ap" "$gdino_tuned_test_ap"
fi

# ---- candidate 3: v5 DEIMv2 + tuned SAM2 (baseline) -------------------------

if [ "${RUN_V5_BASELINE,,}" != "false" ]; then
    echo "--- v5_baseline (deimv2 + tuned sam2) ---"
    if [ ! -f "$V5_PACKAGE_FPATH" ]; then
        echo "  WARNING: v5 package not found: $V5_PACKAGE_FPATH — skipping" >&2
    else
        vali_dpath="$SWEEP_ROOT/evals/v5_baseline/vali"
        if have_metrics "$vali_dpath"; then
            echo "  Reusing vali metrics"
        else
            run_combined_eval "$V5_PACKAGE_FPATH" "$VALI_FPATH" "$vali_dpath"
        fi
        v5_vali_ap="$(read_ap "$vali_dpath/eval/detect_metrics.json")"

        if [ "${RUN_TEST,,}" != "false" ]; then
            test_dpath="$SWEEP_ROOT/evals/v5_baseline/test"
            if have_metrics "$test_dpath"; then
                echo "  Reusing test metrics"
            else
                run_combined_eval "$V5_PACKAGE_FPATH" "$TEST_FPATH" "$test_dpath"
            fi
            v5_test_ap="$(read_ap "$test_dpath/eval/detect_metrics.json")"
        else
            v5_test_ap="N/A"
        fi
        print_result "v5_baseline (deimv2+tuned_sam2)" "$v5_vali_ap" "$v5_test_ap"
    fi
fi

# ---- summary ----------------------------------------------------------------

echo
echo "=== Summary ==="
printf '  %-32s  %-12s  %s\n' "candidate" "vali_ap" "test_ap"
printf '  %-32s  %-12s  %s\n' "---------" "-------" "-------"
[ "$SKIP_ZEROSHOT" = "false" ] && \
    print_result "gdino_zeroshot"                  "$gdino_zeroshot_vali_ap"  "$gdino_zeroshot_test_ap"
[ "$SKIP_TUNED" = "false" ] && \
    print_result "gdino_tuned"                     "$gdino_tuned_vali_ap"     "$gdino_tuned_test_ap"
[ "${RUN_V5_BASELINE,,}" != "false" ] && [ -f "$V5_PACKAGE_FPATH" ] && \
    print_result "v5_baseline (deimv2+tuned_sam2)" "$v5_vali_ap"              "$v5_test_ap"
echo
printf '  SWEEP_ROOT: %s\n' "$SWEEP_ROOT"
