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

export DEIMV2_CKPT="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/foundation_detseg_v3/deimv2_m/best_stg2.pth"
export SAM2_CKPT="/data/joncrall/dvc-repos/shitspotter_expt_dvc/training/toothbrush/joncrall/ShitSpotter/runs/foundation_detseg_v3/sam2.1_hiera_base_plus/checkpoints/checkpoint.pt"

PACKAGE_FPATH="${PACKAGE_FPATH:-$FOUNDATION_V3_PACKAGE_DPATH/deimv2_sam2_trained_oneoff.yaml}"
VALI_FPATH="${VALI_FPATH:-${FOUNDATION_V3_VALI_KWCOCO_FPATH:?Set FOUNDATION_V3_VALI_KWCOCO_FPATH or install geowatch_dvc}}"
TEST_FPATH="${TEST_FPATH:-${FOUNDATION_V3_TEST_KWCOCO_FPATH:?Set FOUNDATION_V3_TEST_KWCOCO_FPATH or install geowatch_dvc}}"

VALI_EVAL_PATH="${VALI_EVAL_PATH:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/_foundation_detseg_v3/vali_trained_oneoff}"
TEST_EVAL_PATH="${TEST_EVAL_PATH:-${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}/_foundation_detseg_v3/test_trained_oneoff}"

print_path_status() {
    local label="$1"
    local path="$2"
    local kind="${3:-either}"
    local mark="X"
    local ok=1
    if [ "$kind" = "file" ]; then
        if [ -f "$path" ]; then
            mark="OK"
            ok=0
        fi
    elif [ "$kind" = "dir" ]; then
        if [ -d "$path" ]; then
            mark="OK"
            ok=0
        fi
    else
        if [ -e "$path" ]; then
            mark="OK"
            ok=0
        fi
    fi
    printf '[%s] %s: %s\n' "$mark" "$label" "$path"
    return "$ok"
}

print_output_parent_status() {
    local label="$1"
    local path="$2"
    local parent
    parent="$(dirname "$path")"
    if [ -d "$path" ]; then
        printf '[OK] %s: %s (already exists)\n' "$label" "$path"
        return 0
    fi
    if [ -d "$parent" ]; then
        printf '[NEW] %s: %s (will be created under existing parent %s)\n' "$label" "$path" "$parent"
        return 0
    fi
    printf '[NEW] %s: %s (parent does not exist yet: %s)\n' "$label" "$path" "$parent"
    return 0
}

echo "Preflight path check"
echo "Required existing inputs:"
print_path_status "SHITSPOTTER_DPATH" "$SHITSPOTTER_DPATH" dir
print_path_status "SHITSPOTTER_DEIMV2_REPO_DPATH" "$SHITSPOTTER_DEIMV2_REPO_DPATH" dir
print_path_status "SHITSPOTTER_SAM2_REPO_DPATH" "$SHITSPOTTER_SAM2_REPO_DPATH" dir
print_path_status "FOUNDATION_V3_PACKAGE_DPATH" "$FOUNDATION_V3_PACKAGE_DPATH" dir
print_path_status "VALI_FPATH" "$VALI_FPATH" file
print_path_status "TEST_FPATH" "$TEST_FPATH" file
print_path_status "DEIMV2_CKPT" "$DEIMV2_CKPT" file
print_path_status "SAM2_CKPT" "$SAM2_CKPT" file
print_path_status "PACKAGE_FPATH parent" "$(dirname "$PACKAGE_FPATH")" dir
echo
echo "Planned outputs:"
print_output_parent_status "PACKAGE_FPATH" "$PACKAGE_FPATH"
print_output_parent_status "VALI_EVAL_PATH" "$VALI_EVAL_PATH"
print_output_parent_status "TEST_EVAL_PATH" "$TEST_EVAL_PATH"
echo

python -m shitspotter.algo_foundation_v3.cli_package build "$PACKAGE_FPATH" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_CKPT" \
    --metadata_name deimv2_sam2_trained_oneoff

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.foundation_v3_evaluation_pipeline()'
        matrix:
            foundation_v3_pred.src:
                - '$VALI_FPATH'
            foundation_v3_pred.package_fpath:
                - '$PACKAGE_FPATH'
            foundation_v3_pred.create_labelme: 0
            detection_evaluation.__enabled__: 1
    " \
    --root_dpath="$VALI_EVAL_PATH" \
    --devices="0," --tmux_workers=1 \
    --backend=serial --skip_existing=1 \
    --run=1

python -m geowatch.mlops.schedule_evaluation \
    --params="
        pipeline: 'shitspotter.pipelines.foundation_v3_evaluation_pipeline()'
        matrix:
            foundation_v3_pred.src:
                - '$TEST_FPATH'
            foundation_v3_pred.package_fpath:
                - '$PACKAGE_FPATH'
            foundation_v3_pred.create_labelme: 0
            detection_evaluation.__enabled__: 1
    " \
    --root_dpath="$TEST_EVAL_PATH" \
    --devices="0," --tmux_workers=1 \
    --backend=serial --skip_existing=1 \
    --run=1
