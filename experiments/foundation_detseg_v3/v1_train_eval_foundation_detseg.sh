#!/bin/bash
set -euo pipefail

# Hard-coded experiment layout for a fully explicit, non-overwriting v1 run.

REPO_DPATH="/home/joncrall/code/shitspotter"
DATA_DPATH="/home/joncrall/data/dvc-repos/shitspotter_dvc"
EXPT_DPATH="/home/joncrall/data/dvc-repos/shitspotter_expt_dvc"

SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"

TRAIN_FPATH="$DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip"
VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip"

DEIMV2_INIT_CKPT="$DATA_DPATH/models/foundation_detseg_v3/deimv2/deimv2_dinov3_m_coco.pth"
SAM2_INIT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"

V1_ROOT="$EXPT_DPATH/foundation_detseg_v3/v1"
DETECTOR_WORKDIR="$V1_ROOT/train_detector_deimv2_m"
SAM2_WORKDIR="$V1_ROOT/train_segmenter_sam2_1_hiera_base_plus"
BOX_VALI_DPATH="$V1_ROOT/eval_detector_only/vali"
BOX_TEST_DPATH="$V1_ROOT/eval_detector_only/test"
COMBINED_VALI_DPATH="$V1_ROOT/eval_detector_segmenter/vali"
COMBINED_TEST_DPATH="$V1_ROOT/eval_detector_segmenter/test"

PACKAGE_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3/packages"
COMBINED_PACKAGE_FPATH="$PACKAGE_DPATH/v1_deimv2_m_sam2_1_hiera_base_plus_trained.yaml"

fail_if_exists() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Refusing to overwrite existing path: $path" >&2
        echo "Use a new script version label (v2, v3, ...) or remove the old path manually." >&2
        exit 1
    fi
}

echo "v1 foundation_detseg_v3 experiment"
printf '  %-28s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-28s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-28s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-28s %s\n' "TRAIN_FPATH" "$TRAIN_FPATH"
printf '  %-28s %s\n' "VALI_FPATH" "$VALI_FPATH"
printf '  %-28s %s\n' "TEST_FPATH" "$TEST_FPATH"
printf '  %-28s %s\n' "DETECTOR_WORKDIR" "$DETECTOR_WORKDIR"
printf '  %-28s %s\n' "SAM2_WORKDIR" "$SAM2_WORKDIR"
printf '  %-28s %s\n' "COMBINED_PACKAGE_FPATH" "$COMBINED_PACKAGE_FPATH"
printf '  %-28s %s\n' "BOX_VALI_DPATH" "$BOX_VALI_DPATH"
printf '  %-28s %s\n' "BOX_TEST_DPATH" "$BOX_TEST_DPATH"
printf '  %-28s %s\n' "COMBINED_VALI_DPATH" "$COMBINED_VALI_DPATH"
printf '  %-28s %s\n' "COMBINED_TEST_DPATH" "$COMBINED_TEST_DPATH"

for required in \
    "$REPO_DPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH" \
    "$SHITSPOTTER_SAM2_REPO_DPATH" \
    "$TRAIN_FPATH" \
    "$VALI_FPATH" \
    "$TEST_FPATH" \
    "$DEIMV2_INIT_CKPT" \
    "$SAM2_INIT_CKPT"; do
    if [ ! -e "$required" ]; then
        echo "Required path does not exist: $required" >&2
        exit 1
    fi
done

fail_if_exists "$DETECTOR_WORKDIR"
fail_if_exists "$SAM2_WORKDIR"
fail_if_exists "$BOX_VALI_DPATH"
fail_if_exists "$BOX_TEST_DPATH"
fail_if_exists "$COMBINED_VALI_DPATH"
fail_if_exists "$COMBINED_TEST_DPATH"
fail_if_exists "$COMBINED_PACKAGE_FPATH"

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_DEIMV2_REPO_DPATH
export SHITSPOTTER_SAM2_REPO_DPATH
export SHITSPOTTER_MASKDINO_REPO_DPATH

echo
echo "=== Train detector ==="
export TRAIN_FPATH
export VALI_FPATH
export WORKDIR="$DETECTOR_WORKDIR"
export VARIANT="deimv2_m"
export DEIMV2_INIT_CKPT
export TRAIN_BATCH_SIZE="24"
export VAL_BATCH_SIZE="48"
export USE_AMP="True"
export ENABLE_RESIZE_PREPROCESS="True"
export FORCE_RESIZE_PREPROCESS="False"
export RESIZE_MAX_DIM="640"
bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"

DEIMV2_TRAINED_CKPT="$DETECTOR_WORKDIR/best_stg2.pth"
if [ ! -f "$DEIMV2_TRAINED_CKPT" ]; then
    echo "Expected detector checkpoint missing: $DEIMV2_TRAINED_CKPT" >&2
    exit 1
fi

echo
echo "=== Train segmenter ==="
export SAM2_WORKDIR
export SAM2_VARIANT="sam2.1_hiera_base_plus"
export SAM2_INIT_CKPT
export SAM2_TRAIN_BATCH_SIZE="1"
export SAM2_NUM_TRAIN_WORKERS="8"
export SAM2_NUM_EPOCHS="20"
export SAM2_NUM_GPUS="1"
export SAM2_BASE_LR="0.000005"
export SAM2_VISION_LR="0.000003"
export SAM2_MAX_NUM_OBJECTS="8"
export SAM2_MULTIPLIER="1"
export SAM2_CHECKPOINT_SAVE_FREQ="1"
export DEIMV2_TRAINED_CKPT
bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"

SAM2_TRAINED_CKPT="$SAM2_WORKDIR/checkpoints/checkpoint.pt"
if [ ! -f "$SAM2_TRAINED_CKPT" ]; then
    echo "Expected segmenter checkpoint missing: $SAM2_TRAINED_CKPT" >&2
    exit 1
fi

echo
echo "=== Build combined package ==="
python -m shitspotter.algo_foundation_v3.cli_package build \
    "$COMBINED_PACKAGE_FPATH" \
    --backend deimv2_sam2 \
    --detector_preset deimv2_m \
    --segmenter_preset sam2.1_hiera_base_plus \
    --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
    --segmenter_checkpoint_fpath "$SAM2_TRAINED_CKPT" \
    --metadata_name v1_deimv2_m_sam2_1_hiera_base_plus_trained

echo
echo "=== Evaluate detector only on validation ==="
export PACKAGE_FPATH="$COMBINED_PACKAGE_FPATH"
export EVAL_PATH="$BOX_VALI_DPATH"
bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_boxes_on_vali.sh"

echo
echo "=== Evaluate detector only on test ==="
export EVAL_PATH="$BOX_TEST_DPATH"
bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_boxes_on_test.sh"

echo
echo "=== Evaluate detector + segmenter on validation ==="
export EVAL_PATH="$COMBINED_VALI_DPATH"
bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh"

echo
echo "=== Evaluate detector + segmenter on test ==="
export EVAL_PATH="$COMBINED_TEST_DPATH"
bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_test.sh"

echo
echo "v1 run completed"
printf '  %-28s %s\n' "DEIMV2_TRAINED_CKPT" "$DEIMV2_TRAINED_CKPT"
printf '  %-28s %s\n' "SAM2_TRAINED_CKPT" "$SAM2_TRAINED_CKPT"
printf '  %-28s %s\n' "COMBINED_PACKAGE_FPATH" "$COMBINED_PACKAGE_FPATH"
printf '  %-28s %s\n' "BOX_VALI_METRICS" "$BOX_VALI_DPATH/eval/detect_metrics.json"
printf '  %-28s %s\n' "BOX_TEST_METRICS" "$BOX_TEST_DPATH/eval/detect_metrics.json"
printf '  %-28s %s\n' "COMBINED_VALI_METRICS" "$COMBINED_VALI_DPATH/eval/detect_metrics.json"
printf '  %-28s %s\n' "COMBINED_TEST_METRICS" "$COMBINED_TEST_DPATH/eval/detect_metrics.json"
