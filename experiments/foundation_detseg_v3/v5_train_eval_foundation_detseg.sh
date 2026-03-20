#!/bin/bash
set -euo pipefail

# Hard-coded experiment layout for a v5 run whose detector stage is meant to
# be comparable to the older GroundingDINO / Detectron / YOLO family.
# The important part is matching the detector-data semantics while still using
# the efficient offline resize path that makes DEIMv2 training practical here.
# So v5 keeps resize preprocessing, disables simplify preprocessing, and then
# exports detector COCO with absolute image paths from the resized kwcoco.

canonical_existing_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
    (cd "$path" && pwd -P)
}

choose_first_existing_file() {
    local candidate
    for candidate in "$@"; do
        if [ -f "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

require_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
}

read_ap() {
    local metrics_fpath="$1"
    python - "$metrics_fpath" <<'PY'
import json
import sys

metrics_fpath = sys.argv[1]
data = json.loads(open(metrics_fpath, 'r').read())

def find_ap(node):
    if isinstance(node, dict):
        if 'nocls_measures' in node and isinstance(node['nocls_measures'], dict):
            val = node['nocls_measures'].get('ap', None)
            if val is not None:
                return val
        for value in node.values():
            found = find_ap(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_ap(value)
            if found is not None:
                return found
    return None

ap = find_ap(data)
if ap is None:
    raise KeyError('Could not find nocls_measures.ap')
print(f'{float(ap):.3f}')
PY
}

ensure_detector_checkpoint() {
    choose_first_existing_file \
        "$DETECTOR_WORKDIR/best_stg2.pth" \
        "$DETECTOR_WORKDIR/best_stg1.pth" \
        "$DETECTOR_WORKDIR/last.pth"
}

ensure_segmenter_checkpoint() {
    choose_first_existing_file \
        "$SAM2_WORKDIR/checkpoints/checkpoint.pt"
}

have_metrics() {
    local dpath="$1"
    [ -f "$dpath/eval/detect_metrics.json" ]
}

is_truthy() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

REPO_DPATH="$(canonical_existing_path /home/joncrall/code/shitspotter)"
DATA_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_dvc)"
EXPT_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_expt_dvc)"

SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"

TRAIN_FPATH="$DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip"
VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip"

DEIMV2_INIT_CKPT="$DATA_DPATH/models/foundation_detseg_v3/deimv2/deimv2_dinov3_m_coco.pth"
SAM2_INIT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"

V5_ROOT="$EXPT_DPATH/foundation_detseg_v3/v5"
DETECTOR_WORKDIR="$V5_ROOT/train_detector_deimv2_m"
SAM2_WORKDIR="$V5_ROOT/train_segmenter_sam2_1_hiera_base_plus"

BOX_VALI_DPATH="$V5_ROOT/eval_detector_only/vali"
BOX_TEST_DPATH="$V5_ROOT/eval_detector_only/test"

GTBOX_TUNED_RAW_VALI_DPATH="$V5_ROOT/eval_gtbox_segmenter/vali/tuned_raw"
GTBOX_ZEROSHOT_RAW_VALI_DPATH="$V5_ROOT/eval_gtbox_segmenter/vali/zeroshot_raw"

COMBINED_TUNED_RAW_VALI_DPATH="$V5_ROOT/eval_detector_segmenter/vali/tuned_raw"
COMBINED_TUNED_RAW_TEST_DPATH="$V5_ROOT/eval_detector_segmenter/test/tuned_raw"

PACKAGE_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3/packages"
TUNED_PACKAGE_FPATH="$PACKAGE_DPATH/v5_deimv2_m_sam2_1_hiera_base_plus_tuned.yaml"
ZEROSHOT_PACKAGE_FPATH="$PACKAGE_DPATH/v5_deimv2_m_sam2_1_hiera_base_plus_zeroshot.yaml"

echo "v5 foundation_detseg_v3 rerooted-kwcoco detector experiment"
printf '  %-30s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-30s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-30s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-30s %s\n' "TRAIN_FPATH" "$TRAIN_FPATH"
printf '  %-30s %s\n' "VALI_FPATH" "$VALI_FPATH"
printf '  %-30s %s\n' "V5_ROOT" "$V5_ROOT"
printf '  %-30s %s\n' "DETECTOR_WORKDIR" "$DETECTOR_WORKDIR"
printf '  %-30s %s\n' "SAM2_WORKDIR" "$SAM2_WORKDIR"

for required in \
    "$REPO_DPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH" \
    "$SHITSPOTTER_SAM2_REPO_DPATH" \
    "$TRAIN_FPATH" \
    "$VALI_FPATH" \
    "$TEST_FPATH" \
    "$DEIMV2_INIT_CKPT" \
    "$SAM2_INIT_CKPT"; do
    require_path "$required"
done

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_DEIMV2_REPO_DPATH
export SHITSPOTTER_SAM2_REPO_DPATH
export SHITSPOTTER_MASKDINO_REPO_DPATH

echo
echo "=== Train detector from resized rerooted kwcoco export ==="
export TRAIN_FPATH
export VALI_FPATH
export WORKDIR="$DETECTOR_WORKDIR"
export VARIANT="deimv2_m"
export DEIMV2_INIT_CKPT
export DEIMV2_NUM_GPUS="${DEIMV2_NUM_GPUS:-2}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
export TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
export VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
export USE_AMP="${USE_AMP:-True}"
export ENABLE_RESIZE_PREPROCESS="${ENABLE_RESIZE_PREPROCESS:-True}"
export RESIZE_MAX_DIM="${RESIZE_MAX_DIM:-640}"
export FORCE_RESIZE_PREPROCESS="${FORCE_RESIZE_PREPROCESS:-False}"
export ENABLE_SIMPLIFY_PREPROCESS="False"
export FORCE_DETECTOR_RERUN="${FORCE_DETECTOR_RERUN:-False}"
DEIMV2_TRAINED_CKPT=""
if ! is_truthy "$FORCE_DETECTOR_RERUN"; then
    DEIMV2_TRAINED_CKPT="$(ensure_detector_checkpoint || true)"
fi
if [ -n "${DEIMV2_TRAINED_CKPT:-}" ]; then
    echo "Reusing existing detector checkpoint: $DEIMV2_TRAINED_CKPT"
else
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
fi

DEIMV2_TRAINED_CKPT="$(ensure_detector_checkpoint)" || {
    echo "Expected detector checkpoint missing in: $DETECTOR_WORKDIR" >&2
    exit 1
}

echo
echo "=== Train segmenter ==="
export TRAIN_FPATH
export VALI_FPATH
export SAM2_WORKDIR
export SAM2_VARIANT="sam2.1_hiera_base_plus"
export SAM2_INIT_CKPT
export SAM2_TRAIN_BATCH_SIZE="1"
export SAM2_NUM_TRAIN_WORKERS="8"
export SAM2_NUM_EPOCHS="20"
export SAM2_NUM_GPUS="${SAM2_NUM_GPUS:-2}"
export SAM2_BASE_LR="0.000005"
export SAM2_VISION_LR="0.000003"
export SAM2_MAX_NUM_OBJECTS="8"
export SAM2_MULTIPLIER="1"
export SAM2_CHECKPOINT_SAVE_FREQ="1"
export DEIMV2_TRAINED_CKPT
SAM2_TRAINED_CKPT="$(ensure_segmenter_checkpoint || true)"
if [ -n "${SAM2_TRAINED_CKPT:-}" ]; then
    echo "Reusing existing segmenter checkpoint: $SAM2_TRAINED_CKPT"
else
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"
fi

SAM2_TRAINED_CKPT="$(ensure_segmenter_checkpoint)" || {
    echo "Expected segmenter checkpoint missing in: $SAM2_WORKDIR/checkpoints" >&2
    exit 1
}

echo
echo "=== Build tuned and zero-shot packages ==="
if [ -f "$TUNED_PACKAGE_FPATH" ]; then
    echo "Reusing existing package: $TUNED_PACKAGE_FPATH"
else
    python -m shitspotter.algo_foundation_v3.cli_package build \
        "$TUNED_PACKAGE_FPATH" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
        --segmenter_checkpoint_fpath "$SAM2_TRAINED_CKPT" \
        --metadata_name v5_deimv2_m_sam2_1_hiera_base_plus_tuned
fi

if [ -f "$ZEROSHOT_PACKAGE_FPATH" ]; then
    echo "Reusing existing package: $ZEROSHOT_PACKAGE_FPATH"
else
    python -m shitspotter.algo_foundation_v3.cli_package build \
        "$ZEROSHOT_PACKAGE_FPATH" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
        --segmenter_checkpoint_fpath "$SAM2_INIT_CKPT" \
        --metadata_name v5_deimv2_m_sam2_1_hiera_base_plus_zeroshot
fi

echo
echo "=== Evaluate detector only on validation ==="
export PACKAGE_FPATH="$TUNED_PACKAGE_FPATH"
export EVAL_PATH="$BOX_VALI_DPATH"
if have_metrics "$BOX_VALI_DPATH"; then
    echo "Reusing existing detector eval: $BOX_VALI_DPATH/eval/detect_metrics.json"
else
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_boxes_on_vali.sh"
fi

echo
echo "=== Evaluate detector only on test ==="
export EVAL_PATH="$BOX_TEST_DPATH"
if have_metrics "$BOX_TEST_DPATH"; then
    echo "Reusing existing detector eval: $BOX_TEST_DPATH/eval/detect_metrics.json"
else
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_boxes_on_test.sh"
fi

echo
echo "=== GT-box SAM2 sanity check on validation: tuned raw ==="
if have_metrics "$GTBOX_TUNED_RAW_VALI_DPATH"; then
    echo "Reusing existing GT-box eval: $GTBOX_TUNED_RAW_VALI_DPATH/eval/detect_metrics.json"
else
    mkdir -p "$GTBOX_TUNED_RAW_VALI_DPATH/eval"
    python -m shitspotter.algo_foundation_v3.cli_predict_gtboxes \
        "$VALI_FPATH" \
        --package_fpath "$TUNED_PACKAGE_FPATH" \
        --dst "$GTBOX_TUNED_RAW_VALI_DPATH/pred.kwcoco.zip" \
        --crop_padding 0 \
        --polygon_simplify 0 \
        --min_component_area 0 \
        --keep_largest_component False
    python -m kwcoco eval \
        --true_dataset "$VALI_FPATH" \
        --pred_dataset "$GTBOX_TUNED_RAW_VALI_DPATH/pred.kwcoco.zip" \
        --out_dpath "$GTBOX_TUNED_RAW_VALI_DPATH/eval" \
        --out_fpath "$GTBOX_TUNED_RAW_VALI_DPATH/eval/detect_metrics.json" \
        --confusion_fpath "$GTBOX_TUNED_RAW_VALI_DPATH/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5
fi

echo
echo "=== GT-box SAM2 sanity check on validation: zero-shot raw ==="
if have_metrics "$GTBOX_ZEROSHOT_RAW_VALI_DPATH"; then
    echo "Reusing existing GT-box eval: $GTBOX_ZEROSHOT_RAW_VALI_DPATH/eval/detect_metrics.json"
else
    mkdir -p "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/eval"
    python -m shitspotter.algo_foundation_v3.cli_predict_gtboxes \
        "$VALI_FPATH" \
        --package_fpath "$ZEROSHOT_PACKAGE_FPATH" \
        --dst "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/pred.kwcoco.zip" \
        --crop_padding 0 \
        --polygon_simplify 0 \
        --min_component_area 0 \
        --keep_largest_component False
    python -m kwcoco eval \
        --true_dataset "$VALI_FPATH" \
        --pred_dataset "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/pred.kwcoco.zip" \
        --out_dpath "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/eval" \
        --out_fpath "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/eval/detect_metrics.json" \
        --confusion_fpath "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5
fi

echo
echo "=== Evaluate detector + segmenter on validation: tuned raw ==="
if have_metrics "$COMBINED_TUNED_RAW_VALI_DPATH"; then
    echo "Reusing existing combined eval: $COMBINED_TUNED_RAW_VALI_DPATH/eval/detect_metrics.json"
else
    mkdir -p "$COMBINED_TUNED_RAW_VALI_DPATH/eval"
    python -m shitspotter.algo_foundation_v3.cli_predict \
        "$VALI_FPATH" \
        --package_fpath "$TUNED_PACKAGE_FPATH" \
        --dst "$COMBINED_TUNED_RAW_VALI_DPATH/pred.kwcoco.zip" \
        --crop_padding 0 \
        --polygon_simplify 0 \
        --min_component_area 0 \
        --keep_largest_component False
    python -m kwcoco eval \
        --true_dataset "$VALI_FPATH" \
        --pred_dataset "$COMBINED_TUNED_RAW_VALI_DPATH/pred.kwcoco.zip" \
        --out_dpath "$COMBINED_TUNED_RAW_VALI_DPATH/eval" \
        --out_fpath "$COMBINED_TUNED_RAW_VALI_DPATH/eval/detect_metrics.json" \
        --confusion_fpath "$COMBINED_TUNED_RAW_VALI_DPATH/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5
fi

echo
echo "=== Evaluate detector + segmenter on test: tuned raw ==="
if have_metrics "$COMBINED_TUNED_RAW_TEST_DPATH"; then
    echo "Reusing existing combined eval: $COMBINED_TUNED_RAW_TEST_DPATH/eval/detect_metrics.json"
else
    mkdir -p "$COMBINED_TUNED_RAW_TEST_DPATH/eval"
    python -m shitspotter.algo_foundation_v3.cli_predict \
        "$TEST_FPATH" \
        --package_fpath "$TUNED_PACKAGE_FPATH" \
        --dst "$COMBINED_TUNED_RAW_TEST_DPATH/pred.kwcoco.zip" \
        --crop_padding 0 \
        --polygon_simplify 0 \
        --min_component_area 0 \
        --keep_largest_component False
    python -m kwcoco eval \
        --true_dataset "$TEST_FPATH" \
        --pred_dataset "$COMBINED_TUNED_RAW_TEST_DPATH/pred.kwcoco.zip" \
        --out_dpath "$COMBINED_TUNED_RAW_TEST_DPATH/eval" \
        --out_fpath "$COMBINED_TUNED_RAW_TEST_DPATH/eval/detect_metrics.json" \
        --confusion_fpath "$COMBINED_TUNED_RAW_TEST_DPATH/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5
fi

echo
echo "=== v5 summary ==="
printf '  %-32s ap=%s\n' "detector_only_vali" "$(read_ap "$BOX_VALI_DPATH/eval/detect_metrics.json")"
printf '  %-32s ap=%s\n' "detector_only_test" "$(read_ap "$BOX_TEST_DPATH/eval/detect_metrics.json")"
printf '  %-32s ap=%s\n' "gtbox_tuned_raw_vali" "$(read_ap "$GTBOX_TUNED_RAW_VALI_DPATH/eval/detect_metrics.json")"
printf '  %-32s ap=%s\n' "gtbox_zeroshot_raw_vali" "$(read_ap "$GTBOX_ZEROSHOT_RAW_VALI_DPATH/eval/detect_metrics.json")"
printf '  %-32s ap=%s\n' "combined_tuned_raw_vali" "$(read_ap "$COMBINED_TUNED_RAW_VALI_DPATH/eval/detect_metrics.json")"
printf '  %-32s ap=%s\n' "combined_tuned_raw_test" "$(read_ap "$COMBINED_TUNED_RAW_TEST_DPATH/eval/detect_metrics.json")"

echo
echo "v5 run completed"
printf '  %-32s %s\n' "DEIMV2_TRAINED_CKPT" "$DEIMV2_TRAINED_CKPT"
printf '  %-32s %s\n' "SAM2_TRAINED_CKPT" "$SAM2_TRAINED_CKPT"
printf '  %-32s %s\n' "TUNED_PACKAGE_FPATH" "$TUNED_PACKAGE_FPATH"
printf '  %-32s %s\n' "ZEROSHOT_PACKAGE_FPATH" "$ZEROSHOT_PACKAGE_FPATH"
