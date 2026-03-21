#!/bin/bash
set -euo pipefail

# Train and checkpoint-select a DEIMv2 detector on a fixed small-data cohort.
# This is the detector-only fast loop for the DINOv3 family. It reuses the
# foundation_detseg_v3 detector wrapper so the preprocessing path stays aligned
# with the larger modern experiments.

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

COHORT_NAME="${COHORT_NAME:-}"
COHORT_DPATH="${COHORT_DPATH:-}"
RUN_NAME="${RUN_NAME:-deimv2_small_default}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DETECTOR_SCORE_THRESH="${DETECTOR_SCORE_THRESH:-0.2}"
DETECTOR_NMS_THRESH="${DETECTOR_NMS_THRESH:-0.5}"
DEIMV2_NUM_GPUS="${DEIMV2_NUM_GPUS:-2}"
DEIMV2_ALLOW_SINGLE_GPU_FALLBACK="${DEIMV2_ALLOW_SINGLE_GPU_FALLBACK:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
ENABLE_RESIZE_PREPROCESS="${ENABLE_RESIZE_PREPROCESS:-True}"
FORCE_RESIZE_PREPROCESS="${FORCE_RESIZE_PREPROCESS:-False}"
RESIZE_MAX_DIM="${RESIZE_MAX_DIM:-640}"
USE_AMP="${USE_AMP:-True}"
FORCE_DETECTOR_RERUN="${FORCE_DETECTOR_RERUN:-False}"
CANDIDATES=(${DEIMV2_CANDIDATES:-checkpoint0019 checkpoint0024 checkpoint0029 checkpoint0034 checkpoint0039 checkpoint0044 checkpoint0049 checkpoint0054 checkpoint0059 best_stg1 last})

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cohort_name) COHORT_NAME="$2"; shift 2 ;;
        --cohort_dpath) COHORT_DPATH="$2"; shift 2 ;;
        --run_name) RUN_NAME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$COHORT_DPATH" ]; then
    if [ -z "$COHORT_NAME" ]; then
        echo "Specify --cohort_name or --cohort_dpath" >&2
        exit 1
    fi
    COHORT_DPATH="$(small_data_cohort_from_name "$COHORT_NAME")"
fi

small_data_require_cohort "$COHORT_DPATH"
eval "$(small_data_export_cohort_env "$COHORT_DPATH")"

REPO_DPATH="$(small_data_repo_dpath)"
DATA_DPATH="$(small_data_data_dpath)"
EXPT_DPATH="$(small_data_expt_dpath)"
RUN_DPATH="$(small_data_runs_root)/deimv2/${SMALL_DATA_COHORT_NAME}/${RUN_NAME}"
DETECTOR_WORKDIR="$RUN_DPATH/train_detector_deimv2_m"
CHECKPOINT_DPATH="$RUN_DPATH/checkpoint_select"
PACKAGE_DPATH="$CHECKPOINT_DPATH/packages"
EVAL_DPATH="$CHECKPOINT_DPATH/evals"
SUMMARY_FPATH="$CHECKPOINT_DPATH/summary.tsv"
RUN_MANIFEST_FPATH="$RUN_DPATH/run_manifest.json"

mkdir -p "$PACKAGE_DPATH" "$EVAL_DPATH"

read_ap() {
    local metrics_fpath="$1"
    "$PYTHON_BIN" - "$metrics_fpath" <<'PY'
import json
import sys

data = json.loads(open(sys.argv[1], 'r').read())

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
print(f'{float(ap):.6f}')
PY
}

resolve_candidate_checkpoint() {
    local candidate_id="$1"
    local ckpt_fpath
    case "$candidate_id" in
        *.pth) ckpt_fpath="$DETECTOR_WORKDIR/$candidate_id" ;;
        *) ckpt_fpath="$DETECTOR_WORKDIR/${candidate_id}.pth" ;;
    esac
    [ -f "$ckpt_fpath" ] || return 1
    printf '%s\n' "$ckpt_fpath"
}

build_detector_package() {
    local package_fpath="$1"
    local detector_ckpt="$2"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$detector_ckpt" \
        --segmenter_checkpoint_fpath "$REPO_DPATH/tpl/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt" \
        --metadata_name "deimv2_small_$(basename "$package_fpath" .yaml)" >/dev/null
}

evaluate_detector_split() {
    local src_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    mkdir -p "$out_dpath/eval"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        "$src_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred_boxes.kwcoco.zip" \
        --score_thresh "$DETECTOR_SCORE_THRESH" \
        --nms_thresh "$DETECTOR_NMS_THRESH"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$src_fpath" \
        --pred_dataset "$out_dpath/pred_boxes.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
}

cat > "$RUN_MANIFEST_FPATH" <<EOF
{
  "model_family": "deimv2_detector",
  "run_name": "$RUN_NAME",
  "cohort_name": "$SMALL_DATA_COHORT_NAME",
  "cohort_manifest_fpath": "$SMALL_DATA_COHORT_MANIFEST_FPATH",
  "run_dpath": "$RUN_DPATH",
  "detector_workdir": "$DETECTOR_WORKDIR",
  "train_fpath": "$SMALL_DATA_TRAIN_KWCOCO_FPATH",
  "vali_fpath": "$SMALL_DATA_VALI_KWCOCO_FPATH",
  "test_fpath": "$SMALL_DATA_TEST_KWCOCO_FPATH",
  "train_batch_size": $TRAIN_BATCH_SIZE,
  "vali_batch_size": $VAL_BATCH_SIZE,
  "resize_max_dim": $RESIZE_MAX_DIM,
  "score_thresh": $DETECTOR_SCORE_THRESH,
  "nms_thresh": $DETECTOR_NMS_THRESH
}
EOF

printf 'DEIMv2 small-data detector run\n'
printf '  %-22s %s\n' "COHORT_DPATH" "$COHORT_DPATH"
printf '  %-22s %s\n' "RUN_DPATH" "$RUN_DPATH"
printf '  %-22s %s\n' "DETECTOR_WORKDIR" "$DETECTOR_WORKDIR"

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
export SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
export SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"

export TRAIN_FPATH="$SMALL_DATA_TRAIN_KWCOCO_FPATH"
export VALI_FPATH="$SMALL_DATA_VALI_KWCOCO_FPATH"
export WORKDIR="$DETECTOR_WORKDIR"
export VARIANT="deimv2_m"
export DEIMV2_INIT_CKPT="$DATA_DPATH/models/foundation_detseg_v3/deimv2/deimv2_dinov3_m_coco.pth"
export DEIMV2_NUM_GPUS
export DEIMV2_ALLOW_SINGLE_GPU_FALLBACK
export TRAIN_BATCH_SIZE
export VAL_BATCH_SIZE
export TRAIN_NUM_WORKERS
export VAL_NUM_WORKERS
export USE_AMP
export ENABLE_RESIZE_PREPROCESS
export FORCE_RESIZE_PREPROCESS
export RESIZE_MAX_DIM
export ENABLE_SIMPLIFY_PREPROCESS="False"

if [ ! -f "$DETECTOR_WORKDIR/last.pth" ] || [ "$FORCE_DETECTOR_RERUN" = "True" ]; then
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
fi

printf 'candidate_id\tckpt_fpath\tvali_ap\n' > "$SUMMARY_FPATH"
BEST_AP=""
BEST_CANDIDATE=""
BEST_CKPT=""
for candidate_id in "${CANDIDATES[@]}"; do
    ckpt_fpath="$(resolve_candidate_checkpoint "$candidate_id" || true)"
    if [ -z "${ckpt_fpath:-}" ]; then
        continue
    fi
    candidate_package_fpath="$PACKAGE_DPATH/${candidate_id}.yaml"
    candidate_eval_dpath="$EVAL_DPATH/${candidate_id}/vali"
    build_detector_package "$candidate_package_fpath" "$ckpt_fpath"
    evaluate_detector_split "$SMALL_DATA_VALI_KWCOCO_FPATH" "$candidate_package_fpath" "$candidate_eval_dpath"
    vali_ap="$(read_ap "$candidate_eval_dpath/eval/detect_metrics.json")"
    printf '%s\t%s\t%s\n' "$candidate_id" "$ckpt_fpath" "$vali_ap" >> "$SUMMARY_FPATH"
    if [ -z "$BEST_AP" ] || "$PYTHON_BIN" - <<PY
best_ap = float("${BEST_AP:-0}")
vali_ap = float("$vali_ap")
raise SystemExit(0 if vali_ap > best_ap else 1)
PY
    then
        BEST_AP="$vali_ap"
        BEST_CANDIDATE="$candidate_id"
        BEST_CKPT="$ckpt_fpath"
    fi
done

if [ -z "$BEST_CKPT" ]; then
    echo "No DEIMv2 checkpoint candidates were found in $DETECTOR_WORKDIR" >&2
    exit 1
fi

FINAL_PACKAGE_FPATH="$RUN_DPATH/${RUN_NAME}_best.yaml"
build_detector_package "$FINAL_PACKAGE_FPATH" "$BEST_CKPT"
evaluate_detector_split "$SMALL_DATA_TEST_KWCOCO_FPATH" "$FINAL_PACKAGE_FPATH" "$RUN_DPATH/test_eval"
TEST_AP="$(read_ap "$RUN_DPATH/test_eval/eval/detect_metrics.json")"

printf 'Selected DEIMv2 checkpoint: %s (vali_ap=%s, test_ap=%s)\n' "$BEST_CANDIDATE" "$BEST_AP" "$TEST_AP"
