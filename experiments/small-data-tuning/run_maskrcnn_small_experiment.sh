#!/bin/bash
set -euo pipefail

# Train the historical Mask R-CNN baseline on a fixed small-data cohort.
# This wrapper favors explicit configuration capture over shell brevity:
# it writes both the train config and a run manifest into the standardized
# small-data run directory before launching detectron2 training.

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

COHORT_NAME="${COHORT_NAME:-}"
COHORT_DPATH="${COHORT_DPATH:-}"
RUN_NAME="${RUN_NAME:-maskrcnn_small_default}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IMS_PER_BATCH="${IMS_PER_BATCH:-2}"
BASE_LR="${BASE_LR:-0.00025}"
MAX_ITER="${MAX_ITER:-12000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
INIT_CONFIG="${INIT_CONFIG:-COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml}"

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

RUN_DPATH="$(small_data_runs_root)/maskrcnn/${SMALL_DATA_COHORT_NAME}/${RUN_NAME}"
mkdir -p "$RUN_DPATH"

TRAIN_CONFIG_FPATH="$RUN_DPATH/train_config.yaml"
RUN_MANIFEST_FPATH="$RUN_DPATH/run_manifest.json"

cat > "$TRAIN_CONFIG_FPATH" <<EOF
default_root_dir: $RUN_DPATH
expt_name: $RUN_NAME
train_fpath: $SMALL_DATA_TRAIN_MSCOCO_FPATH
vali_fpath: $SMALL_DATA_VALI_MSCOCO_FPATH
cfg:
    MODEL:
        WEIGHTS: ''
    DATALOADER:
        NUM_WORKERS: $NUM_WORKERS
    SOLVER:
        IMS_PER_BATCH: $IMS_PER_BATCH
        BASE_LR: $BASE_LR
        MAX_ITER: $MAX_ITER
        STEPS: []
EOF

cat > "$RUN_MANIFEST_FPATH" <<EOF
{
  "model_family": "maskrcnn",
  "run_name": "$RUN_NAME",
  "cohort_name": "$SMALL_DATA_COHORT_NAME",
  "cohort_manifest_fpath": "$SMALL_DATA_COHORT_MANIFEST_FPATH",
  "train_fpath": "$SMALL_DATA_TRAIN_MSCOCO_FPATH",
  "vali_fpath": "$SMALL_DATA_VALI_MSCOCO_FPATH",
  "test_fpath": "$SMALL_DATA_TEST_MSCOCO_FPATH",
  "train_config_fpath": "$TRAIN_CONFIG_FPATH",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "solver": {
    "ims_per_batch": $IMS_PER_BATCH,
    "base_lr": $BASE_LR,
    "max_iter": $MAX_ITER,
    "num_workers": $NUM_WORKERS,
    "init_config": "$INIT_CONFIG"
  }
}
EOF

printf 'Mask R-CNN small-data run\n'
printf '  %-22s %s\n' "COHORT_DPATH" "$COHORT_DPATH"
printf '  %-22s %s\n' "RUN_DPATH" "$RUN_DPATH"
printf '  %-22s %s\n' "TRAIN_FPATH" "$SMALL_DATA_TRAIN_MSCOCO_FPATH"
printf '  %-22s %s\n' "VALI_FPATH" "$SMALL_DATA_VALI_MSCOCO_FPATH"
printf '  %-22s %s\n' "TRAIN_CONFIG" "$TRAIN_CONFIG_FPATH"

export CUDA_VISIBLE_DEVICES
"$PYTHON_BIN" -m shitspotter.detectron2.fit --config "$TRAIN_CONFIG_FPATH"
