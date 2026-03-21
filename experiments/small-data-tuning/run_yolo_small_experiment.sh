#!/bin/bash
set -euo pipefail

# Train the historical YOLO baseline on a fixed small-data cohort.
# This wrapper mirrors the older YOLO-v9 experiment recipe while centralizing
# subset paths, output directories, and hyperparameters in a small-data run
# manifest.

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

COHORT_NAME="${COHORT_NAME:-}"
COHORT_DPATH="${COHORT_DPATH:-}"
RUN_NAME="${RUN_NAME:-yolo_small_default}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-[640,640]}"
CPU_NUM="${CPU_NUM:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-50}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

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

RUN_DPATH="$(small_data_runs_root)/yolo/${SMALL_DATA_COHORT_NAME}/${RUN_NAME}"
mkdir -p "$RUN_DPATH"

DATASET_CONFIG_FPATH="$RUN_DPATH/yolo_dataset.yaml"
RUN_MANIFEST_FPATH="$RUN_DPATH/run_manifest.json"

read -r CLASS_YAML <<EOF
$("$PYTHON_BIN" - <<PY
import json
import pathlib
manifest = json.loads(pathlib.Path("$SMALL_DATA_COHORT_MANIFEST_FPATH").read_text())
cats = []
mscoco_fpath = pathlib.Path(manifest["subsets"]["train"]["mscoco_fpath"])
data = json.loads(mscoco_fpath.read_text())
categories = sorted(data.get("categories", []), key=lambda cat: cat["id"])
class_list = [cat["name"] for cat in categories]
print(f"class_num: {len(class_list)}")
print(f"class_list: {class_list}")
PY
)
EOF

cat > "$DATASET_CONFIG_FPATH" <<EOF
path: $COHORT_DPATH
train: $SMALL_DATA_TRAIN_MSCOCO_FPATH
validation: $SMALL_DATA_VALI_MSCOCO_FPATH

$CLASS_YAML
EOF

cat > "$RUN_MANIFEST_FPATH" <<EOF
{
  "model_family": "yolo",
  "run_name": "$RUN_NAME",
  "cohort_name": "$SMALL_DATA_COHORT_NAME",
  "cohort_manifest_fpath": "$SMALL_DATA_COHORT_MANIFEST_FPATH",
  "dataset_config_fpath": "$DATASET_CONFIG_FPATH",
  "train_fpath": "$SMALL_DATA_TRAIN_MSCOCO_FPATH",
  "vali_fpath": "$SMALL_DATA_VALI_MSCOCO_FPATH",
  "test_fpath": "$SMALL_DATA_TEST_MSCOCO_FPATH",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "trainer": {
    "image_size": "$IMAGE_SIZE",
    "cpu_num": $CPU_NUM,
    "batch_size": $BATCH_SIZE,
    "accumulate_grad_batches": $ACCUMULATE_GRAD_BATCHES,
    "lr": $LR,
    "weight_decay": $WEIGHT_DECAY
  }
}
EOF

printf 'YOLO small-data run\n'
printf '  %-22s %s\n' "COHORT_DPATH" "$COHORT_DPATH"
printf '  %-22s %s\n' "RUN_DPATH" "$RUN_DPATH"
printf '  %-22s %s\n' "DATASET_CONFIG" "$DATASET_CONFIG_FPATH"

YOLO_REPO_DPATH="$("$PYTHON_BIN" - <<'PY'
import pathlib
import yolo
print(pathlib.Path(yolo.__file__).parent.parent)
PY
)"

export CUDA_VISIBLE_DEVICES
cd "$YOLO_REPO_DPATH"
LOG_BATCH_VIZ_TO_DISK=1 "$PYTHON_BIN" -m yolo.lazy \
    task=train \
    dataset="$DATASET_CONFIG_FPATH" \
    use_tensorboard=True \
    use_wandb=False \
    out_path="$RUN_DPATH" \
    name="$RUN_NAME" \
    cpu_num="$CPU_NUM" \
    device=0 \
    task.optimizer.type=AdamW \
    +task.optimizer.args="{lr: $LR, weight_decay: $WEIGHT_DECAY, betas: [0.9, 0.99]}" \
    ~task.optimizer.args.nesterov \
    ~task.optimizer.args.momentum \
    trainer.gradient_clip_val=1.0 \
    trainer.gradient_clip_algorithm=norm \
    trainer.accelerator=auto \
    trainer.accumulate_grad_batches="$ACCUMULATE_GRAD_BATCHES" \
    task.data.batch_size="$BATCH_SIZE" \
    "image_size=$IMAGE_SIZE" \
    task.data.data_augment.RemoveOutliers=1e-12 \
    task.data.data_augment.RandomCrop=0.5 \
    task.data.data_augment.Mosaic=0.0 \
    task.data.data_augment.HorizontalFlip=0.5 \
    task.data.data_augment.VerticalFlip=0.5 \
    task.loss.objective.BCELoss=0.5 \
    weight="null"
