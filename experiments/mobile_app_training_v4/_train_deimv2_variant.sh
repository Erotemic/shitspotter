#!/bin/bash
# Library script — sourced by the per-variant 02_train_*.sh entrypoints.
#
# Required env vars (set by the caller):
#   V4_VARIANT          one of deimv2_pico, deimv2_n, deimv2_s, deimv2_m
#   V4_INPUT_HW         space-separated "H W", e.g. "320 320"
#   V4_TRAIN_BATCH      total training batch size
#   V4_VAL_BATCH        validation batch size
#   V4_NUM_EPOCHS       number of epochs (overrides upstream config)
#   V4_NUM_GPUS         number of GPUs to use (1 if no slot)
#
# Optional:
#   V4_DEIMV2_INIT_CKPT path to a pretrained .pth to fine-tune from. If
#                       unset, we look for one under
#                       $V4_ROOT/pretrained/deimv2/$V4_VARIANT_coco.pth.
#   V4_USE_AMP          True/False — defaults to True
#   V4_RUN_TAG          extra tag appended to the workdir, e.g. "tile_g2"
#   V4_DEIMV2_CONFIG_OVERRIDES  raw YAML fragment merged into the generated
#                               train config (advanced)
#   V4_LR / V4_BACKBONE_LR      explicit LR overrides; otherwise we scale
#                               from the upstream config
#   V4_RESUME           path to checkpoint to resume from
#   FORCE_RETRAIN       set to 1 to skip the "already trained" check

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

: "${V4_VARIANT:?V4_VARIANT must be set by the caller}"
: "${V4_INPUT_HW:?V4_INPUT_HW must be set by the caller (e.g. \"320 320\")}"
: "${V4_TRAIN_BATCH:?V4_TRAIN_BATCH must be set by the caller}"
: "${V4_VAL_BATCH:?V4_VAL_BATCH must be set by the caller}"
: "${V4_NUM_EPOCHS:?V4_NUM_EPOCHS must be set by the caller}"

V4_USE_AMP="${V4_USE_AMP:-True}"
V4_NUM_GPUS="${V4_NUM_GPUS:-1}"
V4_RUN_TAG="${V4_RUN_TAG:-tile_g${V4_TILE_GRID}}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

# ---------------------------------------------------------------------------
# Resolve I/O paths
# ---------------------------------------------------------------------------
DATA_DPATH="$V4_ROOT/data"
TRAIN_MSCOCO_FPATH="$DATA_DPATH/train_tile_g${V4_TILE_GRID}.simplified.mscoco.json"
VALI_MSCOCO_FPATH="$DATA_DPATH/vali_tile_g${V4_TILE_GRID}.simplified.mscoco.json"
v4_require_path "$TRAIN_MSCOCO_FPATH"
v4_require_path "$VALI_MSCOCO_FPATH"

WORKDIR="$V4_ROOT/runs/${V4_VARIANT}_${V4_RUN_TAG}_${V4_INPUT_HW// /x}"
mkdir -p "$WORKDIR"

UPSTREAM_CFG_RELPATH="$(v4_variant_repo_config "$V4_VARIANT")"
UPSTREAM_CFG_FPATH="$SHITSPOTTER_DEIMV2_REPO_DPATH/$UPSTREAM_CFG_RELPATH"
v4_require_path "$UPSTREAM_CFG_FPATH"

# Default init checkpoint location (filled by 00_setup.sh)
V4_DEIMV2_INIT_CKPT="${V4_DEIMV2_INIT_CKPT:-$V4_ROOT/pretrained/deimv2/${V4_VARIANT}_coco.pth}"

if [ ! -f "$V4_DEIMV2_INIT_CKPT" ]; then
    echo "  no pretrained init checkpoint found at $V4_DEIMV2_INIT_CKPT"
    echo "  → training from the configured upstream backbone init only"
    V4_DEIMV2_INIT_CKPT=""
fi

V4_RESUME="${V4_RESUME:-}"

# ---------------------------------------------------------------------------
# Generate the per-run DEIMv2 train config
# ---------------------------------------------------------------------------
GENERATED_CFG_DPATH="$WORKDIR/generated_configs"
mkdir -p "$GENERATED_CFG_DPATH"
GENERATED_CFG_FPATH="$GENERATED_CFG_DPATH/train.yml"

INPUT_H="${V4_INPUT_HW% *}"
INPUT_W="${V4_INPUT_HW#* }"

# Scale main LR linearly with batch size (relative to the upstream
# default of 32). The user can override with V4_LR / V4_BACKBONE_LR.
read -r DEFAULT_BACKBONE_LR DEFAULT_MAIN_LR <<EOF
$("$PYTHON_BIN" - <<PY
import os
batch = float("$V4_TRAIN_BATCH")
base_batch = 32.0
main = 5e-4 * (batch / base_batch)
backbone = 2.5e-5 * (batch / base_batch)
print(f"{backbone:.10f} {main:.10f}")
PY
)
EOF
V4_LR="${V4_LR:-$DEFAULT_MAIN_LR}"
V4_BACKBONE_LR="${V4_BACKBONE_LR:-$DEFAULT_BACKBONE_LR}"

# Optimizer block depends on backbone family (HGNetv2 vs DINOv3)
case "$V4_VARIANT" in
    deimv2_atto|deimv2_femto|deimv2_pico|deimv2_n)
        OPTIMIZER_BLOCK=$(cat <<EOF
optimizer:
  type: AdamW
  params:
    - params: '^(?=.*backbone)(?!.*norm|bn).*$'
      lr: ${V4_BACKBONE_LR}
    - params: '^(?=.*backbone)(?=.*norm|bn).*$'
      lr: ${V4_BACKBONE_LR}
      weight_decay: 0.0
    - params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0
  lr: ${V4_LR}
  betas: [0.9, 0.999]
  weight_decay: 0.0001
EOF
)
        ;;
    deimv2_s|deimv2_m|deimv2_l|deimv2_x)
        OPTIMIZER_BLOCK=$(cat <<EOF
optimizer:
  type: AdamW
  params:
    - params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: ${V4_BACKBONE_LR}
    - params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: ${V4_BACKBONE_LR}
      weight_decay: 0.0
    - params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0
  lr: ${V4_LR}
  betas: [0.9, 0.999]
  weight_decay: 0.0001
EOF
)
        ;;
    *)
        echo "Unsupported V4_VARIANT=$V4_VARIANT for v4 trainer" >&2
        exit 1
        ;;
esac

cat > "$GENERATED_CFG_FPATH" <<EOF
__include__:
  - $UPSTREAM_CFG_FPATH

# ---- mobile_app_training_v4 overrides ----
output_dir: $WORKDIR
summary_dir: $WORKDIR/summary
use_amp: $V4_USE_AMP

task: detection
num_classes: 1
remap_mscoco_category: false
evaluator:
  type: CocoEvaluator
  iou_types:
    - bbox

eval_spatial_size: [$INPUT_H, $INPUT_W]

train_dataloader:
  total_batch_size: $V4_TRAIN_BATCH
  num_workers: 4
  dataset:
    img_folder: /
    ann_file: $TRAIN_MSCOCO_FPATH
    return_masks: false

val_dataloader:
  total_batch_size: $V4_VAL_BATCH
  num_workers: 2
  dataset:
    img_folder: /
    ann_file: $VALI_MSCOCO_FPATH
    return_masks: false

epoches: $V4_NUM_EPOCHS

$OPTIMIZER_BLOCK
EOF

echo "=== mobile_app_training_v4 / 02 train $V4_VARIANT ==="
v4_print_env
printf '  %-32s %s\n' "WORKDIR"             "$WORKDIR"
printf '  %-32s %s\n' "UPSTREAM_CFG_FPATH"  "$UPSTREAM_CFG_FPATH"
printf '  %-32s %s\n' "GENERATED_CFG_FPATH" "$GENERATED_CFG_FPATH"
printf '  %-32s %s\n' "TRAIN_MSCOCO"        "$TRAIN_MSCOCO_FPATH"
printf '  %-32s %s\n' "VALI_MSCOCO"         "$VALI_MSCOCO_FPATH"
printf '  %-32s %s\n' "INPUT_HW"            "$V4_INPUT_HW"
printf '  %-32s %s\n' "TRAIN_BATCH"         "$V4_TRAIN_BATCH"
printf '  %-32s %s\n' "VAL_BATCH"           "$V4_VAL_BATCH"
printf '  %-32s %s\n' "NUM_EPOCHS"          "$V4_NUM_EPOCHS"
printf '  %-32s %s\n' "NUM_GPUS"            "$V4_NUM_GPUS"
printf '  %-32s %s\n' "LR / BACKBONE_LR"    "$V4_LR / $V4_BACKBONE_LR"
printf '  %-32s %s\n' "INIT_CKPT"           "${V4_DEIMV2_INIT_CKPT:-<none>}"
printf '  %-32s %s\n' "RESUME"              "${V4_RESUME:-<none>}"

# ---------------------------------------------------------------------------
# Already trained?
# ---------------------------------------------------------------------------
LAST_EPOCH_IDX=$(( V4_NUM_EPOCHS - 1 ))
LAST_CKPT=$(printf '%s/checkpoint%04d.pth' "$WORKDIR" "$LAST_EPOCH_IDX")
BEST_CKPT="$WORKDIR/best_stg2.pth"

if [ "$FORCE_RETRAIN" != "1" ] && { [ -f "$LAST_CKPT" ] || [ -f "$BEST_CKPT" ]; }; then
    echo "  already trained — found $LAST_CKPT or $BEST_CKPT"
    echo "  set FORCE_RETRAIN=1 to retrain"
    echo "  next: bash $V4_DEV_DPATH/03_export_onnx.sh $V4_VARIANT $V4_RUN_TAG $V4_INPUT_HW"
    exit 0
fi

# ---------------------------------------------------------------------------
# Launch DEIMv2 train.py
# ---------------------------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

TRAIN_ARGS=( "$SHITSPOTTER_DEIMV2_REPO_DPATH/train.py" -c "$GENERATED_CFG_FPATH" )
if [ -n "$V4_DEIMV2_INIT_CKPT" ]; then
    TRAIN_ARGS+=( -t "$V4_DEIMV2_INIT_CKPT" )
fi
if [ -n "$V4_RESUME" ]; then
    TRAIN_ARGS+=( -r "$V4_RESUME" )
fi
if v4_is_truthy "$V4_USE_AMP"; then
    TRAIN_ARGS+=( --use-amp )
fi

if [ "$V4_NUM_GPUS" -gt 1 ]; then
    "$PYTHON_BIN" -m torch.distributed.run \
        --master_port "${V4_MASTER_PORT:-29500}" \
        --nproc_per_node "$V4_NUM_GPUS" \
        "${TRAIN_ARGS[@]}"
else
    "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
fi

echo
echo "Training complete:"
printf '  %-32s %s\n' "WORKDIR"   "$WORKDIR"
printf '  %-32s %s\n' "BEST_CKPT" "$BEST_CKPT"
printf '  %-32s %s\n' "LAST_CKPT" "$LAST_CKPT"
echo
echo "Next: bash $V4_DEV_DPATH/03_export_onnx.sh $V4_VARIANT $V4_RUN_TAG $V4_INPUT_HW"
