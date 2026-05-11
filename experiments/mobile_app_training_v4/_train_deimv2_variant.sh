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

# Variant-keyed defaults for the batch/epoch knobs. The per-variant
# 02_train_*.sh entrypoints export these explicitly; the sweep
# (02_sweep.sh) calls this trainer library directly without going
# through them, so the defaults below are what the sweep uses unless
# the user overrides via env.
#
# Tuned for a SINGLE 24 GB GPU at the smallest sweep cell. Upstream's
# total_batch_size assumes 8-GPU training (per-GPU 4 for N inheriting
# the base dataloader); we pick batches that comfortably fit on one
# 3090 with the deformable-attention sampling-locations allocation
# headroom. If you have multiple GPUs, set V4_NUM_GPUS=N (the trainer
# launches torch.distributed.run for N>1) and the per-GPU batch is
# V4_TRAIN_BATCH / N.
#
# If you OOM, halve V4_TRAIN_BATCH:
#   V4_TRAIN_BATCH=16 bash run_all.sh
case "$V4_VARIANT" in
    deimv2_n)
        V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-32}"
        V4_VAL_BATCH="${V4_VAL_BATCH:-64}"
        V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-60}"
        ;;
    deimv2_pico)
        V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-64}"
        V4_VAL_BATCH="${V4_VAL_BATCH:-128}"
        V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-80}"
        ;;
    deimv2_s)
        V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-16}"
        V4_VAL_BATCH="${V4_VAL_BATCH:-32}"
        V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-30}"
        ;;
    deimv2_m|deimv2_l|deimv2_x)
        V4_TRAIN_BATCH="${V4_TRAIN_BATCH:-8}"
        V4_VAL_BATCH="${V4_VAL_BATCH:-16}"
        V4_NUM_EPOCHS="${V4_NUM_EPOCHS:-30}"
        ;;
    *)
        # Unknown variants must still set explicit values.
        : "${V4_TRAIN_BATCH:?V4_TRAIN_BATCH must be set for variant $V4_VARIANT}"
        : "${V4_VAL_BATCH:?V4_VAL_BATCH must be set for variant $V4_VARIANT}"
        : "${V4_NUM_EPOCHS:?V4_NUM_EPOCHS must be set for variant $V4_VARIANT}"
        ;;
esac

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

# ---------------------------------------------------------------------------
# Training resolution policy (separate from the EXPORT resolution)
#
# Two axes of resolution to track per candidate:
#
#   export resolution: the fixed HxW the ONNX is exported at and the
#       phone benchmarks against. Comes from V4_INPUT_HW.
#
#   train resolution policy: how the trainer samples scales during
#       fine-tuning. Drives DEIMv2's BatchImageCollateFunction. Either:
#         fixed                  — single scale, equals export size
#         multiscale             — ±25% band around base_size=max(H,W)
#                                  at 32-px granularity (the natural
#                                  DEIMv2 generate_scales output)
#         multiscale_<lo>_<hi>   — band whose floor/ceiling targets
#                                  <lo>..<hi> pixels (e.g. multiscale_320_512)
#         multiscale_<S>         — band centered on <S>
#
# The export resolution and the train policy are recorded separately in
# the per-run manifest so the eligibility step can compare apples-to-
# apples.
#
# DEIMv2 internals (see tpl/DEIMv2/engine/data/dataloader.py:generate_scales):
#   scales = [.75*B step 32 ... B repeat=N ... 1.25*B step 32 down to B]
# so multi-scale always produces a ±25% band at 32-px granularity.
# ---------------------------------------------------------------------------
# Per-variant default. The HGNetv2 hybrid encoder (Atto/Femto/Pico/N)
# pre-bakes positional embeddings at eval_spatial_size and does not
# dynamically interpolate per batch — multi-scale collate produces
# tensor-size mismatches deep inside the encoder. Upstream's HGNetv2
# configs all set `base_size_repeat: ~` (multiscale disabled) for the
# same reason. The DINOv3-backed variants (S/M/L/X) DO support
# multi-scale (upstream sets base_size_repeat=20).
case "$V4_VARIANT" in
    deimv2_atto|deimv2_femto|deimv2_pico|deimv2_n)
        V4_TRAIN_POLICY="${V4_TRAIN_POLICY:-fixed}"
        ;;
    *)
        V4_TRAIN_POLICY="${V4_TRAIN_POLICY:-multiscale}"
        ;;
esac
V4_MULTISCALE_REPEAT="${V4_MULTISCALE_REPEAT:-12}"
_V4_DEFAULT_MS_STOP=$(( V4_NUM_EPOCHS - 4 ))
if [ "$_V4_DEFAULT_MS_STOP" -lt 1 ]; then _V4_DEFAULT_MS_STOP=$V4_NUM_EPOCHS; fi
V4_MULTISCALE_STOP_EPOCH="${V4_MULTISCALE_STOP_EPOCH:-$_V4_DEFAULT_MS_STOP}"

# Translate the policy string into (multiscale_repeat, base_size) for
# the BatchImageCollateFunction override. Compute the actual scale list
# so we can record it in the manifest. We also remember what the user
# requested so the loud "requested vs effective" banner can show both.
_V4_EXPORT_LONG=$(( INPUT_H > INPUT_W ? INPUT_H : INPUT_W ))
REQUESTED_MIN=""
REQUESTED_MAX=""
case "$V4_TRAIN_POLICY" in
    fixed)
        MS_REPEAT=0
        MS_BASE="$_V4_EXPORT_LONG"
        REQUESTED_MIN="$_V4_EXPORT_LONG"
        REQUESTED_MAX="$_V4_EXPORT_LONG"
        ;;
    multiscale)
        MS_REPEAT="$V4_MULTISCALE_REPEAT"
        MS_BASE="$_V4_EXPORT_LONG"
        # ±25% band around base
        REQUESTED_MIN=$(( (MS_BASE * 75) / 100 ))
        REQUESTED_MAX=$(( (MS_BASE * 125) / 100 ))
        ;;
    multiscale_*_*)
        MS_LO="${V4_TRAIN_POLICY#multiscale_}"
        MS_HI="${MS_LO#*_}"
        MS_LO="${MS_LO%_*}"
        # Pick base so that 0.75*base = lo and 1.25*base = hi when
        # possible; default to (lo+hi)/2 rounded to 32.
        MS_BASE=$(( (MS_LO + MS_HI) / 2 ))
        MS_BASE=$(( ((MS_BASE + 16) / 32) * 32 ))
        MS_REPEAT="$V4_MULTISCALE_REPEAT"
        REQUESTED_MIN="$MS_LO"
        REQUESTED_MAX="$MS_HI"
        ;;
    multiscale_*)
        MS_BASE="${V4_TRAIN_POLICY#multiscale_}"
        MS_REPEAT="$V4_MULTISCALE_REPEAT"
        REQUESTED_MIN=$(( (MS_BASE * 75) / 100 ))
        REQUESTED_MAX=$(( (MS_BASE * 125) / 100 ))
        ;;
    *)
        echo "Unsupported V4_TRAIN_POLICY=$V4_TRAIN_POLICY" >&2
        echo "  expected: fixed | multiscale | multiscale_<S> | multiscale_<lo>_<hi>" >&2
        exit 1
        ;;
esac

# Compute the actual scale list DEIMv2 will sample from — this is what
# generate_scales(MS_BASE, MS_REPEAT) produces. Used both for the loud
# log line below and for policy.json.
read -r EFFECTIVE_MIN EFFECTIVE_MAX EFFECTIVE_LIST <<EOF
$("$PYTHON_BIN" - <<PY
def generate_scales(base_size, base_size_repeat):
    if not base_size_repeat:
        return [int(base_size)]
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales  = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return sorted(set(scales))
scales = generate_scales(int($MS_BASE), int($MS_REPEAT) or None)
print(min(scales), max(scales), ",".join(str(s) for s in scales))
PY
)
EOF

# Loud banner. The reviewer pointed out that policy *names* like
# multiscale_320_512 don't necessarily mean the actual sampled scales
# stay inside [320, 512] — DEIMv2 rounds to 32-px granularity around
# base, so the band may overshoot or undershoot the request. Print
# both so the divergence is impossible to miss.
echo
echo "  ---- TRAIN RESOLUTION POLICY ----"
echo "    requested:    $V4_TRAIN_POLICY  (min=${REQUESTED_MIN}, max=${REQUESTED_MAX})"
echo "    effective:    base=${MS_BASE}, repeat=${MS_REPEAT}"
echo "                  scales=${EFFECTIVE_LIST}"
echo "                  min=${EFFECTIVE_MIN}, max=${EFFECTIVE_MAX}"
if [ -n "$REQUESTED_MIN" ] && [ "$EFFECTIVE_MIN" -lt "$REQUESTED_MIN" ]; then
    echo "    *** WARNING: effective min (${EFFECTIVE_MIN}) is BELOW requested (${REQUESTED_MIN})"
fi
if [ -n "$REQUESTED_MAX" ] && [ "$EFFECTIVE_MAX" -gt "$REQUESTED_MAX" ]; then
    echo "    *** WARNING: effective max (${EFFECTIVE_MAX}) is ABOVE requested (${REQUESTED_MAX})"
fi
echo "  ---------------------------------"
echo

# IMPORTANT: 2-space outer indent so collate_fn is a sibling of
# dataset under train_dataloader. With 4 spaces YAML reads collate_fn
# as a child of dataset, which makes DEIMv2's workspace.create() pass
# it as a kwarg to CocoDetection.__init__ — and CocoDetection has no
# `collate_fn` parameter, so the trainer dies with TypeError.
if [ "$MS_REPEAT" -gt 0 ]; then
    MULTISCALE_BLOCK=$(cat <<EOF
  collate_fn:
    type: BatchImageCollateFunction
    base_size: ${MS_BASE}
    base_size_repeat: ${MS_REPEAT}
    stop_epoch: ${V4_MULTISCALE_STOP_EPOCH}
EOF
)
else
    MULTISCALE_BLOCK=$(cat <<EOF
  collate_fn:
    type: BatchImageCollateFunction
    base_size: ${MS_BASE}
    base_size_repeat: ~
    stop_epoch: 1
EOF
)
fi

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
$MULTISCALE_BLOCK

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

# ---------------------------------------------------------------------------
# Dump per-run policy manifest + resolved effective config so the
# eligibility manifest can read everything back without guessing what
# DEIMv2's __include__ chain actually produced. This is the answer to
# "do not just set eval_spatial_size and assume training uses it".
# ---------------------------------------------------------------------------
POLICY_FPATH="$WORKDIR/policy.json"
RESOLVED_CFG_FPATH="$WORKDIR/resolved_effective_config.yml"

"$PYTHON_BIN" - <<PY > "$POLICY_FPATH"
import json
policy = {
    "candidate_id": "${V4_VARIANT}_${V4_RUN_TAG}_${V4_INPUT_HW// /x}",
    "variant": "${V4_VARIANT}",
    "run_tag": "${V4_RUN_TAG}",
    "export_input_h": ${INPUT_H},
    "export_input_w": ${INPUT_W},
    "train_resolution_policy": "${V4_TRAIN_POLICY}",
    "requested_train_resolution_min": ${REQUESTED_MIN:-null},
    "requested_train_resolution_max": ${REQUESTED_MAX:-null},
    "multiscale_base_size": ${MS_BASE},
    "multiscale_repeat": ${MS_REPEAT},
    "multiscale_stop_epoch": ${V4_MULTISCALE_STOP_EPOCH},
    "tile_training_policy": "tile_g${V4_TILE_GRID}_overlap${V4_TILE_OVERLAP}_out${V4_TILE_OUTPUT_DIM}",
    "tile_grid": ${V4_TILE_GRID},
    "tile_overlap": ${V4_TILE_OVERLAP},
    "tile_output_dim": ${V4_TILE_OUTPUT_DIM},
    "train_batch": ${V4_TRAIN_BATCH},
    "val_batch": ${V4_VAL_BATCH},
    "num_epochs": ${V4_NUM_EPOCHS},
    "lr": ${V4_LR},
    "backbone_lr": ${V4_BACKBONE_LR},
    "use_amp": "${V4_USE_AMP}",
    "init_ckpt": "${V4_DEIMV2_INIT_CKPT}",
    "generated_train_cfg": "${GENERATED_CFG_FPATH}",
}

# Compute the actual scale list the trainer will see, mirroring
# tpl/DEIMv2/engine/data/dataloader.py:generate_scales.
def generate_scales(base_size, base_size_repeat):
    if not base_size_repeat:
        return [int(base_size)]
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales  = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales

scales = generate_scales(int(${MS_BASE}), int(${MS_REPEAT}) or None)
policy["effective_train_scales"] = sorted(set(scales))
policy["effective_train_scale_min"] = min(scales)
policy["effective_train_scale_max"] = max(scales)
print(json.dumps(policy, indent=2))
PY

# Dump the fully-resolved DEIMv2 config (with __include__ expansion) so
# we can see what train transforms / collate actually got. This catches
# the "I set eval_spatial_size but did the train pipeline actually
# resize there?" class of confusion.
"$PYTHON_BIN" - <<PY > "$RESOLVED_CFG_FPATH" 2>/dev/null || \
    echo "  (could not import engine.core.YAMLConfig; resolved config dump skipped)"
import os, sys, yaml
sys.path.insert(0, os.environ['SHITSPOTTER_DEIMV2_REPO_DPATH'])
from engine.core import YAMLConfig
cfg = YAMLConfig("$GENERATED_CFG_FPATH")
print(yaml.safe_dump(cfg.yaml_cfg, sort_keys=False))
PY

echo "=== mobile_app_training_v4 / 02 train $V4_VARIANT ==="
v4_print_env
printf '  %-32s %s\n' "WORKDIR"             "$WORKDIR"
printf '  %-32s %s\n' "UPSTREAM_CFG_FPATH"  "$UPSTREAM_CFG_FPATH"
printf '  %-32s %s\n' "GENERATED_CFG_FPATH" "$GENERATED_CFG_FPATH"
printf '  %-32s %s\n' "TRAIN_MSCOCO"        "$TRAIN_MSCOCO_FPATH"
printf '  %-32s %s\n' "VALI_MSCOCO"         "$VALI_MSCOCO_FPATH"
printf '  %-32s %s\n' "EXPORT_INPUT_HW"     "$V4_INPUT_HW"
printf '  %-32s %s\n' "TRAIN_POLICY"        "$V4_TRAIN_POLICY  (base=$MS_BASE, repeat=$MS_REPEAT, stop_epoch=$V4_MULTISCALE_STOP_EPOCH)"
printf '  %-32s %s\n' "POLICY_FPATH"        "$POLICY_FPATH"
printf '  %-32s %s\n' "RESOLVED_CFG_FPATH"  "$RESOLVED_CFG_FPATH"
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

# Raise per-process file-descriptor limit before launching the torch
# trainer. The default 1024 is too low for typical (workers x batch x
# shared-tensor) IPC traffic — torch.multiprocessing.reduce_storage
# opens a unix-domain socket per shared tensor, and we'd hit
# `OSError: [Errno 24] Too many open files` deep inside DupFd a few
# iterations into training. 65536 is well below the kernel default
# hard limit (~1M on modern Linux) and well above what even an
# 8-worker heavy-tensor loader needs.
#
# Override with V4_FD_LIMIT in env if your kernel has a lower cap.
V4_FD_LIMIT="${V4_FD_LIMIT:-65536}"
_v4_current_nofile=$(ulimit -n 2>/dev/null || echo 0)
if [ "$_v4_current_nofile" -lt "$V4_FD_LIMIT" ] 2>/dev/null; then
    if ulimit -n "$V4_FD_LIMIT" 2>/dev/null; then
        echo "  raised RLIMIT_NOFILE: $_v4_current_nofile -> $(ulimit -n)"
    else
        echo "  WARNING: failed to raise RLIMIT_NOFILE from $_v4_current_nofile to $V4_FD_LIMIT" >&2
        echo "    your shell hard-limit may be lower; check 'ulimit -Hn'" >&2
    fi
fi

# Optional second-line defense: switch torch.multiprocessing's tensor
# sharing from `file_descriptor` (one FD per shared tensor) to
# `file_system` (uses /tmp shared memory, no FD per tensor). Slower
# but FD-bounded. Enable with V4_TORCH_MP_SHARING=file_system if even
# 65536 FDs isn't enough for your batch x worker config.
#
# We inject this via PYTHONSTARTUP because DEIMv2's train.py doesn't
# expose the knob directly. PYTHONSTARTUP runs in interactive shells
# only; for scripts we use -c to prepend the set call. Easiest: a
# wrapper module via PYTHONPATH + sitecustomize.
V4_TORCH_MP_SHARING="${V4_TORCH_MP_SHARING:-}"
if [ -n "$V4_TORCH_MP_SHARING" ]; then
    _v4_sitecust_dpath="$WORKDIR/_v4_sitecustomize"
    mkdir -p "$_v4_sitecust_dpath"
    cat > "$_v4_sitecust_dpath/sitecustomize.py" <<PY
"""Auto-injected by mobile_app_training_v4: set torch IPC sharing
strategy before any user code imports torch."""
try:
    import torch.multiprocessing as _v4_mp
    _v4_mp.set_sharing_strategy("$V4_TORCH_MP_SHARING")
except Exception:
    pass
PY
    export PYTHONPATH="$_v4_sitecust_dpath${PYTHONPATH:+:$PYTHONPATH}"
    echo "  torch.multiprocessing sharing strategy = $V4_TORCH_MP_SHARING"
fi

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

_v4_train_failed() {
    cat <<EOF >&2

[v4 trainer] DEIMv2 train.py exited non-zero. Common causes & fixes:

  - CUDA OOM:  retry with V4_TRAIN_BATCH=$((V4_TRAIN_BATCH / 2)) (half the current batch).
              Multi-GPU (V4_NUM_GPUS=N) only helps when the GPUs have matched
              PCIe bandwidth — gradient all-reduce runs at the slowest peer,
              and a 2x-PCIe second GPU will tank throughput vs single-GPU.
  - 'collate_fn' TypeError:  generated train.yml has wrong YAML indent
                              (rare since the indent fix; check $GENERATED_CFG_FPATH).
  - 'pos_embed' shape mismatch:  HGNetv2 encoder doesn't support multi-scale;
                                  set V4_TRAIN_POLICY=fixed.
  - 'Too many open files':  raise V4_FD_LIMIT or set V4_TORCH_MP_SHARING=file_system.

Logs:  $WORKDIR/  +  generated cfg at $GENERATED_CFG_FPATH
EOF
    return 1
}

if [ "$V4_NUM_GPUS" -gt 1 ]; then
    "$PYTHON_BIN" -m torch.distributed.run \
        --master_port "${V4_MASTER_PORT:-29500}" \
        --nproc_per_node "$V4_NUM_GPUS" \
        "${TRAIN_ARGS[@]}" || _v4_train_failed
else
    "$PYTHON_BIN" "${TRAIN_ARGS[@]}" || _v4_train_failed
fi

echo
echo "Training complete:"
printf '  %-32s %s\n' "WORKDIR"   "$WORKDIR"
printf '  %-32s %s\n' "BEST_CKPT" "$BEST_CKPT"
printf '  %-32s %s\n' "LAST_CKPT" "$LAST_CKPT"
echo
echo "Next: bash $V4_DEV_DPATH/03_export_onnx.sh $V4_VARIANT $V4_RUN_TAG $V4_INPUT_HW"
