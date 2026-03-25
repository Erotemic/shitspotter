#!/bin/bash
set -euo pipefail

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_DPATH="$(small_data_repo_dpath)"
DATA_DPATH="$(small_data_data_dpath)"
EXPT_DPATH="$(small_data_expt_dpath)"

BENCHMARK_ROOT="${BENCHMARK_ROOT:-$EXPT_DPATH/small_data_tuning/dino_detector_benchmark}"
BENCHMARK_MANIFEST_FPATH="$BENCHMARK_ROOT/benchmark_manifest.json"
RUN_ROOT="$BENCHMARK_ROOT/runs/deimv2"
DEIMV2_NUM_GPUS="${DEIMV2_NUM_GPUS:-2}"
DEIMV2_ALLOW_SINGLE_GPU_FALLBACK="${DEIMV2_ALLOW_SINGLE_GPU_FALLBACK:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
USE_AMP="${USE_AMP:-True}"
FORCE_DETECTOR_RERUN="${FORCE_DETECTOR_RERUN:-False}"
DEIMV2_SCORE_THRESH="${DEIMV2_SCORE_THRESH:-0.2}"
DEIMV2_NMS_THRESH="${DEIMV2_NMS_THRESH:-0.5}"
DEIMV2_CANDIDATES=(${DEIMV2_CANDIDATES:-checkpoint0019 checkpoint0024 checkpoint0029 checkpoint0034 checkpoint0039 checkpoint0044 checkpoint0049 checkpoint0054 checkpoint0059 best_stg1 last})
DEIMV2_CONFIG_SPECS="${DEIMV2_CONFIG_SPECS:-baseline|1.0|1.0|24|True low_lr_all_0p7|0.7|0.7|24|True low_lr_all_0p5|0.5|0.5|24|True backbone_tiny_0p25|1.0|0.25|24|True backbone_tiny_0p10|1.0|0.10|24|True small_batch16|1.0|1.0|16|True}"

read_metric() {
    local metrics_fpath="$1"
    local key="$2"
    "$PYTHON_BIN" - "$metrics_fpath" "$key" <<'PY'
import json
import sys

metrics_fpath = sys.argv[1]
metric_key = sys.argv[2]
data = json.loads(open(metrics_fpath, 'r').read())
root = next(iter(data.values()))
if metric_key == 'poop_ap':
    val = root['ovr_measures']['poop']['ap']
elif metric_key == 'nocls_ap':
    val = root['nocls_measures']['ap']
else:
    raise KeyError(metric_key)
print(f'{float(val):.6f}')
PY
}

is_finite_metric() {
    local value="$1"
    "$PYTHON_BIN" - "$value" <<'PY'
import math
import sys

value = float(sys.argv[1])
raise SystemExit(0 if math.isfinite(value) else 1)
PY
}

metric_is_greater() {
    local left="$1"
    local right="$2"
    "$PYTHON_BIN" - "$left" "$right" <<'PY'
import math
import sys

left = float(sys.argv[1])
right = float(sys.argv[2])
if not math.isfinite(left):
    raise SystemExit(1)
if not math.isfinite(right):
    raise SystemExit(0)
raise SystemExit(0 if left > right else 1)
PY
}

resolve_candidate_checkpoint() {
    local detector_workdir="$1"
    local candidate_id="$2"
    local ckpt_fpath
    case "$candidate_id" in
        *.pth) ckpt_fpath="$detector_workdir/$candidate_id" ;;
        *) ckpt_fpath="$detector_workdir/${candidate_id}.pth" ;;
    esac
    [ -f "$ckpt_fpath" ] || return 1
    printf '%s\n' "$ckpt_fpath"
}

build_package() {
    local package_fpath="$1"
    local ckpt_fpath="$2"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$ckpt_fpath" \
        --segmenter_checkpoint_fpath "$REPO_DPATH/tpl/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt" \
        --metadata_name "benchmark_$(basename "$package_fpath" .yaml)" >/dev/null
}

evaluate_package() {
    local true_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    mkdir -p "$out_dpath/eval"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        "$true_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred_boxes.kwcoco.zip" \
        --score_thresh "$DEIMV2_SCORE_THRESH" \
        --nms_thresh "$DEIMV2_NMS_THRESH"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$true_fpath" \
        --pred_dataset "$out_dpath/pred_boxes.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
}

if [ ! -f "$BENCHMARK_MANIFEST_FPATH" ]; then
    echo "Expected benchmark manifest missing: $BENCHMARK_MANIFEST_FPATH" >&2
    echo "Run prepare_dino_detector_benchmark.sh first." >&2
    exit 1
fi

mapfile -t TRAIN_ROWS < <("$PYTHON_BIN" - "$BENCHMARK_MANIFEST_FPATH" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], 'r').read())
for subset_name, subset in sorted(data['train_subsets'].items(), key=lambda item: item[1]['train_size']):
    print('\t'.join([
        subset_name,
        str(subset['train_size']),
        subset['prepared_kwcoco_fpath'],
    ]))
PY
)

VALI_FPATH="$("$PYTHON_BIN" - "$BENCHMARK_MANIFEST_FPATH" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], 'r').read())
print(data['eval_sets']['vali']['prepared_kwcoco_fpath'])
PY
)"
TEST_FPATH="$("$PYTHON_BIN" - "$BENCHMARK_MANIFEST_FPATH" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], 'r').read())
print(data['eval_sets']['test']['prepared_kwcoco_fpath'])
PY
)"

for row in "${TRAIN_ROWS[@]}"; do
    IFS=$'\t' read -r subset_name train_size train_fpath <<<"$row"
    for config_spec in ${DEIMV2_CONFIG_SPECS}; do
        IFS='|' read -r config_tag config_main_lr_scale config_backbone_lr_scale config_train_batch_size config_use_amp <<<"$config_spec"
        config_tag="${config_tag:-baseline}"
        config_main_lr_scale="${config_main_lr_scale:-1.0}"
        config_backbone_lr_scale="${config_backbone_lr_scale:-1.0}"
        config_train_batch_size="${config_train_batch_size:-$TRAIN_BATCH_SIZE}"
        config_use_amp="${config_use_amp:-$USE_AMP}"
        config_val_batch_size="$VAL_BATCH_SIZE"
        read -r backbone_lr main_lr <<EOF
$("$PYTHON_BIN" - <<PY
train_batch = int("$config_train_batch_size")
base_batch = 32.0
main_scale = float("$config_main_lr_scale")
backbone_scale = float("$config_backbone_lr_scale")
main_lr = 5e-4 * (train_batch / base_batch) * main_scale
backbone_lr = 2.5e-5 * (train_batch / base_batch) * backbone_scale
print(f"{backbone_lr:.10f} {main_lr:.10f}")
PY
)
EOF
        config_overrides="$(cat <<EOF
use_amp: ${config_use_amp}
train_dataloader:
  total_batch_size: ${config_train_batch_size}
  num_workers: ${TRAIN_NUM_WORKERS}
val_dataloader:
  total_batch_size: ${config_val_batch_size}
  num_workers: ${VAL_NUM_WORKERS}
optimizer:
  params:
    - params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: ${backbone_lr}
    - params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: ${backbone_lr}
      weight_decay: 0.0
    - params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0
  lr: ${main_lr}
EOF
)"

        run_dpath="$RUN_ROOT/$config_tag/$subset_name"
        detector_workdir="$run_dpath/train_detector_deimv2_m"
        checkpoint_dpath="$run_dpath/checkpoint_select"
        package_dpath="$checkpoint_dpath/packages"
        eval_dpath="$checkpoint_dpath/evals"
        summary_fpath="$run_dpath/summary.tsv"
        run_manifest_fpath="$run_dpath/run_manifest.json"
        mkdir -p "$package_dpath" "$eval_dpath"

        printf 'DEIMv2 detector benchmark\n'
        printf '  %-22s %s\n' "CONFIG_TAG" "$config_tag"
        printf '  %-22s %s\n' "SUBSET" "$subset_name"
        printf '  %-22s %s\n' "TRAIN_SIZE" "$train_size"
        printf '  %-22s %s\n' "RUN_DPATH" "$run_dpath"

        cat > "$run_manifest_fpath" <<EOF
{
  "model_family": "deimv2",
  "config_tag": "$config_tag",
  "benchmark_manifest_fpath": "$BENCHMARK_MANIFEST_FPATH",
  "subset_name": "$subset_name",
  "train_size": $train_size,
  "train_fpath": "$train_fpath",
  "vali_fpath": "$VALI_FPATH",
  "test_fpath": "$TEST_FPATH",
  "run_dpath": "$run_dpath",
  "detector_workdir": "$detector_workdir",
  "score_thresh": $DEIMV2_SCORE_THRESH,
  "nms_thresh": $DEIMV2_NMS_THRESH,
  "candidate_ids": ["$(printf '%s","' "${DEIMV2_CANDIDATES[@]}" | sed 's/,$//')"],
  "num_gpus_requested": $DEIMV2_NUM_GPUS,
  "allow_single_gpu_fallback": "$DEIMV2_ALLOW_SINGLE_GPU_FALLBACK",
  "train_batch_size": $config_train_batch_size,
  "vali_batch_size": $config_val_batch_size,
  "use_amp": "$config_use_amp",
  "main_lr_scale": $config_main_lr_scale,
  "backbone_lr_scale": $config_backbone_lr_scale,
  "resolved_main_lr": $main_lr,
  "resolved_backbone_lr": $backbone_lr
}
EOF

        export SHITSPOTTER_DPATH="$REPO_DPATH"
        export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
        export DVC_DATA_DPATH="$DATA_DPATH"
        export DVC_EXPT_DPATH="$EXPT_DPATH"
        export SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
        export SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
        export SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"
        export TRAIN_FPATH="$train_fpath"
        export VALI_FPATH="$VALI_FPATH"
        export WORKDIR="$detector_workdir"
        export VARIANT="deimv2_m"
        export DEIMV2_INIT_CKPT="$DATA_DPATH/models/foundation_detseg_v3/deimv2/deimv2_dinov3_m_coco.pth"
        export DEIMV2_NUM_GPUS
        export DEIMV2_ALLOW_SINGLE_GPU_FALLBACK
        export TRAIN_BATCH_SIZE="$config_train_batch_size"
        export VAL_BATCH_SIZE="$config_val_batch_size"
        export TRAIN_NUM_WORKERS
        export VAL_NUM_WORKERS
        export USE_AMP="$config_use_amp"
        export ENABLE_RESIZE_PREPROCESS="False"
        export ENABLE_SIMPLIFY_PREPROCESS="False"
        export DEIMV2_CONFIG_OVERRIDES="$config_overrides"

        if [ ! -f "$detector_workdir/last.pth" ] || [ "$FORCE_DETECTOR_RERUN" = "True" ]; then
            bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
        fi

        printf 'candidate_id\ttrain_size\tpoop_vali_ap\tnocls_vali_ap\tpoop_test_ap\tnocls_test_ap\tselected\tckpt_fpath\trun_dpath\tsummary_fpath\tconfig_tag\n' > "$summary_fpath"
        best_candidate=""
        best_ckpt=""
        best_poop_vali=""
        for candidate_id in "${DEIMV2_CANDIDATES[@]}"; do
            ckpt_fpath="$(resolve_candidate_checkpoint "$detector_workdir" "$candidate_id" || true)"
            if [ -z "${ckpt_fpath:-}" ]; then
                continue
            fi
            candidate_package_fpath="$package_dpath/${candidate_id}.yaml"
            candidate_eval_dpath="$eval_dpath/${candidate_id}/vali"
            build_package "$candidate_package_fpath" "$ckpt_fpath"
            evaluate_package "$VALI_FPATH" "$candidate_package_fpath" "$candidate_eval_dpath"
            poop_vali_ap="$(read_metric "$candidate_eval_dpath/eval/detect_metrics.json" poop_ap)"
            nocls_vali_ap="$(read_metric "$candidate_eval_dpath/eval/detect_metrics.json" nocls_ap)"
            printf '%s\t%s\t%s\t%s\tNA\tNA\t0\t%s\t%s\t%s\t%s\n' \
                "$candidate_id" "$train_size" "$poop_vali_ap" "$nocls_vali_ap" "$ckpt_fpath" "$run_dpath" "$summary_fpath" "$config_tag" >> "$summary_fpath"
            if is_finite_metric "$poop_vali_ap" && { [ -z "$best_poop_vali" ] || metric_is_greater "$poop_vali_ap" "$best_poop_vali"; }; then
                best_candidate="$candidate_id"
                best_ckpt="$ckpt_fpath"
                best_poop_vali="$poop_vali_ap"
            fi
        done

        if [ -z "$best_ckpt" ]; then
            printf '%s\t%s\tNA\tNA\tNA\tNA\t0\t%s\t%s\t%s\t%s\n' \
                "" \
                "$train_size" "$detector_workdir" "$run_dpath" "$summary_fpath" "$config_tag" >> "$summary_fpath"
            printf 'DEIMv2 benchmark incomplete\n'
            printf '  %-22s %s\n' "CONFIG_TAG" "$config_tag"
            printf '  %-22s %s\n' "SUBSET" "$subset_name"
            printf '  %-22s %s\n' "SUMMARY_FPATH" "$summary_fpath"
            printf '  %-22s %s\n' "REASON" "No finite validation AP for any candidate"
            continue
        fi

        final_package_fpath="$run_dpath/deimv2_selected.yaml"
        test_eval_dpath="$run_dpath/test_eval"
        build_package "$final_package_fpath" "$best_ckpt"
        evaluate_package "$TEST_FPATH" "$final_package_fpath" "$test_eval_dpath"
        poop_test_ap="$(read_metric "$test_eval_dpath/eval/detect_metrics.json" poop_ap)"
        nocls_test_ap="$(read_metric "$test_eval_dpath/eval/detect_metrics.json" nocls_ap)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\t%s\t%s\n' \
            "$best_candidate" "$train_size" "$best_poop_vali" \
            "$(read_metric "$eval_dpath/${best_candidate}/vali/eval/detect_metrics.json" nocls_ap)" \
            "$poop_test_ap" "$nocls_test_ap" "$best_ckpt" "$run_dpath" "$summary_fpath" "$config_tag" >> "$summary_fpath"

        printf 'DEIMv2 benchmark complete\n'
        printf '  %-22s %s\n' "CONFIG_TAG" "$config_tag"
        printf '  %-22s %s\n' "SELECTED_CANDIDATE" "$best_candidate"
        printf '  %-22s %s\n' "SUMMARY_FPATH" "$summary_fpath"
        printf '  %-22s %s\n' "RUN_DPATH" "$run_dpath"
    done
done
