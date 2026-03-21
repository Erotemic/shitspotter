#!/bin/bash
set -euo pipefail

# v6 is intentionally narrow:
# - keep the v5 detector data semantics (offline resize on, simplify off)
# - keep SAM2 fixed by reusing the tuned v5 segmenter checkpoint
# - change only detector optimization, then select the detector checkpoint
#   using combined raw validation before touching test

canonical_existing_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
    (cd "$path" && pwd -P)
}

require_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
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

read_ap() {
    local metrics_fpath="$1"
    "$PYTHON_BIN" - "$metrics_fpath" <<'PY'
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
print(f'{float(ap):.6f}')
PY
}

resolve_candidate_checkpoint() {
    local candidate_id="$1"
    local ckpt_fpath
    case "$candidate_id" in
        *.pth)
            ckpt_fpath="$DETECTOR_WORKDIR/$candidate_id"
            ;;
        *)
            ckpt_fpath="$DETECTOR_WORKDIR/${candidate_id}.pth"
            ;;
    esac
    [ -f "$ckpt_fpath" ] || return 1
    printf '%s\n' "$ckpt_fpath"
}

build_package() {
    local package_fpath="$1"
    local detector_ckpt="$2"
    local segmenter_ckpt="$3"
    local metadata_name="$4"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$detector_ckpt" \
        --segmenter_checkpoint_fpath "$segmenter_ckpt" \
        --metadata_name "$metadata_name" >/dev/null
}

evaluate_detector_split() {
    local src_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    local score_thresh="$4"
    local nms_thresh="$5"
    mkdir -p "$out_dpath/eval"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        "$src_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred_boxes.kwcoco.zip" \
        --score_thresh "$score_thresh" \
        --nms_thresh "$nms_thresh"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$src_fpath" \
        --pred_dataset "$out_dpath/pred_boxes.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
}

evaluate_combined_raw_split() {
    local src_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    mkdir -p "$out_dpath/eval"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict \
        "$src_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred.kwcoco.zip" \
        --crop_padding 0 \
        --polygon_simplify 0 \
        --min_component_area 0 \
        --keep_largest_component False
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$src_fpath" \
        --pred_dataset "$out_dpath/pred.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
}

select_best_candidate() {
    local summary_fpath="$1"
    "$PYTHON_BIN" - "$summary_fpath" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
rows = []
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        try:
            row['_combined_vali_ap'] = float(row['combined_vali_ap'])
            row['_detector_vali_ap'] = float(row['detector_vali_ap'])
        except Exception:
            continue
        rows.append(row)

if not rows:
    raise SystemExit('No candidate rows found in summary')

rows.sort(
    key=lambda row: (
        row['_combined_vali_ap'],
        row['_detector_vali_ap'],
    ),
    reverse=True,
)
best = rows[0]
print('\t'.join([
    best['candidate_id'],
    best['ckpt_fpath'],
    best['detector_vali_ap'],
    best['combined_vali_ap'],
]))
PY
}

print_top_candidates() {
    local summary_fpath="$1"
    local limit="${2:-10}"
    "$PYTHON_BIN" - "$summary_fpath" "$limit" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
limit = int(sys.argv[2])
rows = []
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        try:
            row['_combined_vali_ap'] = float(row['combined_vali_ap'])
            row['_detector_vali_ap'] = float(row['detector_vali_ap'])
        except Exception:
            continue
        rows.append(row)
rows.sort(key=lambda row: (row['_combined_vali_ap'], row['_detector_vali_ap']), reverse=True)
print('candidate_id\tdetector_vali_ap\tcombined_vali_ap\tapprox_note')
for row in rows[:limit]:
    print(f"{row['candidate_id']}\t{row['detector_vali_ap']}\t{row['combined_vali_ap']}\t{row['selection_note']}")
PY
}

write_selected_manifest() {
    local manifest_fpath="$1"
    local selected_candidate_id="$2"
    local selected_ckpt_fpath="$3"
    local detector_vali_ap="$4"
    local combined_vali_ap="$5"
    local final_detector_test_ap="$6"
    local final_combined_test_ap="$7"
    local final_package_fpath="$8"
    local tmp_fpath="${manifest_fpath}.tmp"
    cat > "$tmp_fpath" <<EOF
version: v6
selected_candidate_id: $selected_candidate_id
selected_detector_checkpoint_fpath: $selected_ckpt_fpath
frozen_segmenter_checkpoint_fpath: $SAM2_TRAINED_CKPT
detector_vali_ap: $detector_vali_ap
combined_vali_ap: $combined_vali_ap
detector_test_ap: $final_detector_test_ap
combined_test_ap: $final_combined_test_ap
package_fpath: $final_package_fpath
timestamp_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
EOF
    mv "$tmp_fpath" "$manifest_fpath"
}

PYTHON_BIN="${PYTHON_BIN:-python}"

REPO_DPATH="$(canonical_existing_path /home/joncrall/code/shitspotter)"
DATA_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_dvc)"
EXPT_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_expt_dvc)"

TRAIN_FPATH="$DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip"
VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip"

SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"

DEIMV2_INIT_CKPT="$DATA_DPATH/models/foundation_detseg_v3/deimv2/deimv2_dinov3_m_coco.pth"
SAM2_TRAINED_CKPT="${SAM2_TRAINED_CKPT:-$EXPT_DPATH/foundation_detseg_v3/v5/train_segmenter_sam2_1_hiera_base_plus/checkpoints/checkpoint.pt}"

V6_ROOT="$EXPT_DPATH/foundation_detseg_v3/v6"
DETECTOR_WORKDIR="$V6_ROOT/train_detector_deimv2_m"
CHECKPOINT_SELECT_DPATH="$V6_ROOT/checkpoint_select"
CHECKPOINT_PACKAGE_DPATH="$CHECKPOINT_SELECT_DPATH/packages"
CHECKPOINT_EVAL_DPATH="$CHECKPOINT_SELECT_DPATH/evals"
SUMMARY_FPATH="$CHECKPOINT_SELECT_DPATH/summary.tsv"
SELECTED_MANIFEST_FPATH="$V6_ROOT/selected_detector_checkpoint.yaml"

PACKAGE_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3/packages"
FINAL_TUNED_PACKAGE_FPATH="$PACKAGE_DPATH/v6_deimv2_m_sam2_1_hiera_base_plus_tuned.yaml"

FINAL_DETECTOR_TEST_DPATH_BASE="$V6_ROOT/eval_detector_only/test"
FINAL_COMBINED_TEST_DPATH_BASE="$V6_ROOT/eval_detector_segmenter/test/tuned_raw"

DETECTOR_SCORE_THRESH="${DETECTOR_SCORE_THRESH:-0.2}"
DETECTOR_NMS_THRESH="${DETECTOR_NMS_THRESH:-0.5}"
FORCE_DETECTOR_RERUN="${FORCE_DETECTOR_RERUN:-False}"
FORCE_CANDIDATE_EVALS="${FORCE_CANDIDATE_EVALS:-False}"
FORCE_FINAL_EVALS="${FORCE_FINAL_EVALS:-False}"

DEIMV2_NUM_GPUS="${DEIMV2_NUM_GPUS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-24}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-48}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-0}"
USE_AMP="${USE_AMP:-True}"
ENABLE_RESIZE_PREPROCESS="${ENABLE_RESIZE_PREPROCESS:-True}"
FORCE_RESIZE_PREPROCESS="${FORCE_RESIZE_PREPROCESS:-False}"
RESIZE_MAX_DIM="${RESIZE_MAX_DIM:-640}"
ENABLE_SIMPLIFY_PREPROCESS="False"

V6_MAIN_LR_SCALE="${V6_MAIN_LR_SCALE:-1.0}"
V6_BACKBONE_LR_SCALE="${V6_BACKBONE_LR_SCALE:-0.25}"

CANDIDATES=(${V6_CANDIDATES:-checkpoint0019 checkpoint0024 checkpoint0029 checkpoint0034 checkpoint0039 checkpoint0044 checkpoint0049 checkpoint0054 checkpoint0059 best_stg1 last})

for required in \
    "$REPO_DPATH" \
    "$DATA_DPATH" \
    "$EXPT_DPATH" \
    "$TRAIN_FPATH" \
    "$VALI_FPATH" \
    "$TEST_FPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH" \
    "$SHITSPOTTER_SAM2_REPO_DPATH" \
    "$DEIMV2_INIT_CKPT" \
    "$SAM2_TRAINED_CKPT"; do
    require_path "$required"
done

mkdir -p "$CHECKPOINT_PACKAGE_DPATH" "$CHECKPOINT_EVAL_DPATH"

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_DEIMV2_REPO_DPATH
export SHITSPOTTER_SAM2_REPO_DPATH
export SHITSPOTTER_MASKDINO_REPO_DPATH
export PYTHONPATH="$REPO_DPATH${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${DEIMV2_CONFIG_OVERRIDES:-}" ]; then
    read -r BACKBONE_LR MAIN_LR <<EOF
$("$PYTHON_BIN" - <<PY
train_batch = int("$TRAIN_BATCH_SIZE")
base_batch = 32.0
main_scale = float("$V6_MAIN_LR_SCALE")
backbone_scale = float("$V6_BACKBONE_LR_SCALE")
main_lr = 5e-4 * (train_batch / base_batch) * main_scale
backbone_lr = 2.5e-5 * (train_batch / base_batch) * backbone_scale
print(f"{backbone_lr:.10f} {main_lr:.10f}")
PY
)
EOF
    DEIMV2_CONFIG_OVERRIDES="$(cat <<EOF
use_amp: ${USE_AMP}
train_dataloader:
  total_batch_size: ${TRAIN_BATCH_SIZE}
  num_workers: ${TRAIN_NUM_WORKERS}
val_dataloader:
  total_batch_size: ${VAL_BATCH_SIZE}
  num_workers: ${VAL_NUM_WORKERS}
optimizer:
  params:
    - params: '^(?=.*.dinov3)(?!.*(?:norm|bn|bias)).*$'
      lr: ${BACKBONE_LR}
    - params: '^(?=.*.dinov3)(?=.*(?:norm|bn|bias)).*$'
      lr: ${BACKBONE_LR}
      weight_decay: 0.0
    - params: '^(?=.*(?:sta|encoder|decoder))(?=.*(?:norm|bn|bias)).*$'
      weight_decay: 0.0
  lr: ${MAIN_LR}
EOF
)"
fi

echo "v6 foundation_detseg_v3 detector-optimization experiment"
printf '  %-32s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-32s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-32s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-32s %s\n' "TRAIN_FPATH" "$TRAIN_FPATH"
printf '  %-32s %s\n' "VALI_FPATH" "$VALI_FPATH"
printf '  %-32s %s\n' "TEST_FPATH" "$TEST_FPATH"
printf '  %-32s %s\n' "V6_ROOT" "$V6_ROOT"
printf '  %-32s %s\n' "DETECTOR_WORKDIR" "$DETECTOR_WORKDIR"
printf '  %-32s %s\n' "SAM2_TRAINED_CKPT" "$SAM2_TRAINED_CKPT"
printf '  %-32s %s\n' "TRAIN_BATCH_SIZE" "$TRAIN_BATCH_SIZE"
printf '  %-32s %s\n' "VAL_BATCH_SIZE" "$VAL_BATCH_SIZE"
printf '  %-32s %s\n' "V6_MAIN_LR_SCALE" "$V6_MAIN_LR_SCALE"
printf '  %-32s %s\n' "V6_BACKBONE_LR_SCALE" "$V6_BACKBONE_LR_SCALE"
printf '  %-32s %s\n' "DETECTOR_SCORE_THRESH" "$DETECTOR_SCORE_THRESH"
printf '  %-32s %s\n' "DETECTOR_NMS_THRESH" "$DETECTOR_NMS_THRESH"
printf '  %-32s %s\n' "CANDIDATES" "${CANDIDATES[*]}"

echo
echo "=== Train detector with gentler backbone LR ==="
export TRAIN_FPATH
export VALI_FPATH
export WORKDIR="$DETECTOR_WORKDIR"
export VARIANT="deimv2_m"
export DEIMV2_INIT_CKPT
export DEIMV2_NUM_GPUS
export TRAIN_BATCH_SIZE
export VAL_BATCH_SIZE
export TRAIN_NUM_WORKERS
export VAL_NUM_WORKERS
export USE_AMP
export ENABLE_RESIZE_PREPROCESS
export FORCE_RESIZE_PREPROCESS
export RESIZE_MAX_DIM
export ENABLE_SIMPLIFY_PREPROCESS
export DEIMV2_CONFIG_OVERRIDES

DETECTOR_SENTINEL_FPATH="$DETECTOR_WORKDIR/last.pth"
if [ -f "$DETECTOR_SENTINEL_FPATH" ] && ! is_truthy "$FORCE_DETECTOR_RERUN"; then
    echo "Reusing existing detector training outputs in: $DETECTOR_WORKDIR"
else
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_deimv2_detector.sh"
fi

require_path "$DETECTOR_WORKDIR"

echo
echo "=== Sweep checkpoint shortlist on validation ==="
printf 'candidate_id\tckpt_fpath\tdetector_vali_ap\tcombined_vali_ap\tselection_note\n' > "$SUMMARY_FPATH"
for candidate_id in "${CANDIDATES[@]}"; do
    ckpt_fpath="$(resolve_candidate_checkpoint "$candidate_id" || true)"
    if [ -z "${ckpt_fpath:-}" ]; then
        echo "Skipping missing candidate: $candidate_id"
        continue
    fi
    candidate_package_fpath="$CHECKPOINT_PACKAGE_DPATH/${candidate_id}.yaml"
    build_package "$candidate_package_fpath" "$ckpt_fpath" "$SAM2_TRAINED_CKPT" "v6_${candidate_id}"

    detector_vali_dpath="$CHECKPOINT_EVAL_DPATH/${candidate_id}/detector_vali"
    combined_vali_dpath="$CHECKPOINT_EVAL_DPATH/${candidate_id}/combined_raw_vali"

    if is_truthy "$FORCE_CANDIDATE_EVALS" || ! have_metrics "$detector_vali_dpath"; then
        evaluate_detector_split "$VALI_FPATH" "$candidate_package_fpath" "$detector_vali_dpath" "$DETECTOR_SCORE_THRESH" "$DETECTOR_NMS_THRESH"
    else
        echo "Reusing detector validation metrics for $candidate_id"
    fi

    if is_truthy "$FORCE_CANDIDATE_EVALS" || ! have_metrics "$combined_vali_dpath"; then
        evaluate_combined_raw_split "$VALI_FPATH" "$candidate_package_fpath" "$combined_vali_dpath"
    else
        echo "Reusing combined validation metrics for $candidate_id"
    fi

    detector_vali_ap="$(read_ap "$detector_vali_dpath/eval/detect_metrics.json")"
    combined_vali_ap="$(read_ap "$combined_vali_dpath/eval/detect_metrics.json")"
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "$candidate_id" \
        "$ckpt_fpath" \
        "$detector_vali_ap" \
        "$combined_vali_ap" \
        "combined_raw_vali_primary" >> "$SUMMARY_FPATH"
    printf '  %-16s det_vali=%s comb_vali=%s\n' "$candidate_id" "$detector_vali_ap" "$combined_vali_ap"
done

echo
echo "Top v6 checkpoint candidates:"
print_top_candidates "$SUMMARY_FPATH" 12

read -r SELECTED_CANDIDATE_ID SELECTED_CKPT_FPATH SELECTED_DETECTOR_VALI_AP SELECTED_COMBINED_VALI_AP <<< "$(select_best_candidate "$SUMMARY_FPATH")"

echo
echo "=== Selected detector checkpoint ==="
printf '  %-32s %s\n' "SELECTED_CANDIDATE_ID" "$SELECTED_CANDIDATE_ID"
printf '  %-32s %s\n' "SELECTED_CKPT_FPATH" "$SELECTED_CKPT_FPATH"
printf '  %-32s %s\n' "SELECTED_DETECTOR_VALI_AP" "$SELECTED_DETECTOR_VALI_AP"
printf '  %-32s %s\n' "SELECTED_COMBINED_VALI_AP" "$SELECTED_COMBINED_VALI_AP"

echo
echo "=== Build final tuned package ==="
build_package "$FINAL_TUNED_PACKAGE_FPATH" "$SELECTED_CKPT_FPATH" "$SAM2_TRAINED_CKPT" "v6_deimv2_m_sam2_1_hiera_base_plus_tuned"

SELECTED_FINAL_ROOT="$V6_ROOT/final_eval/${SELECTED_CANDIDATE_ID}"
FINAL_DETECTOR_TEST_DPATH="$SELECTED_FINAL_ROOT/detector_test"
FINAL_COMBINED_TEST_DPATH="$SELECTED_FINAL_ROOT/combined_raw_test"

echo
echo "=== Evaluate selected detector on test ==="
if is_truthy "$FORCE_FINAL_EVALS" || ! have_metrics "$FINAL_DETECTOR_TEST_DPATH"; then
    evaluate_detector_split "$TEST_FPATH" "$FINAL_TUNED_PACKAGE_FPATH" "$FINAL_DETECTOR_TEST_DPATH" "$DETECTOR_SCORE_THRESH" "$DETECTOR_NMS_THRESH"
else
    echo "Reusing final detector test metrics: $FINAL_DETECTOR_TEST_DPATH/eval/detect_metrics.json"
fi

echo
echo "=== Evaluate selected detector + frozen SAM on test ==="
if is_truthy "$FORCE_FINAL_EVALS" || ! have_metrics "$FINAL_COMBINED_TEST_DPATH"; then
    evaluate_combined_raw_split "$TEST_FPATH" "$FINAL_TUNED_PACKAGE_FPATH" "$FINAL_COMBINED_TEST_DPATH"
else
    echo "Reusing final combined test metrics: $FINAL_COMBINED_TEST_DPATH/eval/detect_metrics.json"
fi

FINAL_DETECTOR_TEST_AP="$(read_ap "$FINAL_DETECTOR_TEST_DPATH/eval/detect_metrics.json")"
FINAL_COMBINED_TEST_AP="$(read_ap "$FINAL_COMBINED_TEST_DPATH/eval/detect_metrics.json")"

write_selected_manifest \
    "$SELECTED_MANIFEST_FPATH" \
    "$SELECTED_CANDIDATE_ID" \
    "$SELECTED_CKPT_FPATH" \
    "$SELECTED_DETECTOR_VALI_AP" \
    "$SELECTED_COMBINED_VALI_AP" \
    "$FINAL_DETECTOR_TEST_AP" \
    "$FINAL_COMBINED_TEST_AP" \
    "$FINAL_TUNED_PACKAGE_FPATH"

echo
echo "=== v6 summary ==="
printf '  %-32s %s\n' "selected_candidate_id" "$SELECTED_CANDIDATE_ID"
printf '  %-32s ap=%s\n' "detector_only_vali" "$SELECTED_DETECTOR_VALI_AP"
printf '  %-32s ap=%s\n' "combined_tuned_raw_vali" "$SELECTED_COMBINED_VALI_AP"
printf '  %-32s ap=%s\n' "detector_only_test" "$FINAL_DETECTOR_TEST_AP"
printf '  %-32s ap=%s\n' "combined_tuned_raw_test" "$FINAL_COMBINED_TEST_AP"

echo
echo "v6 run completed"
printf '  %-32s %s\n' "SELECTED_DETECTOR_CKPT" "$SELECTED_CKPT_FPATH"
printf '  %-32s %s\n' "FROZEN_SAM2_CKPT" "$SAM2_TRAINED_CKPT"
printf '  %-32s %s\n' "FINAL_TUNED_PACKAGE_FPATH" "$FINAL_TUNED_PACKAGE_FPATH"
printf '  %-32s %s\n' "SUMMARY_FPATH" "$SUMMARY_FPATH"
printf '  %-32s %s\n' "SELECTED_MANIFEST_FPATH" "$SELECTED_MANIFEST_FPATH"
