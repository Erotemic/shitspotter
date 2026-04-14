#!/bin/bash
set -euo pipefail

# v9: Full retrain of OpenGroundingDINO (DINOv2) + SAM2 on the new 10671-image
# dataset split (train_imgs10671_b277c63d / vali_imgs1258_577e331c /
# test_imgs121_d39956b1).
#
# Design rationale:
# - OpenGroundingDINO (DINOv2 + BERT text encoder) led DEIMv2 (DINOv3) by
#   ~0.10 AP on the test split, making it the best current detector family.
# - Both the OpenGroundingDINO detector and the SAM2 segmenter are retrained
#   from scratch on the new larger dataset; reusing v5 SAM2 would be unfair to
#   the 2x data increase.
# - Detector hyperparameters carry forward the baseline that was validated in
#   the small-data benchmark (batch=4, lr=0.0001, lr_backbone=1e-05, 15 ep).
# - Checkpoint selection uses combined-raw-vali AP as the primary key, same as
#   v6/v7; test is only touched once at the very end.

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

ensure_segmenter_checkpoint() {
    local ckpt="$SAM2_WORKDIR/checkpoints/checkpoint.pt"
    [ -f "$ckpt" ] || return 1
    printf '%s\n' "$ckpt"
}

build_package() {
    local package_fpath="$1"
    local detector_ckpt="$2"
    local detector_cfg="$3"
    local segmenter_ckpt="$4"
    local metadata_name="$5"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend opengroundingdino_sam2 \
        --detector_preset opengroundingdino_shitspotter \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$detector_ckpt" \
        --detector_config_fpath "$detector_cfg" \
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

have_sseg_metrics() {
    local dpath="$1"
    [ -f "$dpath/sseg_eval/summary_metrics.json" ]
}

rasterize_pred_heatmap() {
    local src_fpath="$1"
    local dst_fpath="$2"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_rasterize_pred_heatmap \
        "$src_fpath" \
        --dst "$dst_fpath"
}

evaluate_segmentation_split() {
    # Run kwcoco.metrics.segmentation_metrics against a simplified GT.
    # Writes a reproduce_sseg_eval.sh script alongside results so the eval
    # can be re-run or inspected independently.
    local true_fpath="$1"
    local pred_heatmap_fpath="$2"
    local out_dpath="$3"
    mkdir -p "$out_dpath/sseg_eval"
    local eval_fpath="$out_dpath/sseg_eval/summary_metrics.json"
    local script_fpath="$out_dpath/sseg_eval/reproduce_sseg_eval.sh"
    # Write reproduce script first so it exists even if the eval fails
    cat > "$script_fpath" <<REPRO
#!/bin/bash
# Auto-generated: re-runs the segmentation evaluation that produced
# ${eval_fpath}
python -m kwcoco.metrics.segmentation_metrics \\
    --true_dataset "${true_fpath}" \\
    --pred_dataset "${pred_heatmap_fpath}" \\
    --eval_dpath "${out_dpath}/sseg_eval" \\
    --eval_fpath "${eval_fpath}" \\
    --score_space image \\
    --draw_curves True \\
    --draw_heatmaps False \\
    --workers 2
REPRO
    chmod +x "$script_fpath"
    "$PYTHON_BIN" -m kwcoco.metrics.segmentation_metrics \
        --true_dataset "$true_fpath" \
        --pred_dataset "$pred_heatmap_fpath" \
        --eval_dpath "$out_dpath/sseg_eval" \
        --eval_fpath "$eval_fpath" \
        --score_space image \
        --draw_curves True \
        --draw_heatmaps False \
        --workers 2
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
    local final_detector_test_simplified_ap="${9:-}"
    local final_combined_test_simplified_ap="${10:-}"
    local tmp_fpath="${manifest_fpath}.tmp"
    cat > "$tmp_fpath" <<EOF
version: v9
selected_candidate_id: $selected_candidate_id
selected_detector_checkpoint_fpath: $selected_ckpt_fpath
detector_config_fpath: $GDINO_CFG_FPATH
tuned_segmenter_checkpoint_fpath: $SAM2_TRAINED_CKPT
detector_vali_ap: $detector_vali_ap
combined_vali_ap: $combined_vali_ap
detector_test_ap: $final_detector_test_ap
combined_test_ap: $final_combined_test_ap
detector_test_simplified_ap: $final_detector_test_simplified_ap
combined_test_simplified_ap: $final_combined_test_simplified_ap
package_fpath: $final_package_fpath
timestamp_utc: $(date -u '+%Y-%m-%dT%H:%M:%SZ')
EOF
    mv "$tmp_fpath" "$manifest_fpath"
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"

REPO_DPATH="$(canonical_existing_path /home/joncrall/code/shitspotter)"
DATA_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_dvc)"
EXPT_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_expt_dvc)"

# New larger splits written 2026-04-13
TRAIN_FPATH="$DATA_DPATH/train_imgs10671_b277c63d.kwcoco.zip"
VALI_FPATH="$DATA_DPATH/vali_imgs1258_577e331c.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_d39956b1.kwcoco.zip"

SHITSPOTTER_SAM2_REPO_DPATH="${SHITSPOTTER_SAM2_REPO_DPATH:-$REPO_DPATH/tpl/segment-anything-2}"
SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH="${SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH:-$REPO_DPATH/tpl/Open-GroundingDino}"

SAM2_INIT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"
GDINO_PRETRAIN_CKPT="${GDINO_PRETRAIN_CKPT:-$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH/groundingdino_swint_ogc.pth}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"

V9_ROOT="$EXPT_DPATH/foundation_detseg_v3/v9"
DETECTOR_WORKDIR="$V9_ROOT/train_detector_opengroundingdino"
SAM2_WORKDIR="$V9_ROOT/train_segmenter_sam2_1_hiera_base_plus"
DETECTOR_PREP_DPATH="$V9_ROOT/detector_prepared"
CHECKPOINT_SELECT_DPATH="$V9_ROOT/checkpoint_select"
CHECKPOINT_PACKAGE_DPATH="$CHECKPOINT_SELECT_DPATH/packages"
CHECKPOINT_EVAL_DPATH="$CHECKPOINT_SELECT_DPATH/evals"
SUMMARY_FPATH="$CHECKPOINT_SELECT_DPATH/summary.tsv"
SELECTED_MANIFEST_FPATH="$V9_ROOT/selected_detector_checkpoint.yaml"

PACKAGE_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3/packages"
FINAL_TUNED_PACKAGE_FPATH="$PACKAGE_DPATH/v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml"

# Preprocessing knobs must be set before paths that reference them
RESIZE_MAX_DIM="${RESIZE_MAX_DIM:-640}"
RESIZE_OUTPUT_EXT="${RESIZE_OUTPUT_EXT:-.jpg}"
FORCE_RESIZE_PREPROCESS="${FORCE_RESIZE_PREPROCESS:-False}"
SIMPLIFY_MINIMUM_INSTANCES="${SIMPLIFY_MINIMUM_INSTANCES:-100}"
FORCE_SIMPLIFY_PREPROCESS="${FORCE_SIMPLIFY_PREPROCESS:-False}"

# Preprocessed kwcoco paths (resize then simplify before COCO export)
PREPROC_TRAIN_RESIZED_FPATH="$DETECTOR_PREP_DPATH/train_maxdim${RESIZE_MAX_DIM}.kwcoco.zip"
PREPROC_VALI_RESIZED_FPATH="$DETECTOR_PREP_DPATH/vali_maxdim${RESIZE_MAX_DIM}.kwcoco.zip"
PREPROC_TRAIN_SIMPLIFIED_FPATH="$DETECTOR_PREP_DPATH/train_maxdim${RESIZE_MAX_DIM}.simplified.kwcoco.zip"
PREPROC_VALI_SIMPLIFIED_FPATH="$DETECTOR_PREP_DPATH/vali_maxdim${RESIZE_MAX_DIM}.simplified.kwcoco.zip"
# Simplified test GT: annotations merged with same parameters as training preprocessing.
# Images are not resized since existing predictions were made on original-resolution test images.
PREPROC_TEST_SIMPLIFIED_FPATH="$DETECTOR_PREP_DPATH/test.simplified.kwcoco.zip"

# Prepared data paths (exported from preprocessed kwcoco)
TRAIN_MSCOCO_FPATH="$DETECTOR_PREP_DPATH/train.mscoco.json"
VALI_MSCOCO_FPATH="$DETECTOR_PREP_DPATH/vali.mscoco.json"
TRAIN_ODVG_FPATH="$DETECTOR_PREP_DPATH/train.odvg.jsonl"
LABEL_MAP_FPATH="$DETECTOR_PREP_DPATH/label_map.json"
DATASETS_JSON_FPATH="$DETECTOR_PREP_DPATH/datasets.json"
GDINO_CFG_FPATH="$DETECTOR_PREP_DPATH/shitspotter_cfg_odvg.py"

# ---------------------------------------------------------------------------
# Tuneable knobs
# ---------------------------------------------------------------------------
DETECTOR_SCORE_THRESH="${DETECTOR_SCORE_THRESH:-0.2}"
DETECTOR_NMS_THRESH="${DETECTOR_NMS_THRESH:-0.5}"
FORCE_DETECTOR_PREP="${FORCE_DETECTOR_PREP:-False}"
FORCE_DETECTOR_RERUN="${FORCE_DETECTOR_RERUN:-False}"
FORCE_SEGMENTER_RERUN="${FORCE_SEGMENTER_RERUN:-False}"
FORCE_CANDIDATE_EVALS="${FORCE_CANDIDATE_EVALS:-False}"
FORCE_FINAL_EVALS="${FORCE_FINAL_EVALS:-False}"
# Re-evaluate existing predictions against simplified test GT (no new inference)
FORCE_SIMPLIFIED_REEVAL="${FORCE_SIMPLIFIED_REEVAL:-False}"
# Rasterize polygon predictions to salient heatmaps and run segmentation eval
FORCE_SSEG_EVAL="${FORCE_SSEG_EVAL:-False}"

# OpenGroundingDINO training - baseline hyperparameters validated in small-data benchmark
GDINO_GPU_NUM="${GDINO_GPU_NUM:-1}"
GDINO_BATCH_SIZE="${GDINO_BATCH_SIZE:-4}"
GDINO_LR="${GDINO_LR:-0.0001}"
GDINO_LR_BACKBONE="${GDINO_LR_BACKBONE:-1e-05}"
GDINO_EPOCHS="${GDINO_EPOCHS:-15}"
CLASSES_TEXT="${CLASSES_TEXT:-[poop]}"

# SAM2 training - same as v5/v8
SAM2_NUM_GPUS="${SAM2_NUM_GPUS:-1}"
SAM2_TRAIN_BATCH_SIZE="${SAM2_TRAIN_BATCH_SIZE:-1}"
SAM2_NUM_TRAIN_WORKERS="${SAM2_NUM_TRAIN_WORKERS:-8}"
SAM2_NUM_EPOCHS="${SAM2_NUM_EPOCHS:-20}"
SAM2_BASE_LR="${SAM2_BASE_LR:-0.000005}"
SAM2_VISION_LR="${SAM2_VISION_LR:-0.000003}"
SAM2_MAX_NUM_OBJECTS="${SAM2_MAX_NUM_OBJECTS:-8}"
SAM2_MULTIPLIER="${SAM2_MULTIPLIER:-1}"
SAM2_CHECKPOINT_SAVE_FREQ="${SAM2_CHECKPOINT_SAVE_FREQ:-1}"

# Checkpoint candidates: 0-indexed epochs 0..N-1 plus 'last' fallback
_GDINO_FINAL_CKPT_IDX=$(( GDINO_EPOCHS - 1 ))
_GDINO_ALL_CANDIDATES=()
for _ep in $(seq 0 "$_GDINO_FINAL_CKPT_IDX"); do
    _GDINO_ALL_CANDIDATES+=( "$(printf 'checkpoint%04d' "$_ep")" )
done
CANDIDATES=( "${_GDINO_ALL_CANDIDATES[@]}" )

for required in \
    "$REPO_DPATH" \
    "$DATA_DPATH" \
    "$EXPT_DPATH" \
    "$TRAIN_FPATH" \
    "$VALI_FPATH" \
    "$TEST_FPATH" \
    "$SHITSPOTTER_SAM2_REPO_DPATH" \
    "$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH" \
    "$SAM2_INIT_CKPT" \
    "$GDINO_PRETRAIN_CKPT"; do
    require_path "$required"
done

mkdir -p "$DETECTOR_PREP_DPATH" "$CHECKPOINT_PACKAGE_DPATH" "$CHECKPOINT_EVAL_DPATH"

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_SAM2_REPO_DPATH
export SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH
export PYTHONPATH="$REPO_DPATH${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

echo "v9 foundation_detseg_v3 OpenGroundingDINO+SAM2 on new 10671-image dataset"
printf '  %-32s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-32s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-32s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-32s %s\n' "TRAIN_FPATH" "$TRAIN_FPATH"
printf '  %-32s %s\n' "VALI_FPATH" "$VALI_FPATH"
printf '  %-32s %s\n' "TEST_FPATH" "$TEST_FPATH"
printf '  %-32s %s\n' "V9_ROOT" "$V9_ROOT"
printf '  %-32s %s\n' "DETECTOR_WORKDIR" "$DETECTOR_WORKDIR"
printf '  %-32s %s\n' "SAM2_WORKDIR" "$SAM2_WORKDIR"
printf '  %-32s %s\n' "GDINO_GPU_NUM" "$GDINO_GPU_NUM"
printf '  %-32s %s\n' "GDINO_BATCH_SIZE" "$GDINO_BATCH_SIZE"
printf '  %-32s %s\n' "GDINO_LR" "$GDINO_LR"
printf '  %-32s %s\n' "GDINO_LR_BACKBONE" "$GDINO_LR_BACKBONE"
printf '  %-32s %s\n' "GDINO_EPOCHS" "$GDINO_EPOCHS"
printf '  %-32s %s\n' "SAM2_NUM_GPUS" "$SAM2_NUM_GPUS"
printf '  %-32s %s\n' "SAM2_NUM_EPOCHS" "$SAM2_NUM_EPOCHS"
printf '  %-32s %s\n' "RESIZE_MAX_DIM" "$RESIZE_MAX_DIM"
printf '  %-32s %s\n' "SIMPLIFY_MINIMUM_INSTANCES" "$SIMPLIFY_MINIMUM_INSTANCES"

# ---------------------------------------------------------------------------
echo
echo "=== Resize preprocessing ==="
# ---------------------------------------------------------------------------
if is_truthy "$FORCE_RESIZE_PREPROCESS" || [ ! -f "$PREPROC_TRAIN_RESIZED_FPATH" ]; then
    "$PYTHON_BIN" -m shitspotter.cli.resize_kwcoco \
        --src "$TRAIN_FPATH" \
        --dst "$PREPROC_TRAIN_RESIZED_FPATH" \
        --max_dim "$RESIZE_MAX_DIM" \
        --asset_dname "train_assets_maxdim${RESIZE_MAX_DIM}" \
        --output_ext "$RESIZE_OUTPUT_EXT"
else
    echo "  Reusing resized train: $PREPROC_TRAIN_RESIZED_FPATH"
fi

if is_truthy "$FORCE_RESIZE_PREPROCESS" || [ ! -f "$PREPROC_VALI_RESIZED_FPATH" ]; then
    "$PYTHON_BIN" -m shitspotter.cli.resize_kwcoco \
        --src "$VALI_FPATH" \
        --dst "$PREPROC_VALI_RESIZED_FPATH" \
        --max_dim "$RESIZE_MAX_DIM" \
        --asset_dname "vali_assets_maxdim${RESIZE_MAX_DIM}" \
        --output_ext "$RESIZE_OUTPUT_EXT"
else
    echo "  Reusing resized vali: $PREPROC_VALI_RESIZED_FPATH"
fi

# ---------------------------------------------------------------------------
echo
echo "=== Simplify preprocessing ==="
# ---------------------------------------------------------------------------
if is_truthy "$FORCE_SIMPLIFY_PREPROCESS" || [ ! -f "$PREPROC_TRAIN_SIMPLIFIED_FPATH" ]; then
    "$PYTHON_BIN" -m shitspotter.cli.simplify_kwcoco \
        --src "$PREPROC_TRAIN_RESIZED_FPATH" \
        --dst "$PREPROC_TRAIN_SIMPLIFIED_FPATH" \
        --minimum_instances "$SIMPLIFY_MINIMUM_INSTANCES"
else
    echo "  Reusing simplified train: $PREPROC_TRAIN_SIMPLIFIED_FPATH"
fi

if is_truthy "$FORCE_SIMPLIFY_PREPROCESS" || [ ! -f "$PREPROC_VALI_SIMPLIFIED_FPATH" ]; then
    "$PYTHON_BIN" -m shitspotter.cli.simplify_kwcoco \
        --src "$PREPROC_VALI_RESIZED_FPATH" \
        --dst "$PREPROC_VALI_SIMPLIFIED_FPATH" \
        --minimum_instances "$SIMPLIFY_MINIMUM_INSTANCES"
else
    echo "  Reusing simplified vali: $PREPROC_VALI_SIMPLIFIED_FPATH"
fi

# ---------------------------------------------------------------------------
echo
echo "=== Prepare OpenGroundingDINO training data ==="
# ---------------------------------------------------------------------------
if is_truthy "$FORCE_DETECTOR_PREP" || [ ! -f "$TRAIN_ODVG_FPATH" ] || [ ! -f "$VALI_MSCOCO_FPATH" ]; then

    echo "  Exporting preprocessed kwcoco → MSCOCO for training split..."
    "$PYTHON_BIN" -c "
from shitspotter.algo_foundation_v3.coco_adapter import _build_coco_export
_build_coco_export(
    src='$PREPROC_TRAIN_SIMPLIFIED_FPATH',
    dst='$TRAIN_MSCOCO_FPATH',
    category_name='poop',
    include_segmentations=False,
    category_id=0,
)
print('Wrote $TRAIN_MSCOCO_FPATH')
"

    echo "  Exporting preprocessed kwcoco → MSCOCO for validation split..."
    "$PYTHON_BIN" -c "
from shitspotter.algo_foundation_v3.coco_adapter import _build_coco_export
_build_coco_export(
    src='$PREPROC_VALI_SIMPLIFIED_FPATH',
    dst='$VALI_MSCOCO_FPATH',
    category_name='poop',
    include_segmentations=False,
    category_id=0,
)
print('Wrote $VALI_MSCOCO_FPATH')
"

    echo "  Converting train MSCOCO → ODVG..."
    (
        cd "$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH"
        "$PYTHON_BIN" tools/coco2odvg.py \
            --input "$TRAIN_MSCOCO_FPATH" \
            --output "$TRAIN_ODVG_FPATH" \
            --idmap=False
    )

    echo "  Writing label map..."
    "$PYTHON_BIN" -c "
import json
from pathlib import Path
data = json.loads(Path('$TRAIN_MSCOCO_FPATH').read_text())
label_map = {str(cat['id']): cat['name'] for cat in data.get('categories', [])}
Path('$LABEL_MAP_FPATH').write_text(json.dumps(label_map, indent=2))
print('Wrote $LABEL_MAP_FPATH')
"

    echo "  Writing datasets JSON..."
    cat > "$DATASETS_JSON_FPATH" <<DATASETS_EOF
{
  "train": [
    {
      "root": "$DETECTOR_PREP_DPATH",
      "anno": "$TRAIN_ODVG_FPATH",
      "label_map": "$LABEL_MAP_FPATH",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "$DETECTOR_PREP_DPATH",
      "anno": "$VALI_MSCOCO_FPATH",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
DATASETS_EOF

    echo "  Writing OpenGroundingDINO config..."
    cp "$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH/config/cfg_odvg.py" "$GDINO_CFG_FPATH"
    sed -i 's|use_coco_eval = True|use_coco_eval = False|g' "$GDINO_CFG_FPATH"
    cat >> "$GDINO_CFG_FPATH" <<CFG_EOF

# --- ShitSpotter v9 overrides ---
label_list = ['poop']
batch_size = $GDINO_BATCH_SIZE
lr = $GDINO_LR
lr_backbone = $GDINO_LR_BACKBONE
epochs = $GDINO_EPOCHS
CFG_EOF

else
    echo "  Reusing existing prepared detector data in: $DETECTOR_PREP_DPATH"
fi

# ---------------------------------------------------------------------------
echo
echo "=== Train OpenGroundingDINO detector ==="
# ---------------------------------------------------------------------------
_GDINO_FINAL_CKPT="$(printf 'checkpoint%04d.pth' "$_GDINO_FINAL_CKPT_IDX")"
DETECTOR_SENTINEL_FPATH="$DETECTOR_WORKDIR/$_GDINO_FINAL_CKPT"

if [ -f "$DETECTOR_SENTINEL_FPATH" ] && ! is_truthy "$FORCE_DETECTOR_RERUN"; then
    echo "Reusing existing detector training in: $DETECTOR_WORKDIR"
else
    (
        cd "$SHITSPOTTER_OPENGROUNDINGDINO_REPO_DPATH"
        export GPU_NUM="$GDINO_GPU_NUM"
        export CFG="$GDINO_CFG_FPATH"
        export DATASETS="$DATASETS_JSON_FPATH"
        export OUTPUT_DIR="$DETECTOR_WORKDIR"
        export PRETRAIN_MODEL_PATH="$GDINO_PRETRAIN_CKPT"
        export TEXT_ENCODER_TYPE
        bash train_dist.sh "$GPU_NUM" "$CFG" "$DATASETS" "$OUTPUT_DIR"
    )
fi

require_path "$DETECTOR_WORKDIR"

# ---------------------------------------------------------------------------
echo
echo "=== Train SAM2 segmenter on new data ==="
# ---------------------------------------------------------------------------
SAM2_TRAINED_CKPT="$(ensure_segmenter_checkpoint || true)"
if [ -n "${SAM2_TRAINED_CKPT:-}" ] && ! is_truthy "$FORCE_SEGMENTER_RERUN"; then
    echo "Reusing existing segmenter checkpoint: $SAM2_TRAINED_CKPT"
else
    export TRAIN_FPATH
    export VALI_FPATH
    export SAM2_WORKDIR
    export SAM2_VARIANT="sam2.1_hiera_base_plus"
    export SAM2_INIT_CKPT
    export SAM2_TRAIN_BATCH_SIZE
    export SAM2_NUM_TRAIN_WORKERS
    export SAM2_NUM_EPOCHS
    export SAM2_NUM_GPUS
    export SAM2_BASE_LR
    export SAM2_VISION_LR
    export SAM2_MAX_NUM_OBJECTS
    export SAM2_MULTIPLIER
    export SAM2_CHECKPOINT_SAVE_FREQ
    bash "$REPO_DPATH/experiments/foundation_detseg_v3/train_sam2_segmenter.sh"
fi

SAM2_TRAINED_CKPT="$(ensure_segmenter_checkpoint)" || {
    echo "Expected segmenter checkpoint missing in: $SAM2_WORKDIR/checkpoints" >&2
    exit 1
}
printf '  %-32s %s\n' "SAM2_TRAINED_CKPT" "$SAM2_TRAINED_CKPT"

# ---------------------------------------------------------------------------
echo
echo "=== Sweep checkpoint shortlist on validation ==="
# ---------------------------------------------------------------------------
printf 'candidate_id\tckpt_fpath\tdetector_vali_ap\tcombined_vali_ap\tselection_note\n' > "$SUMMARY_FPATH"
for candidate_id in "${CANDIDATES[@]}"; do
    ckpt_fpath="$DETECTOR_WORKDIR/${candidate_id}.pth"
    if [ ! -f "$ckpt_fpath" ]; then
        echo "Skipping missing candidate: $candidate_id"
        continue
    fi
    candidate_package_fpath="$CHECKPOINT_PACKAGE_DPATH/${candidate_id}.yaml"
    build_package "$candidate_package_fpath" "$ckpt_fpath" "$GDINO_CFG_FPATH" "$SAM2_TRAINED_CKPT" "v9_${candidate_id}"

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
echo "Top v9 checkpoint candidates:"
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
build_package "$FINAL_TUNED_PACKAGE_FPATH" "$SELECTED_CKPT_FPATH" "$GDINO_CFG_FPATH" "$SAM2_TRAINED_CKPT" "v9_opengroundingdino_sam2_1_hiera_base_plus_tuned"

SELECTED_FINAL_ROOT="$V9_ROOT/final_eval/${SELECTED_CANDIDATE_ID}"
FINAL_DETECTOR_TEST_DPATH="$SELECTED_FINAL_ROOT/detector_test"
FINAL_COMBINED_TEST_DPATH="$SELECTED_FINAL_ROOT/combined_raw_test"
FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH="$SELECTED_FINAL_ROOT/detector_test_simplified"
FINAL_COMBINED_TEST_SIMPLIFIED_DPATH="$SELECTED_FINAL_ROOT/combined_raw_test_simplified"
# Segmentation evaluation paths (heatmap kwcoco + sseg metrics)
FINAL_COMBINED_HEATMAP_FPATH="$SELECTED_FINAL_ROOT/combined_raw_test/pred_salient.kwcoco.zip"
FINAL_COMBINED_SIMPLIFIED_HEATMAP_FPATH="$SELECTED_FINAL_ROOT/combined_raw_test_simplified/pred_salient.kwcoco.zip"

echo
echo "=== Evaluate selected detector on test ==="
if is_truthy "$FORCE_FINAL_EVALS" || ! have_metrics "$FINAL_DETECTOR_TEST_DPATH"; then
    evaluate_detector_split "$TEST_FPATH" "$FINAL_TUNED_PACKAGE_FPATH" "$FINAL_DETECTOR_TEST_DPATH" "$DETECTOR_SCORE_THRESH" "$DETECTOR_NMS_THRESH"
else
    echo "Reusing final detector test metrics: $FINAL_DETECTOR_TEST_DPATH/eval/detect_metrics.json"
fi

echo
echo "=== Evaluate selected detector + tuned SAM2 on test ==="
if is_truthy "$FORCE_FINAL_EVALS" || ! have_metrics "$FINAL_COMBINED_TEST_DPATH"; then
    evaluate_combined_raw_split "$TEST_FPATH" "$FINAL_TUNED_PACKAGE_FPATH" "$FINAL_COMBINED_TEST_DPATH"
else
    echo "Reusing final combined test metrics: $FINAL_COMBINED_TEST_DPATH/eval/detect_metrics.json"
fi

FINAL_DETECTOR_TEST_AP="$(read_ap "$FINAL_DETECTOR_TEST_DPATH/eval/detect_metrics.json")"
FINAL_COMBINED_TEST_AP="$(read_ap "$FINAL_COMBINED_TEST_DPATH/eval/detect_metrics.json")"

# ---------------------------------------------------------------------------
# Simplified test GT re-evaluation
#
# The raw test set has ~16 poop annotations per image vs ~2 per image in the
# training and validation splits. This density mismatch is due to inconsistent
# per-instance vs per-cluster annotation conventions across the dataset.
# Applying the same simplify_kwcoco merge step to the test GT puts all splits
# on the same semantic footing (detect the cluster, not individual instances)
# and is the canonical metric for this experiment family going forward.
#
# Importantly, this is a pure GT re-evaluation: existing prediction files are
# reused as-is. No new inference is run.
# ---------------------------------------------------------------------------

echo
echo "=== Create simplified test GT ==="
if is_truthy "$FORCE_SIMPLIFIED_REEVAL" || [ ! -f "$PREPROC_TEST_SIMPLIFIED_FPATH" ]; then
    "$PYTHON_BIN" -m shitspotter.cli.simplify_kwcoco \
        --src "$TEST_FPATH" \
        --dst "$PREPROC_TEST_SIMPLIFIED_FPATH" \
        --minimum_instances "$SIMPLIFY_MINIMUM_INSTANCES"
else
    echo "  Reusing simplified test GT: $PREPROC_TEST_SIMPLIFIED_FPATH"
fi

echo
echo "=== Re-evaluate selected detector on simplified test GT ==="
if is_truthy "$FORCE_SIMPLIFIED_REEVAL" || ! have_metrics "$FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH"; then
    mkdir -p "$FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH/eval"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$PREPROC_TEST_SIMPLIFIED_FPATH" \
        --pred_dataset "$FINAL_DETECTOR_TEST_DPATH/pred_boxes.kwcoco.zip" \
        --out_dpath "$FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH/eval" \
        --out_fpath "$FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH/eval/detect_metrics.json" \
        --confusion_fpath "$FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
else
    echo "  Reusing simplified test detector metrics: $FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH/eval/detect_metrics.json"
fi

echo
echo "=== Re-evaluate combined on simplified test GT ==="
if is_truthy "$FORCE_SIMPLIFIED_REEVAL" || ! have_metrics "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH"; then
    mkdir -p "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/eval"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$PREPROC_TEST_SIMPLIFIED_FPATH" \
        --pred_dataset "$FINAL_COMBINED_TEST_DPATH/pred.kwcoco.zip" \
        --out_dpath "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/eval" \
        --out_fpath "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/eval/detect_metrics.json" \
        --confusion_fpath "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
else
    echo "  Reusing simplified test combined metrics: $FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/eval/detect_metrics.json"
fi

FINAL_DETECTOR_TEST_SIMPLIFIED_AP="$(read_ap "$FINAL_DETECTOR_TEST_SIMPLIFIED_DPATH/eval/detect_metrics.json")"
FINAL_COMBINED_TEST_SIMPLIFIED_AP="$(read_ap "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/eval/detect_metrics.json")"

# ---------------------------------------------------------------------------
# Segmentation-level evaluation
#
# kwcoco.metrics.segmentation_metrics measures pixel-level AP (salient_ap)
# and max-F1 against a binary foreground/background truth mask.  It requires
# a per-pixel probability channel named 'salient' on each prediction image.
#
# Our SAM2 combined predictions store results as polygon annotations with a
# float 'score' field.  cli_rasterize_pred_heatmap converts those into uint8
# PNG assets (pixel value = max annotation score * 255) and writes a new
# kwcoco that references them as an auxiliary 'salient' channel.
#
# We evaluate against two GT variants:
#   1. Raw test GT (dense per-instance annotations)
#   2. Simplified test GT (merged cluster-level annotations, canonical)
#
# For each eval a reproduce_sseg_eval.sh script is written alongside the
# results so the eval can be re-run or inspected independently.
# ---------------------------------------------------------------------------

echo
echo "=== Rasterize combined predictions to salient heatmaps ==="
if is_truthy "$FORCE_SSEG_EVAL" || [ ! -f "$FINAL_COMBINED_HEATMAP_FPATH" ]; then
    rasterize_pred_heatmap \
        "$FINAL_COMBINED_TEST_DPATH/pred.kwcoco.zip" \
        "$FINAL_COMBINED_HEATMAP_FPATH"
else
    echo "  Reusing heatmap kwcoco: $FINAL_COMBINED_HEATMAP_FPATH"
fi

if is_truthy "$FORCE_SSEG_EVAL" || [ ! -f "$FINAL_COMBINED_SIMPLIFIED_HEATMAP_FPATH" ]; then
    rasterize_pred_heatmap \
        "$FINAL_COMBINED_TEST_DPATH/pred.kwcoco.zip" \
        "$FINAL_COMBINED_SIMPLIFIED_HEATMAP_FPATH"
else
    echo "  Reusing simplified heatmap kwcoco: $FINAL_COMBINED_SIMPLIFIED_HEATMAP_FPATH"
fi

echo
echo "=== Segmentation eval: combined vs raw test GT ==="
if is_truthy "$FORCE_SSEG_EVAL" || ! have_sseg_metrics "$FINAL_COMBINED_TEST_DPATH"; then
    evaluate_segmentation_split \
        "$TEST_FPATH" \
        "$FINAL_COMBINED_HEATMAP_FPATH" \
        "$FINAL_COMBINED_TEST_DPATH"
else
    echo "  Reusing sseg metrics: $FINAL_COMBINED_TEST_DPATH/sseg_eval/summary_metrics.json"
fi

echo
echo "=== Segmentation eval: combined vs simplified test GT ==="
if is_truthy "$FORCE_SSEG_EVAL" || ! have_sseg_metrics "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH"; then
    evaluate_segmentation_split \
        "$PREPROC_TEST_SIMPLIFIED_FPATH" \
        "$FINAL_COMBINED_SIMPLIFIED_HEATMAP_FPATH" \
        "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH"
else
    echo "  Reusing sseg metrics: $FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/sseg_eval/summary_metrics.json"
fi

write_selected_manifest \
    "$SELECTED_MANIFEST_FPATH" \
    "$SELECTED_CANDIDATE_ID" \
    "$SELECTED_CKPT_FPATH" \
    "$SELECTED_DETECTOR_VALI_AP" \
    "$SELECTED_COMBINED_VALI_AP" \
    "$FINAL_DETECTOR_TEST_AP" \
    "$FINAL_COMBINED_TEST_AP" \
    "$FINAL_TUNED_PACKAGE_FPATH" \
    "$FINAL_DETECTOR_TEST_SIMPLIFIED_AP" \
    "$FINAL_COMBINED_TEST_SIMPLIFIED_AP"

echo
echo "=== v9 summary ==="
printf '  %-36s %s\n' "selected_candidate_id" "$SELECTED_CANDIDATE_ID"
printf '  %-36s ap=%s\n' "detector_only_vali" "$SELECTED_DETECTOR_VALI_AP"
printf '  %-36s ap=%s\n' "combined_tuned_raw_vali" "$SELECTED_COMBINED_VALI_AP"
printf '  %-36s ap=%s\n' "detector_only_test (dense GT)" "$FINAL_DETECTOR_TEST_AP"
printf '  %-36s ap=%s\n' "combined_tuned_raw_test (dense GT)" "$FINAL_COMBINED_TEST_AP"
printf '  %-36s ap=%s\n' "detector_only_test (simplified GT)" "$FINAL_DETECTOR_TEST_SIMPLIFIED_AP"
printf '  %-36s ap=%s\n' "combined_tuned_raw_test (simplified GT)" "$FINAL_COMBINED_TEST_SIMPLIFIED_AP"

echo
echo "v9 run completed"
printf '  %-32s %s\n' "sseg_eval (raw GT)" "$FINAL_COMBINED_TEST_DPATH/sseg_eval/summary_metrics.json"
printf '  %-32s %s\n' "sseg_eval (simplified GT)" "$FINAL_COMBINED_TEST_SIMPLIFIED_DPATH/sseg_eval/summary_metrics.json"
printf '  %-32s %s\n' "SELECTED_DETECTOR_CKPT" "$SELECTED_CKPT_FPATH"
printf '  %-32s %s\n' "DETECTOR_CONFIG_FPATH" "$GDINO_CFG_FPATH"
printf '  %-32s %s\n' "TUNED_SAM2_CKPT" "$SAM2_TRAINED_CKPT"
printf '  %-32s %s\n' "FINAL_TUNED_PACKAGE_FPATH" "$FINAL_TUNED_PACKAGE_FPATH"
printf '  %-32s %s\n' "SUMMARY_FPATH" "$SUMMARY_FPATH"
printf '  %-32s %s\n' "SELECTED_MANIFEST_FPATH" "$SELECTED_MANIFEST_FPATH"
