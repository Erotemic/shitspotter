#!/bin/bash
set -euo pipefail

# Hard-coded validation-centric debug run for segmenter / geometry issues.
# This keeps the trained v1 detector fixed and compares tuned vs zero-shot
# SAM2 under raw and default postprocess settings.

REPO_DPATH="/home/joncrall/code/shitspotter"
DATA_DPATH="/home/joncrall/data/dvc-repos/shitspotter_dvc"
EXPT_DPATH="/home/joncrall/data/dvc-repos/shitspotter_expt_dvc"

SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"

VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip"

V1_ROOT="$EXPT_DPATH/foundation_detseg_v3/v1"
V2_ROOT="$EXPT_DPATH/foundation_detseg_v3/v2"
PACKAGE_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3/packages"

DEIMV2_TRAINED_CKPT="$V1_ROOT/train_detector_deimv2_m/best_stg2.pth"
SAM2_TUNED_CKPT="$V1_ROOT/train_segmenter_sam2_1_hiera_base_plus/checkpoints/checkpoint.pt"
SAM2_ZERO_SHOT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"

TUNED_PACKAGE_FPATH="$PACKAGE_DPATH/v2_deimv2_m_sam2_1_hiera_base_plus_tuned.yaml"
ZEROSHOT_PACKAGE_FPATH="$PACKAGE_DPATH/v2_deimv2_m_sam2_1_hiera_base_plus_zeroshot.yaml"

SIZE_DIAG_DPATH="$V2_ROOT/dataset_geometry"
BOX_VALI_DPATH="$V2_ROOT/eval_detector_only/vali"
BOX_TEST_DPATH="$V2_ROOT/eval_detector_only/test"

GTBOX_TUNED_RAW_VALI_DPATH="$V2_ROOT/eval_gtbox_segmenter/vali/tuned_raw"
GTBOX_TUNED_DEFAULT_VALI_DPATH="$V2_ROOT/eval_gtbox_segmenter/vali/tuned_default"
GTBOX_ZEROSHOT_RAW_VALI_DPATH="$V2_ROOT/eval_gtbox_segmenter/vali/zeroshot_raw"
GTBOX_ZEROSHOT_DEFAULT_VALI_DPATH="$V2_ROOT/eval_gtbox_segmenter/vali/zeroshot_default"

COMBINED_TUNED_RAW_VALI_DPATH="$V2_ROOT/eval_detector_segmenter/vali/tuned_raw"
COMBINED_TUNED_DEFAULT_VALI_DPATH="$V2_ROOT/eval_detector_segmenter/vali/tuned_default"
COMBINED_ZEROSHOT_RAW_VALI_DPATH="$V2_ROOT/eval_detector_segmenter/vali/zeroshot_raw"
COMBINED_ZEROSHOT_DEFAULT_VALI_DPATH="$V2_ROOT/eval_detector_segmenter/vali/zeroshot_default"

fail_if_exists() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Refusing to overwrite existing path: $path" >&2
        echo "Use a new script version label (v3, v4, ...) or remove the old path manually." >&2
        exit 1
    fi
}

require_path() {
    local path="$1"
    if [ ! -e "$path" ]; then
        echo "Required path does not exist: $path" >&2
        exit 1
    fi
}

write_dataset_size_report() {
    local dataset_fpath="$1"
    local report_fpath="$2"
    python - "$dataset_fpath" "$report_fpath" <<'PY'
import json
import sys
from pathlib import Path

import kwcoco
import kwimage

dataset_fpath = sys.argv[1]
report_fpath = Path(sys.argv[2])
dset = kwcoco.CocoDataset(dataset_fpath)

exact_match = 0
swapped_match = 0
other_mismatch = 0
load_errors = []
mismatch_examples = []

for img in dset.images().objs:
    gid = img['id']
    meta_w = int(img.get('width', 0))
    meta_h = int(img.get('height', 0))
    try:
        actual_shape = kwimage.load_image_shape(dset.get_image_fpath(gid))
        actual_h, actual_w = int(actual_shape[0]), int(actual_shape[1])
    except Exception as ex:
        load_errors.append({
            'gid': gid,
            'name': img.get('name'),
            'file_name': img.get('file_name'),
            'error': repr(ex),
        })
        continue

    if (meta_w, meta_h) == (actual_w, actual_h):
        exact_match += 1
    elif (meta_w, meta_h) == (actual_h, actual_w):
        swapped_match += 1
        if len(mismatch_examples) < 20:
            mismatch_examples.append({
                'kind': 'swapped',
                'gid': gid,
                'name': img.get('name'),
                'meta_wh': [meta_w, meta_h],
                'actual_wh': [actual_w, actual_h],
            })
    else:
        other_mismatch += 1
        if len(mismatch_examples) < 20:
            mismatch_examples.append({
                'kind': 'other_mismatch',
                'gid': gid,
                'name': img.get('name'),
                'meta_wh': [meta_w, meta_h],
                'actual_wh': [actual_w, actual_h],
            })

report = {
    'dataset_fpath': dataset_fpath,
    'summary': {
        'num_images': len(dset.images()),
        'exact_match': exact_match,
        'swapped_match': swapped_match,
        'other_mismatch': other_mismatch,
        'load_errors': len(load_errors),
    },
    'mismatch_examples': mismatch_examples,
    'load_errors': load_errors[:20],
}
report_fpath.parent.mkdir(parents=True, exist_ok=True)
report_fpath.write_text(json.dumps(report, indent=2))
print(json.dumps(report['summary'], indent=2))
PY
}

write_prediction_geometry_report() {
    local true_fpath="$1"
    local pred_fpath="$2"
    local report_fpath="$3"
    python - "$true_fpath" "$pred_fpath" "$report_fpath" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

import kwcoco

true_fpath, pred_fpath, report_fpath = sys.argv[1:4]
true_dset = kwcoco.CocoDataset(true_fpath)
pred_dset = kwcoco.CocoDataset(pred_fpath)


def coco_to_ltrb(bbox):
    x, y, w, h = map(float, bbox)
    return (x, y, x + w, y + h)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(box1, box2):
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = box_area((ix1, iy1, ix2, iy2))
    if inter <= 0:
        return 0.0
    union = box_area(box1) + box_area(box2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def summarize(values):
    if not values:
        return None
    return {
        'count': len(values),
        'mean': float(statistics.fmean(values)),
        'median': float(statistics.median(values)),
        'min': float(min(values)),
        'max': float(max(values)),
    }


pred_annots = pred_dset.annots().objs
true_annots = true_dset.annots().objs
out_of_bounds = []
prompt_ious = []
prompt_area_ratios = []
prompt_center_shift = []
source_gt_ious = []
low_prompt_examples = []
count_deltas = []

for img in pred_dset.images().objs:
    gid = img['id']
    true_count = len(true_dset.annots(gid=gid))
    pred_count = len(pred_dset.annots(gid=gid))
    count_deltas.append(pred_count - true_count)

    width = float(img.get('width', 0))
    height = float(img.get('height', 0))
    for ann in pred_dset.annots(gid=gid).objs:
        bbox = ann.get('bbox')
        if bbox is None:
            continue
        box = coco_to_ltrb(bbox)
        if box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height:
            if len(out_of_bounds) < 20:
                out_of_bounds.append({
                    'gid': gid,
                    'name': img.get('name'),
                    'bbox': bbox,
                    'image_wh': [width, height],
                })

        prompt_bbox = ann.get('prompt_bbox')
        if prompt_bbox is not None:
            prompt_box = coco_to_ltrb(prompt_bbox)
            pred_box = coco_to_ltrb(bbox)
            iou = box_iou(prompt_box, pred_box)
            prompt_ious.append(iou)
            prompt_area = box_area(prompt_box)
            pred_area = box_area(pred_box)
            if prompt_area > 0:
                prompt_area_ratios.append(pred_area / prompt_area)
            pcx, pcy = box_center(prompt_box)
            qcx, qcy = box_center(pred_box)
            diag = math.hypot(width, height) or 1.0
            prompt_center_shift.append(math.hypot(qcx - pcx, qcy - pcy) / diag)
            if iou < 0.1 and len(low_prompt_examples) < 20:
                low_prompt_examples.append({
                    'gid': gid,
                    'name': img.get('name'),
                    'score': ann.get('score'),
                    'prompt_bbox': prompt_bbox,
                    'pred_bbox': bbox,
                    'prompt_iou': iou,
                })

        source_gt_bbox = ann.get('source_gt_bbox')
        if source_gt_bbox is not None:
            source_gt_ious.append(box_iou(coco_to_ltrb(source_gt_bbox), coco_to_ltrb(bbox)))

summary = {
    'num_images': len(pred_dset.images()),
    'num_true_annotations': len(true_annots),
    'num_pred_annotations': len(pred_annots),
    'out_of_bounds_count': len(out_of_bounds),
    'mean_abs_count_delta_per_image': float(statistics.fmean(abs(v) for v in count_deltas)) if count_deltas else 0.0,
    'images_with_pred_lt_truth': sum(1 for v in count_deltas if v < 0),
    'images_with_pred_gt_truth': sum(1 for v in count_deltas if v > 0),
    'prompt_iou': summarize(prompt_ious),
    'prompt_area_ratio': summarize(prompt_area_ratios),
    'prompt_center_shift_norm': summarize(prompt_center_shift),
    'source_gt_iou': summarize(source_gt_ious),
}

report = {
    'true_fpath': true_fpath,
    'pred_fpath': pred_fpath,
    'summary': summary,
    'out_of_bounds_examples': out_of_bounds,
    'low_prompt_iou_examples': low_prompt_examples,
}
report_fpath.parent.mkdir(parents=True, exist_ok=True)
report_fpath.write_text(json.dumps(report, indent=2))
print(json.dumps(summary, indent=2))
PY
}

run_detector_eval() {
    local dataset_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    mkdir -p "$out_dpath/eval"
    python -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        "$dataset_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred_boxes.kwcoco.zip"
    python -m kwcoco eval \
        --true_dataset "$dataset_fpath" \
        --pred_dataset "$out_dpath/pred_boxes.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw=False \
        --iou_thresh=0.5
}

run_gtbox_eval() {
    local dataset_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    local mode="$4"
    mkdir -p "$out_dpath/eval"
    local extra_args=()
    if [ "$mode" = "raw" ]; then
        extra_args=(
            --crop_padding 0
            --polygon_simplify 0
            --min_component_area 0
            --keep_largest_component False
        )
    fi
    python -m shitspotter.algo_foundation_v3.cli_predict_gtboxes \
        "$dataset_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred.kwcoco.zip" \
        "${extra_args[@]}"
    python -m kwcoco eval \
        --true_dataset "$dataset_fpath" \
        --pred_dataset "$out_dpath/pred.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw=False \
        --iou_thresh=0.5
    write_prediction_geometry_report \
        "$dataset_fpath" \
        "$out_dpath/pred.kwcoco.zip" \
        "$out_dpath/geometry_report.json"
}

run_combined_eval() {
    local dataset_fpath="$1"
    local package_fpath="$2"
    local out_dpath="$3"
    local mode="$4"
    mkdir -p "$out_dpath/eval"
    local extra_args=()
    if [ "$mode" = "raw" ]; then
        extra_args=(
            --crop_padding 0
            --polygon_simplify 0
            --min_component_area 0
            --keep_largest_component False
        )
    fi
    python -m shitspotter.algo_foundation_v3.cli_predict \
        "$dataset_fpath" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred.kwcoco.zip" \
        "${extra_args[@]}"
    python -m kwcoco eval \
        --true_dataset "$dataset_fpath" \
        --pred_dataset "$out_dpath/pred.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw=False \
        --iou_thresh=0.5
    write_prediction_geometry_report \
        "$dataset_fpath" \
        "$out_dpath/pred.kwcoco.zip" \
        "$out_dpath/geometry_report.json"
}

build_package() {
    local package_fpath="$1"
    local segmenter_ckpt="$2"
    local metadata_name="$3"
    python -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$DEIMV2_TRAINED_CKPT" \
        --segmenter_checkpoint_fpath "$segmenter_ckpt" \
        --metadata_name "$metadata_name"
}

echo "v2 foundation_detseg_v3 segmenter-debug experiment"
printf '  %-32s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-32s %s\n' "DATA_DPATH" "$DATA_DPATH"
printf '  %-32s %s\n' "EXPT_DPATH" "$EXPT_DPATH"
printf '  %-32s %s\n' "V1_ROOT" "$V1_ROOT"
printf '  %-32s %s\n' "V2_ROOT" "$V2_ROOT"
printf '  %-32s %s\n' "DEIMV2_TRAINED_CKPT" "$DEIMV2_TRAINED_CKPT"
printf '  %-32s %s\n' "SAM2_TUNED_CKPT" "$SAM2_TUNED_CKPT"
printf '  %-32s %s\n' "SAM2_ZERO_SHOT_CKPT" "$SAM2_ZERO_SHOT_CKPT"
printf '  %-32s %s\n' "TUNED_PACKAGE_FPATH" "$TUNED_PACKAGE_FPATH"
printf '  %-32s %s\n' "ZEROSHOT_PACKAGE_FPATH" "$ZEROSHOT_PACKAGE_FPATH"

for required in \
    "$REPO_DPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH" \
    "$SHITSPOTTER_SAM2_REPO_DPATH" \
    "$VALI_FPATH" \
    "$TEST_FPATH" \
    "$DEIMV2_TRAINED_CKPT" \
    "$SAM2_TUNED_CKPT" \
    "$SAM2_ZERO_SHOT_CKPT"; do
    require_path "$required"
done

fail_if_exists "$V2_ROOT"
fail_if_exists "$TUNED_PACKAGE_FPATH"
fail_if_exists "$ZEROSHOT_PACKAGE_FPATH"

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_DEIMV2_REPO_DPATH
export SHITSPOTTER_SAM2_REPO_DPATH
export SHITSPOTTER_MASKDINO_REPO_DPATH

echo
echo "=== Build tuned and zero-shot packages ==="
build_package "$TUNED_PACKAGE_FPATH" "$SAM2_TUNED_CKPT" "v2_deimv2_m_sam2_tuned"
build_package "$ZEROSHOT_PACKAGE_FPATH" "$SAM2_ZERO_SHOT_CKPT" "v2_deimv2_m_sam2_zeroshot"

echo
echo "=== Check dataset image geometry / orientation ==="
write_dataset_size_report "$VALI_FPATH" "$SIZE_DIAG_DPATH/vali_image_size_report.json"
write_dataset_size_report "$TEST_FPATH" "$SIZE_DIAG_DPATH/test_image_size_report.json"

echo
echo "=== Reconfirm detector-only baseline on validation ==="
run_detector_eval "$VALI_FPATH" "$TUNED_PACKAGE_FPATH" "$BOX_VALI_DPATH"

echo
echo "=== Reconfirm detector-only baseline on test ==="
run_detector_eval "$TEST_FPATH" "$TUNED_PACKAGE_FPATH" "$BOX_TEST_DPATH"

echo
echo "=== GT-box SAM2 on validation: tuned raw ==="
run_gtbox_eval "$VALI_FPATH" "$TUNED_PACKAGE_FPATH" "$GTBOX_TUNED_RAW_VALI_DPATH" raw

echo
echo "=== GT-box SAM2 on validation: tuned default ==="
run_gtbox_eval "$VALI_FPATH" "$TUNED_PACKAGE_FPATH" "$GTBOX_TUNED_DEFAULT_VALI_DPATH" default

echo
echo "=== GT-box SAM2 on validation: zero-shot raw ==="
run_gtbox_eval "$VALI_FPATH" "$ZEROSHOT_PACKAGE_FPATH" "$GTBOX_ZEROSHOT_RAW_VALI_DPATH" raw

echo
echo "=== GT-box SAM2 on validation: zero-shot default ==="
run_gtbox_eval "$VALI_FPATH" "$ZEROSHOT_PACKAGE_FPATH" "$GTBOX_ZEROSHOT_DEFAULT_VALI_DPATH" default

echo
echo "=== Combined detector + segmenter on validation: tuned raw ==="
run_combined_eval "$VALI_FPATH" "$TUNED_PACKAGE_FPATH" "$COMBINED_TUNED_RAW_VALI_DPATH" raw

echo
echo "=== Combined detector + segmenter on validation: tuned default ==="
run_combined_eval "$VALI_FPATH" "$TUNED_PACKAGE_FPATH" "$COMBINED_TUNED_DEFAULT_VALI_DPATH" default

echo
echo "=== Combined detector + segmenter on validation: zero-shot raw ==="
run_combined_eval "$VALI_FPATH" "$ZEROSHOT_PACKAGE_FPATH" "$COMBINED_ZEROSHOT_RAW_VALI_DPATH" raw

echo
echo "=== Combined detector + segmenter on validation: zero-shot default ==="
run_combined_eval "$VALI_FPATH" "$ZEROSHOT_PACKAGE_FPATH" "$COMBINED_ZEROSHOT_DEFAULT_VALI_DPATH" default

echo
echo "=== v2 summary ==="
python - "$V2_ROOT" <<'PY'
import json
import sys
from pathlib import Path

v2_root = Path(sys.argv[1])
rows = [
    ('detector_only_vali', v2_root / 'eval_detector_only/vali/eval/detect_metrics.json', None),
    ('detector_only_test', v2_root / 'eval_detector_only/test/eval/detect_metrics.json', None),
    ('gtbox_tuned_raw_vali', v2_root / 'eval_gtbox_segmenter/vali/tuned_raw/eval/detect_metrics.json', v2_root / 'eval_gtbox_segmenter/vali/tuned_raw/geometry_report.json'),
    ('gtbox_tuned_default_vali', v2_root / 'eval_gtbox_segmenter/vali/tuned_default/eval/detect_metrics.json', v2_root / 'eval_gtbox_segmenter/vali/tuned_default/geometry_report.json'),
    ('gtbox_zeroshot_raw_vali', v2_root / 'eval_gtbox_segmenter/vali/zeroshot_raw/eval/detect_metrics.json', v2_root / 'eval_gtbox_segmenter/vali/zeroshot_raw/geometry_report.json'),
    ('gtbox_zeroshot_default_vali', v2_root / 'eval_gtbox_segmenter/vali/zeroshot_default/eval/detect_metrics.json', v2_root / 'eval_gtbox_segmenter/vali/zeroshot_default/geometry_report.json'),
    ('combined_tuned_raw_vali', v2_root / 'eval_detector_segmenter/vali/tuned_raw/eval/detect_metrics.json', v2_root / 'eval_detector_segmenter/vali/tuned_raw/geometry_report.json'),
    ('combined_tuned_default_vali', v2_root / 'eval_detector_segmenter/vali/tuned_default/eval/detect_metrics.json', v2_root / 'eval_detector_segmenter/vali/tuned_default/geometry_report.json'),
    ('combined_zeroshot_raw_vali', v2_root / 'eval_detector_segmenter/vali/zeroshot_raw/eval/detect_metrics.json', v2_root / 'eval_detector_segmenter/vali/zeroshot_raw/geometry_report.json'),
    ('combined_zeroshot_default_vali', v2_root / 'eval_detector_segmenter/vali/zeroshot_default/eval/detect_metrics.json', v2_root / 'eval_detector_segmenter/vali/zeroshot_default/geometry_report.json'),
]

for name, metrics_fpath, geom_fpath in rows:
    metrics = json.loads(metrics_fpath.read_text())
    ap = metrics['nocls_measures']['ap']
    line = f'{name:30s} ap={ap:.3f}'
    if geom_fpath is not None:
        geom = json.loads(geom_fpath.read_text())['summary']
        prompt_iou = geom.get('prompt_iou')
        source_gt_iou = geom.get('source_gt_iou')
        prompt_part = 'prompt_iou=n/a' if prompt_iou is None else f'prompt_iou_med={prompt_iou["median"]:.3f}'
        gt_part = '' if source_gt_iou is None else f' source_gt_iou_med={source_gt_iou["median"]:.3f}'
        line += f' {prompt_part}{gt_part}'
    print(line)

for label, report_name in [
    ('vali_image_size_report', 'vali_image_size_report.json'),
    ('test_image_size_report', 'test_image_size_report.json'),
]:
    report = json.loads((v2_root / 'dataset_geometry' / report_name).read_text())
    summary = report['summary']
    print(
        f'{label:30s} exact={summary["exact_match"]} '
        f'swapped={summary["swapped_match"]} other={summary["other_mismatch"]} '
        f'load_errors={summary["load_errors"]}'
    )
PY

echo
echo "v2 debug run completed"
printf '  %-32s %s\n' "TUNED_PACKAGE_FPATH" "$TUNED_PACKAGE_FPATH"
printf '  %-32s %s\n' "ZEROSHOT_PACKAGE_FPATH" "$ZEROSHOT_PACKAGE_FPATH"
printf '  %-32s %s\n' "VALI_SIZE_REPORT" "$SIZE_DIAG_DPATH/vali_image_size_report.json"
printf '  %-32s %s\n' "TEST_SIZE_REPORT" "$SIZE_DIAG_DPATH/test_image_size_report.json"
