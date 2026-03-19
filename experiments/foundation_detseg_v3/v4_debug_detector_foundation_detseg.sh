#!/bin/bash
set -euo pipefail

# Hard-coded debug run for the v4 simplify-preprocessed detector branch.
# The goal is to determine if the catastrophic detector collapse is caused by:
#   1) bad checkpoint selection,
#   2) bad score-threshold / postprocess defaults,
#   3) surprising changes in the simplified training data,
#   4) or a genuinely broken detector training outcome.

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
import math
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
if isinstance(ap, float) and math.isnan(ap):
    print('nan')
else:
    print(f'{float(ap):.3f}')
PY
}

REPO_DPATH="$(canonical_existing_path /home/joncrall/code/shitspotter)"
DATA_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_dvc)"
EXPT_DPATH="$(canonical_existing_path /home/joncrall/data/dvc-repos/shitspotter_expt_dvc)"

SHITSPOTTER_DEIMV2_REPO_DPATH="$REPO_DPATH/tpl/DEIMv2"
SHITSPOTTER_SAM2_REPO_DPATH="$REPO_DPATH/tpl/segment-anything-2"
SHITSPOTTER_MASKDINO_REPO_DPATH="$REPO_DPATH/tpl/MaskDINO"

TRAIN_FPATH="$DATA_DPATH/train_imgs5747_1e73d54f.kwcoco.zip"
VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"

V3_ROOT="$EXPT_DPATH/foundation_detseg_v3/v3"
V4_ROOT="$EXPT_DPATH/foundation_detseg_v3/v4"
DEBUG_ROOT="$EXPT_DPATH/foundation_detseg_v3/v4_debug_detector"
PACKAGE_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3/packages"

V3_DETECTOR_WORKDIR="$V3_ROOT/train_detector_deimv2_m"
V4_DETECTOR_WORKDIR="$V4_ROOT/train_detector_deimv2_m"

V3_DETECTOR_CKPT="$(choose_first_existing_file \
    "$V3_DETECTOR_WORKDIR/best_stg2.pth" \
    "$V3_DETECTOR_WORKDIR/best_stg1.pth" \
    "$V3_DETECTOR_WORKDIR/last.pth")"

V4_BEST_STG2="$V4_DETECTOR_WORKDIR/best_stg2.pth"
V4_BEST_STG1="$V4_DETECTOR_WORKDIR/best_stg1.pth"
V4_LAST="$V4_DETECTOR_WORKDIR/last.pth"

V4_PREPROC_DPATH="$V4_DETECTOR_WORKDIR/preprocessed_kwcoco"
V4_RESIZED_TRAIN_FPATH="$V4_PREPROC_DPATH/train_maxdim640.kwcoco.zip"
V4_RESIZED_VALI_FPATH="$V4_PREPROC_DPATH/vali_maxdim640.kwcoco.zip"
V4_SIMPLIFIED_TRAIN_FPATH="$V4_PREPROC_DPATH/train_maxdim640.kwcoco.simplified.kwcoco.zip"
V4_SIMPLIFIED_VALI_FPATH="$V4_PREPROC_DPATH/vali_maxdim640.kwcoco.simplified.kwcoco.zip"

SAM2_ZERO_SHOT_CKPT="$SHITSPOTTER_SAM2_REPO_DPATH/checkpoints/sam2.1_hiera_base_plus.pt"

for required in \
    "$REPO_DPATH" \
    "$SHITSPOTTER_DEIMV2_REPO_DPATH" \
    "$SHITSPOTTER_SAM2_REPO_DPATH" \
    "$TRAIN_FPATH" \
    "$VALI_FPATH" \
    "$V3_DETECTOR_CKPT" \
    "$V4_RESIZED_TRAIN_FPATH" \
    "$V4_SIMPLIFIED_TRAIN_FPATH" \
    "$SAM2_ZERO_SHOT_CKPT"; do
    require_path "$required"
done

export SHITSPOTTER_DPATH="$REPO_DPATH"
export FOUNDATION_V3_DEV_DPATH="$REPO_DPATH/experiments/foundation_detseg_v3"
export DVC_DATA_DPATH="$DATA_DPATH"
export DVC_EXPT_DPATH="$EXPT_DPATH"
export SHITSPOTTER_DEIMV2_REPO_DPATH
export SHITSPOTTER_SAM2_REPO_DPATH
export SHITSPOTTER_MASKDINO_REPO_DPATH

mkdir -p "$DEBUG_ROOT"

echo "v4 detector debug"
printf '  %-28s %s\n' "REPO_DPATH" "$REPO_DPATH"
printf '  %-28s %s\n' "V3_DETECTOR_CKPT" "$V3_DETECTOR_CKPT"
printf '  %-28s %s\n' "V4_BEST_STG2" "$V4_BEST_STG2"
printf '  %-28s %s\n' "V4_BEST_STG1" "$V4_BEST_STG1"
printf '  %-28s %s\n' "V4_LAST" "$V4_LAST"
printf '  %-28s %s\n' "DEBUG_ROOT" "$DEBUG_ROOT"

echo
echo "=== Dataset Stats ==="
python - "$TRAIN_FPATH" "$V4_RESIZED_TRAIN_FPATH" "$V4_SIMPLIFIED_TRAIN_FPATH" \
          "$VALI_FPATH" "$V4_RESIZED_VALI_FPATH" "$V4_SIMPLIFIED_VALI_FPATH" \
          "$DEBUG_ROOT/dataset_stats.json" <<'PY'
import json
import statistics
import sys
from pathlib import Path

import kwcoco

inputs = sys.argv[1:-1]
output_fpath = Path(sys.argv[-1])

def summarize_dataset(fpath):
    dset = kwcoco.CocoDataset(fpath)
    annots = dset.annots().objs
    images = dset.images().objs
    bbox_areas = []
    anns_per_image = []
    category_hist = {}
    for img in images:
        anns = dset.annots(gid=img['id']).objs
        anns_per_image.append(len(anns))
    for ann in annots:
        bbox = ann.get('bbox', None)
        if bbox is not None:
            bbox_areas.append(float(bbox[2]) * float(bbox[3]))
        cid = ann.get('category_id', None)
        catname = dset.cats.get(cid, {}).get('name', str(cid))
        category_hist[catname] = category_hist.get(catname, 0) + 1
    report = {
        'fpath': str(fpath),
        'num_images': len(images),
        'num_annotations': len(annots),
        'category_hist': category_hist,
        'images_with_annotations': int(sum(n > 0 for n in anns_per_image)),
        'images_with_multiple_annotations': int(sum(n > 1 for n in anns_per_image)),
        'max_annotations_per_image': int(max(anns_per_image) if anns_per_image else 0),
        'mean_annotations_per_image': float(statistics.fmean(anns_per_image) if anns_per_image else 0.0),
        'bbox_area_mean': float(statistics.fmean(bbox_areas) if bbox_areas else 0.0),
        'bbox_area_median': float(statistics.median(bbox_areas) if bbox_areas else 0.0),
    }
    return report

reports = [summarize_dataset(fpath) for fpath in inputs]
output_fpath.write_text(json.dumps(reports, indent=2))
for report in reports:
    print(json.dumps({
        'fpath': report['fpath'],
        'num_images': report['num_images'],
        'num_annotations': report['num_annotations'],
        'category_hist': report['category_hist'],
        'images_with_multiple_annotations': report['images_with_multiple_annotations'],
        'mean_annotations_per_image': report['mean_annotations_per_image'],
        'bbox_area_median': report['bbox_area_median'],
    }, indent=2))
PY

run_detector_probe() {
    local tag="$1"
    local ckpt="$2"
    local score_thresh="$3"

    local probe_root="$DEBUG_ROOT/$tag/score_${score_thresh}"
    local package_fpath="$PACKAGE_DPATH/${tag}_score${score_thresh}.yaml"
    local pred_fpath="$probe_root/pred_boxes.kwcoco.zip"
    local metrics_dpath="$probe_root/eval"
    local metrics_fpath="$metrics_dpath/detect_metrics.json"
    local confusion_fpath="$metrics_dpath/confusion.kwcoco.zip"
    local pred_report_fpath="$probe_root/prediction_report.json"

    mkdir -p "$probe_root" "$metrics_dpath"

    python -m shitspotter.algo_foundation_v3.cli_package build \
        "$package_fpath" \
        --backend deimv2_sam2 \
        --detector_preset deimv2_m \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$ckpt" \
        --segmenter_checkpoint_fpath "$SAM2_ZERO_SHOT_CKPT" \
        --metadata_name "$tag"

    python -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        --src="$VALI_FPATH" \
        --package_fpath="$package_fpath" \
        --score_thresh="$score_thresh" \
        --dst="$pred_fpath"

    python -m kwcoco eval \
        --true_dataset="$VALI_FPATH" \
        --pred_dataset="$pred_fpath" \
        --out_dpath="$metrics_dpath" \
        --out_fpath="$metrics_fpath" \
        --confusion_fpath="$confusion_fpath" \
        --draw=False \
        --iou_thresh=0.5

    python - "$pred_fpath" "$pred_report_fpath" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

import kwcoco

pred_fpath = sys.argv[1]
report_fpath = Path(sys.argv[2])
dset = kwcoco.CocoDataset(pred_fpath)
anns = dset.annots().objs
scores = [float(a.get('score', 0.0)) for a in anns]
per_image = [len(dset.annots(gid=img['id'])) for img in dset.images().objs]
report = {
    'pred_fpath': pred_fpath,
    'num_images': len(dset.images()),
    'num_pred_annotations': len(anns),
    'images_with_predictions': int(sum(n > 0 for n in per_image)),
    'max_predictions_per_image': int(max(per_image) if per_image else 0),
    'score_mean': None if not scores else float(statistics.fmean(scores)),
    'score_median': None if not scores else float(statistics.median(scores)),
    'score_min': None if not scores else float(min(scores)),
    'score_max': None if not scores else float(max(scores)),
}
report_fpath.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY

    printf '  %-28s ap=%s score_thresh=%s ckpt=%s\n' \
        "$tag" "$(read_ap "$metrics_fpath")" "$score_thresh" "$ckpt"
}

echo
echo "=== Detector Probes On Validation ==="

run_detector_probe "v3_reference" "$V3_DETECTOR_CKPT" "0.2"
run_detector_probe "v3_reference" "$V3_DETECTOR_CKPT" "0.0"

if [ -f "$V4_BEST_STG2" ]; then
    run_detector_probe "v4_best_stg2" "$V4_BEST_STG2" "0.2"
    run_detector_probe "v4_best_stg2" "$V4_BEST_STG2" "0.0"
fi

if [ -f "$V4_BEST_STG1" ]; then
    run_detector_probe "v4_best_stg1" "$V4_BEST_STG1" "0.2"
    run_detector_probe "v4_best_stg1" "$V4_BEST_STG1" "0.0"
fi

if [ -f "$V4_LAST" ]; then
    run_detector_probe "v4_last" "$V4_LAST" "0.2"
    run_detector_probe "v4_last" "$V4_LAST" "0.0"
fi

echo
echo "=== Detector Workdir Snapshot ==="
find "$V4_DETECTOR_WORKDIR" -maxdepth 2 -type f \
    \( -name 'best_stg*.pth' -o -name 'last.pth' -o -name '*.json' -o -name '*.txt' \) | sort

echo
echo "v4 detector debug completed"
