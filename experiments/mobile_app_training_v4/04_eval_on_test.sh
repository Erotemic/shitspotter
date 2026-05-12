#!/bin/bash
# Evaluate a trained DEIMv2 detector on the v9 simplified test split, using
# the same kwcoco eval path the v9 results were reported with.
#
# This re-uses the existing foundation_v3 cli_predict_boxes path, so the
# numbers come back in the canonical form (simplified GT, IoU=0.5).
#
# Usage:
#   bash 04_eval_on_test.sh <variant> [<run_tag>] [<input_h> <input_w>]
#
# Outputs:
#   $V4_ROOT/eval/<variant>_<tag>_<H>x<W>/
#     pred_boxes.kwcoco.zip
#     eval/detect_metrics.json   ← "primary detection number" for v4 vs v9

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

V4_VARIANT="${1:-deimv2_n}"
V4_RUN_TAG="${2:-tile_g${V4_TILE_GRID}}"
INPUT_H="${3:-}"
INPUT_W="${4:-}"

if [ -z "$INPUT_H" ] || [ -z "$INPUT_W" ]; then
    read -r INPUT_H INPUT_W <<< "$(v4_variant_input_size "$V4_VARIANT")"
fi

WORKDIR="$V4_ROOT/runs/${V4_VARIANT}_${V4_RUN_TAG}_${INPUT_H}x${INPUT_W}"
GENERATED_CFG_FPATH="$WORKDIR/generated_configs/train.yml"
v4_require_path "$WORKDIR"
v4_require_path "$GENERATED_CFG_FPATH"

# Dispatch v4_mock_* through the in-tree mock evaluator. Same on-disk
# layout as the DEIMv2 path so eligibility_manifest reads it uniformly.
case "$V4_VARIANT" in
    v4_mock*)
        EVAL_DPATH="$V4_ROOT/eval/${V4_VARIANT}_${V4_RUN_TAG}_${INPUT_H}x${INPUT_W}"
        mkdir -p "$EVAL_DPATH/eval"
        # Mock dispatcher always uses the V4_TEST_FPATH as-is — the
        # mock is for plumbing tests, so we don't try to share v9's
        # simplified test GT (which may have annotations missing bbox
        # fields that crash kwcoco eval). Override with
        # V4_MOCK_TEST_KWCOCO if you want a different test set.
        SIMPLIFIED_TEST_FPATH="${V4_MOCK_TEST_KWCOCO:-$V4_TEST_FPATH}"
        echo "=== mobile_app_training_v4 / 04 eval (mock dispatcher) ==="
        printf '  %-32s %s\n' "WORKDIR"               "$WORKDIR"
        printf '  %-32s %s\n' "EVAL_DPATH"            "$EVAL_DPATH"
        printf '  %-32s %s\n' "SIMPLIFIED_TEST_FPATH" "$SIMPLIFIED_TEST_FPATH"
        "$PYTHON_BIN" "$V4_DEV_DPATH/v4_mock.py" evaluate \
            --workdir "$WORKDIR" \
            --test_kwcoco "$SIMPLIFIED_TEST_FPATH" \
            --out_dir "$EVAL_DPATH/eval" \
            --score_thresh 0.05 \
            --input_h "$INPUT_H" --input_w "$INPUT_W"
        exit 0
        ;;
esac

# Pick the best checkpoint (same selection rule as 03_export_onnx.sh)
CKPT_FPATH=""
for cand in best_stg2.pth best_stg1.pth last.pth; do
    if [ -f "$WORKDIR/$cand" ]; then
        CKPT_FPATH="$WORKDIR/$cand"; break
    fi
done
if [ -z "$CKPT_FPATH" ]; then
    CKPT_FPATH=$(ls -1 "$WORKDIR"/checkpoint*.pth 2>/dev/null | sort -V | tail -n 1)
fi
v4_require_path "$CKPT_FPATH"

# v9-canonical simplified test GT — produced by the v9 script under the
# read-only DVC root. If it's missing on this host (because v9 was never
# executed locally), fall back to running the same simplify step into
# $V4_ROOT/data so the numbers are still canonical.
SIMPLIFIED_TEST_FPATH="$DVC_EXPT_DPATH_RO/foundation_detseg_v3/v9/detector_prepared/test.simplified.kwcoco.zip"
if [ ! -f "$SIMPLIFIED_TEST_FPATH" ]; then
    SIMPLIFIED_TEST_FPATH="$V4_ROOT/data/test.simplified.kwcoco.zip"
    if [ ! -f "$SIMPLIFIED_TEST_FPATH" ]; then
        echo "  generating simplified test GT (v9-style merge) at $SIMPLIFIED_TEST_FPATH"
        "$PYTHON_BIN" -m shitspotter.cli.simplify_kwcoco \
            --src "$V4_TEST_FPATH" \
            --dst "$SIMPLIFIED_TEST_FPATH" \
            --minimum_instances "$V4_SIMPLIFY_MIN_INSTANCES"
    fi
fi
v4_require_path "$SIMPLIFIED_TEST_FPATH"

EVAL_DPATH="$V4_ROOT/eval/${V4_VARIANT}_${V4_RUN_TAG}_${INPUT_H}x${INPUT_W}"
mkdir -p "$EVAL_DPATH"
PACKAGE_FPATH="$EVAL_DPATH/package.yaml"

# ---------------------------------------------------------------------------
# Build a foundation_v3 deimv2_sam2 package that points at our trained
# detector and the existing zero-shot SAM2.1 base+ checkpoint. We do not
# re-train SAM2 here — we are evaluating the *detector* in isolation, with
# pred_boxes only.
# ---------------------------------------------------------------------------
SAM2_BPLUS_CKPT="${SAM2_BPLUS_CKPT:-$SHITSPOTTER_DPATH/tpl/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt}"

DETECTOR_PRESET="deimv2_s"
case "$V4_VARIANT" in
    deimv2_pico|deimv2_n)
        # No upstream preset for these in shitspotter.algo_foundation_v3 yet,
        # so we synthesise an inline preset by passing repo_envvar +
        # config_relpath via the package YAML directly.
        DETECTOR_PRESET="inline"
        ;;
    deimv2_s)
        DETECTOR_PRESET="deimv2_s"
        ;;
    deimv2_m)
        DETECTOR_PRESET="deimv2_m"
        ;;
esac

if [ "$DETECTOR_PRESET" = "inline" ]; then
    cat > "$PACKAGE_FPATH" <<EOF
format_version: 1
backend: deimv2_sam2
detector:
  backend: deimv2
  variant: $V4_VARIANT
  repo_envvar: SHITSPOTTER_DEIMV2_REPO_DPATH
  config_fpath: $GENERATED_CFG_FPATH
  checkpoint_fpath: $CKPT_FPATH
  input_size: [$INPUT_H, $INPUT_W]
  device: cuda:0
segmenter:
  preset: sam2.1_hiera_base_plus
  checkpoint_fpath: $SAM2_BPLUS_CKPT
postprocess:
  score_thresh: 0.30
  nms_thresh: 0.50
  crop_padding: 32
  polygon_simplify: 2.0
  min_component_area: 32
  keep_largest_component: true
label_mapping:
  '0': poop
  poop: poop
metadata:
  name: v4_${V4_VARIANT}_${V4_RUN_TAG}
  family: mobile_app_training_v4
EOF
else
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_package build \
        "$PACKAGE_FPATH" \
        --backend deimv2_sam2 \
        --detector_preset "$DETECTOR_PRESET" \
        --segmenter_preset sam2.1_hiera_base_plus \
        --detector_checkpoint_fpath "$CKPT_FPATH" \
        --segmenter_checkpoint_fpath "$SAM2_BPLUS_CKPT" \
        --metadata_name "v4_${V4_VARIANT}_${V4_RUN_TAG}"
    # Re-write the detector input_size + config in case the preset doesn't
    # match what we trained at.
    "$PYTHON_BIN" - <<PY
from pathlib import Path
import yaml
fpath = Path("$PACKAGE_FPATH")
data = yaml.safe_load(fpath.read_text())
det = data.setdefault("detector", {})
det["config_fpath"] = "$GENERATED_CFG_FPATH"
det["input_size"] = [int("$INPUT_H"), int("$INPUT_W")]
fpath.write_text(yaml.safe_dump(data, sort_keys=False))
PY
fi

# ---------------------------------------------------------------------------
# Predict + evaluate
# ---------------------------------------------------------------------------
PRED_BOXES_FPATH="$EVAL_DPATH/pred_boxes.kwcoco.zip"
if [ ! -f "$PRED_BOXES_FPATH" ] || [ "${FORCE_REPRED:-0}" = "1" ]; then
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        "$SIMPLIFIED_TEST_FPATH" \
        --package_fpath "$PACKAGE_FPATH" \
        --dst "$PRED_BOXES_FPATH" \
        --score_thresh 0.30 \
        --nms_thresh 0.50
else
    echo "  reusing $PRED_BOXES_FPATH"
fi

mkdir -p "$EVAL_DPATH/eval"

# Skip if eval is already complete (sweep restart). Override with
# FORCE_REEVAL=1.
if [ -f "$EVAL_DPATH/eval/detect_metrics.json" ] && [ "${FORCE_REEVAL:-0}" != "1" ]; then
    echo "  reusing existing $EVAL_DPATH/eval/detect_metrics.json"
    echo "  set FORCE_REEVAL=1 to re-run kwcoco eval"
    "$PYTHON_BIN" - <<PY
import json
fpath = "$EVAL_DPATH/eval/detect_metrics.json"
with open(fpath) as f:
    data = json.load(f)
def find_ap(node):
    if isinstance(node, dict):
        if "nocls_measures" in node and isinstance(node["nocls_measures"], dict):
            v = node["nocls_measures"].get("ap")
            if v is not None: return v
        for v in node.values():
            r = find_ap(v)
            if r is not None: return r
    elif isinstance(node, list):
        for v in node:
            r = find_ap(v)
            if r is not None: return r
    return None
ap = find_ap(data)
print(f"\\nv4 ${V4_VARIANT} ${V4_RUN_TAG} ${INPUT_H}x${INPUT_W} test AP (simplified GT, IoU=0.5) = {ap}")
print("(v9 OpenGroundingDINO baseline = 0.766)")
PY
    exit 0
fi

# Pre-filter both true GT and predictions so kwcoco eval doesn't trip on
# annotations with bbox=None or no 'bbox' key — the v9 simplified test
# GT carries "caption-only" negatives that look like that, and the
# kwcoco coercer iterates them via [a['bbox'] for a in anns] which
# raises KeyError. See Q6 in
# dev/benchmark-candidates/pipeline-bootstrap-questions.md.
TRUE_FILTERED_FPATH="$EVAL_DPATH/test.simplified.bbox_only.kwcoco.zip"
PRED_FILTERED_FPATH="$EVAL_DPATH/pred_boxes.bbox_only.kwcoco.zip"
"$PYTHON_BIN" - <<PY
"""Drop anns where 'bbox' is missing/None or not a length-4 sequence.
Caches the filtered output. Idempotent: re-running is a no-op when the
output is already newer than the input."""
import os, sys
import kwcoco

def _filter(src, dst):
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        print(f"  reusing filtered: {dst}")
        return
    d = kwcoco.CocoDataset(src)
    keep, drop = 0, 0
    drop_ids = []
    for ann in list(d.anns.values()):
        bbox = ann.get('bbox')
        ok = (
            isinstance(bbox, (list, tuple))
            and len(bbox) == 4
            and all(v is not None for v in bbox)
        )
        if ok:
            keep += 1
        else:
            drop_ids.append(ann['id']); drop += 1
    for aid in drop_ids:
        d.remove_annotation(aid)
    d._update_fpath(dst)
    d.dump(newroot=os.path.dirname(dst) or '.')
    print(f"  filtered {src} → {dst}  (kept={keep}, dropped={drop})")

_filter("$SIMPLIFIED_TEST_FPATH", "$TRUE_FILTERED_FPATH")
_filter("$PRED_BOXES_FPATH",      "$PRED_FILTERED_FPATH")
PY

"$PYTHON_BIN" -m kwcoco eval \
    --true_dataset "$TRUE_FILTERED_FPATH" \
    --pred_dataset "$PRED_FILTERED_FPATH" \
    --out_dpath "$EVAL_DPATH/eval" \
    --out_fpath "$EVAL_DPATH/eval/detect_metrics.json" \
    --confusion_fpath "$EVAL_DPATH/eval/confusion.kwcoco.zip" \
    --draw False \
    --iou_thresh 0.5 >/dev/null

"$PYTHON_BIN" - <<PY
import json
fpath = "$EVAL_DPATH/eval/detect_metrics.json"
with open(fpath) as f:
    data = json.load(f)
def find_ap(node):
    if isinstance(node, dict):
        if "nocls_measures" in node and isinstance(node["nocls_measures"], dict):
            v = node["nocls_measures"].get("ap")
            if v is not None: return v
        for v in node.values():
            r = find_ap(v)
            if r is not None: return r
    elif isinstance(node, list):
        for v in node:
            r = find_ap(v)
            if r is not None: return r
    return None
ap = find_ap(data)
print(f"\\nv4 ${V4_VARIANT} ${V4_RUN_TAG} ${INPUT_H}x${INPUT_W} test AP (simplified GT, IoU=0.5) = {ap}")
print("(v9 OpenGroundingDINO baseline = 0.766)")
PY
