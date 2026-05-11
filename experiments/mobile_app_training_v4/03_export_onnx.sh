#!/bin/bash
# Export a trained DEIMv2 checkpoint to ONNX for the phone app.
#
# Usage:
#   bash 03_export_onnx.sh <variant> [<run_tag>] [<input_h> <input_w>]
#
# Examples:
#   bash 03_export_onnx.sh deimv2_n
#   bash 03_export_onnx.sh deimv2_n tile_g2 320 320
#   bash 03_export_onnx.sh deimv2_pico tile_g2 320 320
#   bash 03_export_onnx.sh deimv2_s tile_g2 640 640
#
# Outputs (under the run workdir):
#   <workdir>/export/deimv2_<variant>_h<H>_w<W>.onnx
#   <workdir>/export/deimv2_<variant>_h<H>_w<W>.modelspec.json
#
# Notes:
# - The DEIMv2 export wraps model + postprocessor; the resulting graph
#   takes (images: 1x3xHxW float32, orig_target_sizes: 1x2 int64) and
#   returns (labels, boxes, scores). Boxes are in pixel coordinates wrt
#   orig_target_sizes — for the phone app, pass [W, H] in display units
#   so the boxes come back already mapped to the model input grid.
# - Batch dim is left dynamic (N) so the exported graph can be run with
#   batch=1 on-device or larger batches on desktop benchmarking.

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

# Dispatch v4_mock_* to the in-tree mock exporter and short-circuit
# the DEIMv2 path. The mock writes to the same on-disk layout
# (workdir/export/<name>.onnx + .modelspec.json) so the rest of the
# pipeline doesn't care.
case "$V4_VARIANT" in
    v4_mock*)
        EXPORT_DPATH="$WORKDIR/export"
        mkdir -p "$EXPORT_DPATH"
        EXPORT_BASENAME="${V4_VARIANT}_h${INPUT_H}_w${INPUT_W}"
        EXPORT_ONNX_FPATH="$EXPORT_DPATH/${EXPORT_BASENAME}.onnx"
        EXPORT_MODELSPEC_FPATH="$EXPORT_DPATH/${EXPORT_BASENAME}.modelspec.json"
        echo "=== mobile_app_training_v4 / 03 ONNX export (mock dispatcher) ==="
        printf '  %-32s %s\n' "VARIANT"           "$V4_VARIANT"
        printf '  %-32s %s\n' "WORKDIR"           "$WORKDIR"
        printf '  %-32s %s\n' "EXPORT_ONNX_FPATH" "$EXPORT_ONNX_FPATH"
        "$PYTHON_BIN" "$V4_DEV_DPATH/v4_mock.py" export \
            --workdir "$WORKDIR" \
            --export_h "$INPUT_H" --export_w "$INPUT_W" \
            --out "$EXPORT_ONNX_FPATH"
        # Generate the same modelspec.json sidecar the DEIMv2 export emits,
        # using policy.json as the source of truth for candidate identity.
        POLICY_JSON_FPATH="$WORKDIR/policy.json"
        v4_require_path "$POLICY_JSON_FPATH"
        "$PYTHON_BIN" - <<PY > "$EXPORT_MODELSPEC_FPATH"
import json, pathlib
policy = json.loads(pathlib.Path("$POLICY_JSON_FPATH").read_text())
variant   = policy.get("variant", "${V4_VARIANT}")
exp_h     = int(policy.get("export_input_h", ${INPUT_H}))
exp_w     = int(policy.get("export_input_w", ${INPUT_W}))
train_pol = policy.get("train_resolution_policy", "")
tile_pol  = policy.get("tile_training_policy", "")
scales    = policy.get("effective_train_scales", [])
phone_model_id = f"shitspotter-{variant}-h{exp_h}w{exp_w}-{train_pol}"
spec = {
    "modelId": phone_model_id,
    "candidateId": policy.get("candidate_id", phone_model_id),
    "phoneModelId": phone_model_id,
    "displayName": f"ShitSpotter {variant} {exp_h}x{exp_w} ({train_pol})",
    "modelFile": "${EXPORT_BASENAME}.onnx",
    "format": "ONNX",
    "inputWidth": exp_w,
    "inputHeight": exp_h,
    "inputLayout": "NCHW",
    "colorOrder": "RGB",
    "normalization": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0],
                       "scale": 0.00392156862745098},
    "resizePolicy": "LETTERBOX",
    "postprocessType": "DEIMV2",
    "classNames": ["poop"],
    "scoreThreshold": 0.30,
    "iouThreshold": 0.45,
    "trainResolutionPolicy": train_pol,
    "effectiveTrainScales": scales,
    "tileTrainingPolicy": tile_pol,
    "trainingDatasetHint": f"v4_mock_smoke + {tile_pol}",
    "v4PolicyJson": "$POLICY_JSON_FPATH",
    "deimv2Schema": {
        "inputNames":  ["images", "orig_target_sizes"],
        "outputNames": ["labels", "boxes", "scores"],
        "boxFormat":   "xyxy_pixels",
        "passOrigSize": True,
    },
    "notes": "v4_mock — tiny torch detector for pipeline smoke tests; not a real detector.",
}
print(json.dumps(spec, indent=2))
PY
        ONNX_SIZE=$(du -h "$EXPORT_ONNX_FPATH" | awk '{print $1}')
        printf '  %-32s %s\n' "ONNX size" "$ONNX_SIZE"
        echo "Next: bash $V4_DEV_DPATH/04_eval_on_test.sh $V4_VARIANT $V4_RUN_TAG $INPUT_H $INPUT_W"
        exit 0
        ;;
esac

# Pick the best-available checkpoint
CKPT_FPATH=""
for cand in best_stg2.pth best_stg1.pth last.pth; do
    if [ -f "$WORKDIR/$cand" ]; then
        CKPT_FPATH="$WORKDIR/$cand"
        break
    fi
done
if [ -z "$CKPT_FPATH" ]; then
    # Fall back to the highest-numbered checkpointXXXX.pth
    CKPT_FPATH=$(ls -1 "$WORKDIR"/checkpoint*.pth 2>/dev/null | sort -V | tail -n 1)
fi
if [ -z "$CKPT_FPATH" ] || [ ! -f "$CKPT_FPATH" ]; then
    echo "No trained checkpoint found in $WORKDIR" >&2
    exit 1
fi

EXPORT_DPATH="$WORKDIR/export"
mkdir -p "$EXPORT_DPATH"
EXPORT_BASENAME="${V4_VARIANT}_h${INPUT_H}_w${INPUT_W}"
EXPORT_ONNX_FPATH="$EXPORT_DPATH/${EXPORT_BASENAME}.onnx"
EXPORT_MODELSPEC_FPATH="$EXPORT_DPATH/${EXPORT_BASENAME}.modelspec.json"

echo "=== mobile_app_training_v4 / 03 ONNX export ==="
printf '  %-32s %s\n' "VARIANT"            "$V4_VARIANT"
printf '  %-32s %s\n' "RUN_TAG"            "$V4_RUN_TAG"
printf '  %-32s %s\n' "INPUT_HW"           "${INPUT_H}x${INPUT_W}"
printf '  %-32s %s\n' "WORKDIR"            "$WORKDIR"
printf '  %-32s %s\n' "CKPT_FPATH"         "$CKPT_FPATH"
printf '  %-32s %s\n' "GENERATED_CFG"      "$GENERATED_CFG_FPATH"
printf '  %-32s %s\n' "EXPORT_ONNX_FPATH"  "$EXPORT_ONNX_FPATH"
printf '  %-32s %s\n' "EXPORT_MODELSPEC"   "$EXPORT_MODELSPEC_FPATH"

# DEIMv2's exporter writes alongside the resume checkpoint, deriving the
# output path by replacing .pth with .onnx. We work in a temp staging
# directory so we don't pollute the workdir, then copy the result out.
STAGING_DPATH="$EXPORT_DPATH/.staging_${EXPORT_BASENAME}"
mkdir -p "$STAGING_DPATH"
STAGING_CKPT="$STAGING_DPATH/${EXPORT_BASENAME}.pth"
ln -sf "$CKPT_FPATH" "$STAGING_CKPT"

(
    cd "$SHITSPOTTER_DEIMV2_REPO_DPATH"
    "$PYTHON_BIN" tools/deployment/export_onnx.py \
        --config "$GENERATED_CFG_FPATH" \
        --resume "$STAGING_CKPT" \
        --opset 17 \
        --check \
        --simplify
)

mv "$STAGING_DPATH/${EXPORT_BASENAME}.onnx" "$EXPORT_ONNX_FPATH"
rm -rf "$STAGING_DPATH"

# ---------------------------------------------------------------------------
# Emit a modelspec.json sidecar so the phone-app side knows exactly how to
# preprocess for this model. Mirrors the Kotlin ModelSpec record fields so
# the data is round-trip compatible.
#
# The candidate identity (variant, export_h, export_w, train_resolution_policy,
# tile_training_policy) is loaded from <workdir>/policy.json so the modelId
# here matches the phone_model_id eligibility_manifest.py emits — there is
# exactly one ID per candidate, and the model file can travel independently
# because every identity field is duplicated into the sidecar.
# ---------------------------------------------------------------------------
POLICY_JSON_FPATH="$WORKDIR/policy.json"
if [ ! -f "$POLICY_JSON_FPATH" ]; then
    echo "Missing $POLICY_JSON_FPATH — was this trained with v4?" >&2
    exit 1
fi

"$PYTHON_BIN" - <<PY > "$EXPORT_MODELSPEC_FPATH"
import json, pathlib
policy = json.loads(pathlib.Path("$POLICY_JSON_FPATH").read_text())

variant   = policy.get("variant", "${V4_VARIANT}")
exp_h     = int(policy.get("export_input_h", ${INPUT_H}))
exp_w     = int(policy.get("export_input_w", ${INPUT_W}))
train_pol = policy.get("train_resolution_policy", "")
tile_pol  = policy.get("tile_training_policy", "")
scales    = policy.get("effective_train_scales", [])

# *** This is the canonical phone_model_id format. eligibility_manifest.py
# *** must produce the same string from the same policy.json fields.
phone_model_id = f"shitspotter-{variant}-h{exp_h}w{exp_w}-{train_pol}"
candidate_id   = policy.get("candidate_id", phone_model_id)

spec = {
    "modelId": phone_model_id,
    "candidateId": candidate_id,
    "phoneModelId": phone_model_id,
    "displayName": f"ShitSpotter {variant} {exp_h}x{exp_w} ({train_pol})",
    "modelFile": "${EXPORT_BASENAME}.onnx",
    "format": "ONNX",
    "inputWidth": exp_w,
    "inputHeight": exp_h,
    "inputLayout": "NCHW",
    "colorOrder": "RGB",
    "normalization": {
        "mean": [0.0, 0.0, 0.0],
        "std":  [1.0, 1.0, 1.0],
        "scale": ${V4_VARIANT_NORM_SCALE_DEFAULT:-0.00392156862745098},
    },
    "resizePolicy": "LETTERBOX",
    "postprocessType": "DEIMV2",
    "classNames": ["poop"],
    "scoreThreshold": 0.30,
    "iouThreshold": 0.45,

    # Full candidate identity — answers "what is this file, exactly?"
    # without needing to consult the eligibility manifest.
    "trainResolutionPolicy": train_pol,
    "effectiveTrainScales": scales,
    "tileTrainingPolicy": tile_pol,
    "trainingDatasetHint": (
        f"v9_split_train_imgs10671_b277c63d "
        f"+ {tile_pol}"
    ),
    "v4PolicyJson": "$POLICY_JSON_FPATH",

    "deimv2Schema": {
        "inputNames":  ["images", "orig_target_sizes"],
        "outputNames": ["labels", "boxes", "scores"],
        "boxFormat":   "xyxy_pixels",
        "passOrigSize": True,
    },
    "notes": (
        "DEIMv2 export with built-in postprocessor. "
        "Pass orig_target_sizes=[[W, H]] (int64) and an HxWx3 RGB image "
        "scaled to [0, 1]. Outputs are top-K detections per query (typically "
        "300 queries); apply a score threshold and (optionally) NMS in the app."
    ),
}
print(json.dumps(spec, indent=2))
PY

echo
echo "Export complete:"
printf '  %-32s %s\n' "ONNX"       "$EXPORT_ONNX_FPATH"
printf '  %-32s %s\n' "modelspec"  "$EXPORT_MODELSPEC_FPATH"

ONNX_SIZE=$(du -h "$EXPORT_ONNX_FPATH" | awk '{print $1}')
printf '  %-32s %s\n' "ONNX size"  "$ONNX_SIZE"

echo
echo "Next:"
echo "  bash $V4_DEV_DPATH/04_eval_on_test.sh $V4_VARIANT $V4_RUN_TAG ${INPUT_H} ${INPUT_W}"
echo "  python $V4_DEV_DPATH/05_desktop_onnx_parity.py \\"
echo "      --pth_ckpt $CKPT_FPATH \\"
echo "      --pth_config $GENERATED_CFG_FPATH \\"
echo "      --onnx $EXPORT_ONNX_FPATH \\"
echo "      --image $SHITSPOTTER_DPATH/tpl/YOLOX/assets/dog.jpg"
