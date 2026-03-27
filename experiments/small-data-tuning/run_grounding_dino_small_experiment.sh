#!/bin/bash
set -euo pipefail

# Prepare and launch the historical Open-GroundingDINO tuning path on a fixed
# small-data cohort. This wrapper is intentionally verbose because the external
# training code has more bespoke dataset expectations than the other families.

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

COHORT_NAME="${COHORT_NAME:-}"
COHORT_DPATH="${COHORT_DPATH:-}"
RUN_NAME="${RUN_NAME:-grounding_dino_small_default}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
_REPO_DPATH="$(small_data_repo_dpath)"
OPEN_GDINO_REPO_DPATH="${OPEN_GDINO_REPO_DPATH:-$_REPO_DPATH/tpl/Open-GroundingDino}"
GPU_NUM="${GPU_NUM:-1}"
CFG_TEMPLATE="${CFG_TEMPLATE:-config/cfg_odvg.py}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-groundingdino_swint_ogc.pth}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"

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

RUN_DPATH="$(small_data_runs_root)/grounding_dino/${SMALL_DATA_COHORT_NAME}/${RUN_NAME}"
PREP_DPATH="$RUN_DPATH/prepared"
mkdir -p "$PREP_DPATH"

RUN_MANIFEST_FPATH="$RUN_DPATH/run_manifest.json"
DATASETS_JSON_FPATH="$PREP_DPATH/shitspotter_small_datasets.json"
TRAIN_JSON_FPATH="$PREP_DPATH/train.tmp.json"
VALI_JSON_FPATH="$PREP_DPATH/vali.tmp.json"
TRAIN_ODVG_FPATH="$PREP_DPATH/train.odvg.jsonl"
LABEL_MAP_FPATH="$PREP_DPATH/label_map.json"
CFG_FPATH="$PREP_DPATH/shitspotter_small_cfg_odvg.py"

cat > "$RUN_MANIFEST_FPATH" <<EOF
{
  "model_family": "grounding_dino",
  "run_name": "$RUN_NAME",
  "cohort_name": "$SMALL_DATA_COHORT_NAME",
  "cohort_manifest_fpath": "$SMALL_DATA_COHORT_MANIFEST_FPATH",
  "repo_dpath": "$OPEN_GDINO_REPO_DPATH",
  "prepared_dpath": "$PREP_DPATH",
  "train_fpath": "$SMALL_DATA_TRAIN_KWCOCO_FPATH",
  "vali_fpath": "$SMALL_DATA_VALI_KWCOCO_FPATH",
  "test_fpath": "$SMALL_DATA_TEST_KWCOCO_FPATH",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES",
  "gpu_num": $GPU_NUM,
  "cfg_template": "$CFG_TEMPLATE",
  "pretrain_model_path": "$PRETRAIN_MODEL_PATH",
  "text_encoder_type": "$TEXT_ENCODER_TYPE"
}
EOF

cd "$OPEN_GDINO_REPO_DPATH"

# The historical Open-GroundingDINO path expects:
# 1. absolute image paths,
# 2. a COCO-like validation json,
# 3. an ODVG-formatted training jsonl,
# 4. an explicit label-map json,
# 5. a datasets json describing the train/val sources.
kwcoco reroot --absolute=True --src "$SMALL_DATA_TRAIN_KWCOCO_FPATH" --dst "$PREP_DPATH/train.absolute.kwcoco.zip"
kwcoco modify_categories --src "$PREP_DPATH/train.absolute.kwcoco.zip" --dst "$TRAIN_JSON_FPATH" --start_id=0
kwcoco reroot --absolute=True --src "$SMALL_DATA_VALI_KWCOCO_FPATH" --dst "$PREP_DPATH/vali.absolute.kwcoco.zip"
kwcoco modify_categories --src "$PREP_DPATH/vali.absolute.kwcoco.zip" --dst "$VALI_JSON_FPATH" --start_id=0
kwcoco conform --legacy=True --src "$VALI_JSON_FPATH" --inplace

"$PYTHON_BIN" tools/coco2odvg.py --input "$TRAIN_JSON_FPATH" --output "$TRAIN_ODVG_FPATH" --idmap=False

"$PYTHON_BIN" - <<PY
import json
import pathlib

train_json_fpath = pathlib.Path("$TRAIN_JSON_FPATH")
label_map_fpath = pathlib.Path("$LABEL_MAP_FPATH")
data = json.loads(train_json_fpath.read_text())
label_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
label_map_fpath.write_text(json.dumps(label_map, indent=2))
PY

cat > "$DATASETS_JSON_FPATH" <<EOF
{
  "train": [
    {
      "root": "$COHORT_DPATH",
      "anno": "$TRAIN_ODVG_FPATH",
      "label_map": "$LABEL_MAP_FPATH",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "$COHORT_DPATH",
      "anno": "$VALI_JSON_FPATH",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
EOF

cp "$CFG_TEMPLATE" "$CFG_FPATH"
sed -i 's|use_coco_eval = True|use_coco_eval = False|g' "$CFG_FPATH"
echo "" >> "$CFG_FPATH"
echo "label_list = ['poop', 'unknown']" >> "$CFG_FPATH"

printf 'GroundingDINO small-data run\n'
printf '  %-22s %s\n' "COHORT_DPATH" "$COHORT_DPATH"
printf '  %-22s %s\n' "RUN_DPATH" "$RUN_DPATH"
printf '  %-22s %s\n' "PREP_DPATH" "$PREP_DPATH"
printf '  %-22s %s\n' "DATASETS_JSON" "$DATASETS_JSON_FPATH"

export CUDA_VISIBLE_DEVICES
export GPU_NUM
export CFG="$CFG_FPATH"
export DATASETS="$DATASETS_JSON_FPATH"
export OUTPUT_DIR="$RUN_DPATH"
export PRETRAIN_MODEL_PATH
export TEXT_ENCODER_TYPE

bash train_dist.sh "$GPU_NUM" "$CFG" "$DATASETS" "$OUTPUT_DIR"
