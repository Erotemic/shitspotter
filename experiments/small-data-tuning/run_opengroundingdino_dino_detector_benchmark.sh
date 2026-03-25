#!/bin/bash
set -euo pipefail

_small_data_source="${BASH_SOURCE[0]-}"
_small_data_script_dpath="$(cd "$(dirname "$_small_data_source")" && pwd)"
# shellcheck source=experiments/small-data-tuning/common.sh
source "$_small_data_script_dpath/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_DPATH="$(small_data_repo_dpath)"
EXPT_DPATH="$(small_data_expt_dpath)"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-$EXPT_DPATH/small_data_tuning/dino_detector_benchmark}"
BENCHMARK_MANIFEST_FPATH="$BENCHMARK_ROOT/benchmark_manifest.json"
RUN_ROOT="$BENCHMARK_ROOT/runs/opengroundingdino"

OPEN_GDINO_REPO_DPATH="${OPEN_GDINO_REPO_DPATH:-$HOME/code/Open-GroundingDino}"
GPU_NUM="${GPU_NUM:-1}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-groundingdino_swint_ogc.pth}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"
CFG_TEMPLATE="${CFG_TEMPLATE:-config/cfg_odvg.py}"
FORCE_GDINO_RERUN="${FORCE_GDINO_RERUN:-False}"
CLASSES_TEXT="${CLASSES_TEXT:-[poop]}"

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

run_single_eval() {
    local true_fpath="$1"
    local checkpoint_fpath="$2"
    local cfg_fpath="$3"
    local root_dpath="$4"
    "$PYTHON_BIN" -m geowatch.mlops.schedule_evaluation \
        --params="
            pipeline: 'shitspotter.other.open_grounding_dino_pipeline.open_grounding_dino_evaluation_pipeline()'
            matrix:
                open_grounding_dino_pred.src:
                    - '$true_fpath'
                open_grounding_dino_pred.checkpoint:
                    - '$checkpoint_fpath'
                open_grounding_dino_pred.config_file:
                    - '$cfg_fpath'
                open_grounding_dino_pred.force_classname:
                    - poop
                open_grounding_dino_pred.classes:
                    - '$CLASSES_TEXT'
        " \
        --root_dpath="$root_dpath" \
        --devices="0," \
        --tmux_workers=1 \
        --backend=serial \
        --skip_existing=1 \
        --cache=0 \
        --run=1 >/dev/null
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
        subset['prepared_mscoco_fpath'],
    ]))
PY
)

VALI_KWCOCO="$("$PYTHON_BIN" - "$BENCHMARK_MANIFEST_FPATH" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], 'r').read())
print(data['eval_sets']['vali']['prepared_kwcoco_fpath'])
PY
)"
VALI_MSCOCO="$("$PYTHON_BIN" - "$BENCHMARK_MANIFEST_FPATH" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], 'r').read())
print(data['eval_sets']['vali']['prepared_mscoco_fpath'])
PY
)"
TEST_KWCOCO="$("$PYTHON_BIN" - "$BENCHMARK_MANIFEST_FPATH" <<'PY'
import json, sys
data = json.loads(open(sys.argv[1], 'r').read())
print(data['eval_sets']['test']['prepared_kwcoco_fpath'])
PY
)"

for row in "${TRAIN_ROWS[@]}"; do
    IFS=$'\t' read -r subset_name train_size train_kwcoco train_mscoco <<<"$row"
    run_dpath="$RUN_ROOT/$subset_name"
    prep_dpath="$run_dpath/prepared"
    output_dir="$run_dpath/train_output"
    summary_fpath="$run_dpath/summary.tsv"
    run_manifest_fpath="$run_dpath/run_manifest.json"
    cfg_fpath="$prep_dpath/shitspotter_cfg_odvg.py"
    datasets_json="$prep_dpath/shitspotter_datasets.json"
    train_odvg="$prep_dpath/train.odvg.jsonl"
    label_map_fpath="$prep_dpath/label_map.json"
    mkdir -p "$prep_dpath"

    printf 'OpenGroundingDINO detector benchmark\n'
    printf '  %-22s %s\n' "SUBSET" "$subset_name"
    printf '  %-22s %s\n' "TRAIN_SIZE" "$train_size"
    printf '  %-22s %s\n' "RUN_DPATH" "$run_dpath"

    cat > "$run_manifest_fpath" <<EOF
{
  "model_family": "opengroundingdino",
  "benchmark_manifest_fpath": "$BENCHMARK_MANIFEST_FPATH",
  "subset_name": "$subset_name",
  "train_size": $train_size,
  "train_kwcoco_fpath": "$train_kwcoco",
  "train_mscoco_fpath": "$train_mscoco",
  "vali_kwcoco_fpath": "$VALI_KWCOCO",
  "vali_mscoco_fpath": "$VALI_MSCOCO",
  "test_kwcoco_fpath": "$TEST_KWCOCO",
  "run_dpath": "$run_dpath",
  "open_gdino_repo_dpath": "$OPEN_GDINO_REPO_DPATH",
  "gpu_num": $GPU_NUM,
  "cfg_template": "$CFG_TEMPLATE",
  "pretrain_model_path": "$PRETRAIN_MODEL_PATH",
  "text_encoder_type": "$TEXT_ENCODER_TYPE",
  "classes_text": "$CLASSES_TEXT"
}
EOF

    cd "$OPEN_GDINO_REPO_DPATH"

    "$PYTHON_BIN" tools/coco2odvg.py --input "$train_mscoco" --output "$train_odvg" --idmap=False

    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
data = json.loads(Path("$train_mscoco").read_text())
label_map = {cat["id"]: cat["name"] for cat in data.get("categories", [])}
Path("$label_map_fpath").write_text(json.dumps(label_map, indent=2))
PY

    cat > "$datasets_json" <<EOF
{
  "train": [
    {
      "root": "$prep_dpath",
      "anno": "$train_odvg",
      "label_map": "$label_map_fpath",
      "dataset_mode": "odvg"
    }
  ],
  "val": [
    {
      "root": "$prep_dpath",
      "anno": "$VALI_MSCOCO",
      "label_map": null,
      "dataset_mode": "coco"
    }
  ]
}
EOF

    cp "$CFG_TEMPLATE" "$cfg_fpath"
    sed -i 's|use_coco_eval = True|use_coco_eval = False|g' "$cfg_fpath"
    echo "" >> "$cfg_fpath"
    echo "label_list = ['poop']" >> "$cfg_fpath"

    export GPU_NUM
    export CFG="$cfg_fpath"
    export DATASETS="$datasets_json"
    export OUTPUT_DIR="$output_dir"
    export PRETRAIN_MODEL_PATH
    export TEXT_ENCODER_TYPE
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

    # If a prior attempt died mid-training, OpenGroundingDINO will try to
    # auto-resume from a stale checkpoint. Under newer PyTorch that resume path
    # is brittle, so prefer a clean restart unless a benchmark summary already
    # proves the subset finished end-to-end.
    if [ -d "$output_dir" ] && [ ! -f "$summary_fpath" ]; then
        rm -rf "$output_dir"
    fi

    case "$FORCE_GDINO_RERUN" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON)
            rm -rf "$output_dir" "$run_dpath/checkpoint_select" "$run_dpath/test_eval" "$summary_fpath"
            ;;
    esac

    if [ ! -f "$output_dir/checkpoint0014.pth" ] || [ "$FORCE_GDINO_RERUN" = "True" ]; then
        bash train_dist.sh "$GPU_NUM" "$CFG" "$DATASETS" "$OUTPUT_DIR"
    fi

    mapfile -t CANDIDATES < <(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint*.pth' | sort)
    if [ "${#CANDIDATES[@]}" -eq 0 ]; then
        echo "No OpenGroundingDINO checkpoints found in $output_dir" >&2
        exit 1
    fi

    printf 'candidate_id\ttrain_size\tpoop_vali_ap\tnocls_vali_ap\tpoop_test_ap\tnocls_test_ap\tselected\tckpt_fpath\trun_dpath\tsummary_fpath\n' > "$summary_fpath"
    best_candidate=""
    best_ckpt=""
    best_poop_vali=""
    for ckpt_fpath in "${CANDIDATES[@]}"; do
        candidate_id="$(basename "$ckpt_fpath" .pth)"
        candidate_root="$run_dpath/checkpoint_select/$candidate_id/vali"
        run_single_eval "$VALI_KWCOCO" "$ckpt_fpath" "$cfg_fpath" "$candidate_root"
        metrics_fpath="$(find "$candidate_root" -name detect_metrics.json | head -n 1)"
        if [ -z "${metrics_fpath:-}" ]; then
            echo "Unable to locate validation metrics under $candidate_root" >&2
            exit 1
        fi
        poop_vali_ap="$(read_metric "$metrics_fpath" poop_ap)"
        nocls_vali_ap="$(read_metric "$metrics_fpath" nocls_ap)"
        printf '%s\t%s\t%s\t%s\tNA\tNA\t0\t%s\t%s\t%s\n' \
            "$candidate_id" "$train_size" "$poop_vali_ap" "$nocls_vali_ap" "$ckpt_fpath" "$run_dpath" "$summary_fpath" >> "$summary_fpath"
        if [ -z "$best_poop_vali" ] || "$PYTHON_BIN" - <<PY
best_score = float("${best_poop_vali:-0}")
candidate_score = float("$poop_vali_ap")
raise SystemExit(0 if candidate_score > best_score else 1)
PY
        then
            best_candidate="$candidate_id"
            best_ckpt="$ckpt_fpath"
            best_poop_vali="$poop_vali_ap"
        fi
    done

    best_test_root="$run_dpath/test_eval/$best_candidate"
    run_single_eval "$TEST_KWCOCO" "$best_ckpt" "$cfg_fpath" "$best_test_root"
    test_metrics_fpath="$(find "$best_test_root" -name detect_metrics.json | head -n 1)"
    poop_test_ap="$(read_metric "$test_metrics_fpath" poop_ap)"
    nocls_test_ap="$(read_metric "$test_metrics_fpath" nocls_ap)"
    best_vali_metrics="$(find "$run_dpath/checkpoint_select/$best_candidate/vali" -name detect_metrics.json | head -n 1)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\t%s\n' \
        "$best_candidate" "$train_size" "$best_poop_vali" "$(read_metric "$best_vali_metrics" nocls_ap)" \
        "$poop_test_ap" "$nocls_test_ap" "$best_ckpt" "$run_dpath" "$summary_fpath" >> "$summary_fpath"
done
