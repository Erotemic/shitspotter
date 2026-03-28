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

OPEN_GDINO_REPO_DPATH="${OPEN_GDINO_REPO_DPATH:-$REPO_DPATH/tpl/Open-GroundingDino}"
GPU_NUM="${GPU_NUM:-1}"
PRETRAIN_MODEL_PATH="${PRETRAIN_MODEL_PATH:-groundingdino_swint_ogc.pth}"
TEXT_ENCODER_TYPE="${TEXT_ENCODER_TYPE:-bert-base-uncased}"
CFG_TEMPLATE="${CFG_TEMPLATE:-config/cfg_odvg.py}"
FORCE_GDINO_RERUN="${FORCE_GDINO_RERUN:-False}"
CLASSES_TEXT="${CLASSES_TEXT:-[poop]}"
# Format: config_tag|gpu_num|pretrain_model|text_encoder|cfg_template[|batch_size|lr_scale|backbone_lr_scale]
# The last three fields are optional. If omitted, the config template defaults are used.
# When batch_size is provided, lr and lr_backbone are scaled linearly from their
# base values (lr=0.0001 at batch=4, lr_backbone=1e-5 at batch=4) then multiplied
# by lr_scale and backbone_lr_scale respectively.
GDINO_CONFIG_SPECS="${GDINO_CONFIG_SPECS:-baseline|1|groundingdino_swint_ogc.pth|bert-base-uncased|config/cfg_odvg.py}"

# OpenGroundingDINO base training hyperparameters (from cfg_odvg.py defaults).
# Used to compute scaled LR values when batch_size is overridden.
GDINO_BASE_BATCH_SIZE="${GDINO_BASE_BATCH_SIZE:-4}"
GDINO_BASE_LR="${GDINO_BASE_LR:-0.0001}"
GDINO_BASE_LR_BACKBONE="${GDINO_BASE_LR_BACKBONE:-1e-05}"

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

# Optional: space-separated list of train sizes to run (e.g. "128" or "128 256").
# If empty, all sizes from the benchmark manifest are used.
GDINO_TRAIN_SIZES="${GDINO_TRAIN_SIZES:-}"

for row in "${TRAIN_ROWS[@]}"; do
    IFS=$'\t' read -r subset_name train_size train_kwcoco train_mscoco <<<"$row"
    if [ -n "$GDINO_TRAIN_SIZES" ]; then
        _skip=1
        for _sz in $GDINO_TRAIN_SIZES; do
            if [ "$train_size" = "$_sz" ]; then _skip=0; break; fi
        done
        if [ "$_skip" = "1" ]; then continue; fi
    fi
    for config_spec in ${GDINO_CONFIG_SPECS}; do
        # Parse config spec: first 5 fields are the original format, last 4 are optional tuning knobs
        # Format: config_tag|gpu_num|pretrain_model|text_encoder|cfg_template[|batch_size|lr_scale|backbone_lr_scale|epochs]
        IFS='|' read -r config_tag config_gpu_num config_pretrain_model_path config_text_encoder_type config_cfg_template config_batch_size config_lr_scale config_backbone_lr_scale config_epochs <<<"$config_spec"
        config_tag="${config_tag:-baseline}"
        config_gpu_num="${config_gpu_num:-$GPU_NUM}"
        config_pretrain_model_path="${config_pretrain_model_path:-$PRETRAIN_MODEL_PATH}"
        config_text_encoder_type="${config_text_encoder_type:-$TEXT_ENCODER_TYPE}"
        config_cfg_template="${config_cfg_template:-$CFG_TEMPLATE}"
        # Optional tuning fields: empty string means "use config template defaults"
        config_batch_size="${config_batch_size:-}"
        config_lr_scale="${config_lr_scale:-}"
        config_backbone_lr_scale="${config_backbone_lr_scale:-}"
        config_epochs="${config_epochs:-}"

        # Resolve effective LR values when batch_size is overridden
        resolved_lr=""
        resolved_backbone_lr=""
        if [ -n "$config_batch_size" ]; then
            local_lr_scale="${config_lr_scale:-1.0}"
            local_backbone_lr_scale="${config_backbone_lr_scale:-1.0}"
            read -r resolved_lr resolved_backbone_lr < <("$PYTHON_BIN" - <<PY
base_batch = float($GDINO_BASE_BATCH_SIZE)
base_lr = float($GDINO_BASE_LR)
base_backbone_lr = float($GDINO_BASE_LR_BACKBONE)
batch = float($config_batch_size)
lr_scale = float($local_lr_scale)
backbone_lr_scale = float($local_backbone_lr_scale)
resolved_lr = base_lr * (batch / base_batch) * lr_scale
resolved_backbone_lr = base_backbone_lr * (batch / base_batch) * backbone_lr_scale
print(f'{resolved_lr:.10g} {resolved_backbone_lr:.10g}')
PY
)
        fi

        run_dpath="$RUN_ROOT/$config_tag/$subset_name"
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
        printf '  %-22s %s\n' "CONFIG_TAG" "$config_tag"
        printf '  %-22s %s\n' "SUBSET" "$subset_name"
        printf '  %-22s %s\n' "TRAIN_SIZE" "$train_size"
        printf '  %-22s %s\n' "RUN_DPATH" "$run_dpath"
        if [ -n "$config_batch_size" ]; then
            printf '  %-22s %s\n' "BATCH_SIZE" "$config_batch_size"
            printf '  %-22s %s\n' "RESOLVED_LR" "$resolved_lr"
            printf '  %-22s %s\n' "RESOLVED_LR_BACKBONE" "$resolved_backbone_lr"
        fi
        if [ -n "$config_epochs" ]; then
            printf '  %-22s %s\n' "EPOCHS" "$config_epochs"
        fi

        # Build run manifest with optional tuning fields
        "$PYTHON_BIN" - "$run_manifest_fpath" <<PY
import json, sys
manifest = {
    "model_family": "opengroundingdino",
    "config_tag": "$config_tag",
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
    "gpu_num": $config_gpu_num,
    "cfg_template": "$config_cfg_template",
    "pretrain_model_path": "$config_pretrain_model_path",
    "text_encoder_type": "$config_text_encoder_type",
    "classes_text": "$CLASSES_TEXT",
}
# Include tuning overrides when batch_size is explicitly set
batch_size_str = "$config_batch_size"
if batch_size_str:
    manifest["train_batch_size"] = int(batch_size_str)
    manifest["lr_scale"] = float("${config_lr_scale:-1.0}")
    manifest["backbone_lr_scale"] = float("${config_backbone_lr_scale:-1.0}")
    manifest["resolved_lr"] = float("$resolved_lr")
    manifest["resolved_backbone_lr"] = float("$resolved_backbone_lr")
    manifest["base_batch_size"] = int("$GDINO_BASE_BATCH_SIZE")
    manifest["base_lr"] = float("$GDINO_BASE_LR")
    manifest["base_lr_backbone"] = float("$GDINO_BASE_LR_BACKBONE")
else:
    manifest["train_batch_size"] = int("$GDINO_BASE_BATCH_SIZE")
epochs_str = "$config_epochs"
if epochs_str:
    manifest["epochs"] = int(epochs_str)
with open(sys.argv[1], 'w') as f:
    json.dump(manifest, f, indent=2)
    f.write('\n')
PY

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

        cp "$config_cfg_template" "$cfg_fpath"
        sed -i 's|use_coco_eval = True|use_coco_eval = False|g' "$cfg_fpath"
        echo "" >> "$cfg_fpath"
        echo "label_list = ['poop']" >> "$cfg_fpath"

        # Apply batch_size, LR, and epoch overrides when tuning fields are specified.
        # Appending to the Python config file overrides earlier definitions.
        if [ -n "$config_batch_size" ] || [ -n "$config_epochs" ]; then
            {
                echo ""
                echo "# --- Hyperparameter overrides (added by benchmark runner) ---"
                [ -n "$config_batch_size" ] && echo "batch_size = $config_batch_size"
                [ -n "$config_batch_size" ] && echo "lr = $resolved_lr"
                [ -n "$config_batch_size" ] && echo "lr_backbone = $resolved_backbone_lr"
                [ -n "$config_epochs" ] && echo "epochs = $config_epochs"
            } >> "$cfg_fpath"
        fi

        export GPU_NUM="$config_gpu_num"
        export CFG="$cfg_fpath"
        export DATASETS="$datasets_json"
        export OUTPUT_DIR="$output_dir"
        export PRETRAIN_MODEL_PATH="$config_pretrain_model_path"
        export TEXT_ENCODER_TYPE="$config_text_encoder_type"
        export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"

        if [ -d "$output_dir" ] && [ ! -f "$summary_fpath" ]; then
            rm -rf "$output_dir"
        fi

        case "$FORCE_GDINO_RERUN" in
            1|true|True|TRUE|yes|Yes|YES|on|On|ON)
                rm -rf "$output_dir" "$run_dpath/checkpoint_select" "$run_dpath/test_eval" "$summary_fpath"
                ;;
        esac

        # Compute expected final checkpoint filename (0-indexed epoch number)
        _gdino_epochs="${config_epochs:-15}"
        _gdino_final_ckpt_idx=$(( _gdino_epochs - 1 ))
        _gdino_final_ckpt="$(printf 'checkpoint%04d.pth' "$_gdino_final_ckpt_idx")"

        if [ ! -f "$output_dir/$_gdino_final_ckpt" ] || [ "$FORCE_GDINO_RERUN" = "True" ]; then
            # Wrap training in kwutil.ProcessContext for timing/environment telemetry.
            # Falls back to plain training if kwutil is not available.
            "$PYTHON_BIN" - "$run_manifest_fpath" "$output_dir" "$GPU_NUM" "$CFG" "$DATASETS" <<'PROCWRAP' || true
import subprocess
import sys
import json
import pathlib

run_manifest_fpath = sys.argv[1]
output_dir = sys.argv[2]
gpu_num, cfg, datasets = sys.argv[3], sys.argv[4], sys.argv[5]
output_dpath = pathlib.Path(output_dir)
output_dpath.mkdir(parents=True, exist_ok=True)

try:
    import kwutil
    manifest = json.loads(pathlib.Path(run_manifest_fpath).read_text())
    proc_context = kwutil.ProcessContext(
        name='opengroundingdino.train',
        config=manifest,
    )
    proc_context.start()
    initial_fpath = output_dpath / 'initial_telemetry.json'
    initial_fpath.write_text(kwutil.Json.dumps(proc_context.obj))
except ImportError:
    proc_context = None

ret = subprocess.call(['bash', 'train_dist.sh', gpu_num, cfg, datasets, output_dir])

if proc_context is not None:
    proc_context.stop()
    final_fpath = output_dpath / 'final_telemetry.json'
    final_fpath.write_text(kwutil.Json.dumps(proc_context.obj))

sys.exit(ret)
PROCWRAP
            # If the Python wrapper itself fails (e.g. syntax error), fall back
            if [ ! -f "$output_dir/$_gdino_final_ckpt" ]; then
                bash train_dist.sh "$GPU_NUM" "$CFG" "$DATASETS" "$OUTPUT_DIR"
            fi
        fi

        mapfile -t CANDIDATES < <(find "$output_dir" -maxdepth 1 -type f -name 'checkpoint*.pth' | sort)
        if [ "${#CANDIDATES[@]}" -eq 0 ]; then
            echo "No OpenGroundingDINO checkpoints found in $output_dir" >&2
            exit 1
        fi

        printf 'candidate_id\ttrain_size\tpoop_vali_ap\tnocls_vali_ap\tpoop_test_ap\tnocls_test_ap\tselected\tckpt_fpath\trun_dpath\tsummary_fpath\tconfig_tag\n' > "$summary_fpath"
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
            printf '%s\t%s\t%s\t%s\tNA\tNA\t0\t%s\t%s\t%s\t%s\n' \
                "$candidate_id" "$train_size" "$poop_vali_ap" "$nocls_vali_ap" "$ckpt_fpath" "$run_dpath" "$summary_fpath" "$config_tag" >> "$summary_fpath"
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
        printf '%s\t%s\t%s\t%s\t%s\t%s\t1\t%s\t%s\t%s\t%s\n' \
            "$best_candidate" "$train_size" "$best_poop_vali" "$(read_metric "$best_vali_metrics" nocls_ap)" \
            "$poop_test_ap" "$nocls_test_ap" "$best_ckpt" "$run_dpath" "$summary_fpath" "$config_tag" >> "$summary_fpath"
    done
done
