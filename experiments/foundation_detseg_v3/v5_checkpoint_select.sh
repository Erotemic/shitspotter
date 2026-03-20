#!/bin/bash
set -euo pipefail

FOUNDATION_V3_DEV_DPATH="${FOUNDATION_V3_DEV_DPATH:-${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}/experiments/foundation_detseg_v3}"
_foundation_v3_source="${BASH_SOURCE[0]-}"
if [ -n "$_foundation_v3_source" ] && [ "$_foundation_v3_source" != "bash" ] && [ "$_foundation_v3_source" != "-bash" ]; then
    _foundation_v3_script_dpath="$(cd "$(dirname "$_foundation_v3_source")" && pwd)"
else
    _foundation_v3_script_dpath="$FOUNDATION_V3_DEV_DPATH"
fi
# shellcheck source=experiments/foundation_detseg_v3/common.sh
source "$_foundation_v3_script_dpath/common.sh"
unset _foundation_v3_source
unset _foundation_v3_script_dpath

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

PYTHON_BIN="${PYTHON_BIN:-python}"

have_metrics() {
    local dpath="$1"
    [ -f "$dpath/eval/detect_metrics.json" ]
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

append_summary_row() {
    local summary_fpath="$1"
    local phase="$2"
    local version="$3"
    local candidate_id="$4"
    local ckpt_fpath="$5"
    local approx_epoch="$6"
    local internal_epoch="$7"
    local internal_ap="$8"
    local score_thresh="$9"
    local nms_thresh="${10}"
    local detector_vali_ap="${11}"
    local combined_vali_ap="${12}"
    local combined_test_ap="${13}"
    local package_fpath="${14}"
    local detector_eval_dpath="${15}"
    local combined_vali_dpath="${16}"
    local combined_test_dpath="${17}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" \
        "$version" \
        "$candidate_id" \
        "$ckpt_fpath" \
        "$approx_epoch" \
        "$internal_epoch" \
        "$internal_ap" \
        "$score_thresh" \
        "$nms_thresh" \
        "$detector_vali_ap" \
        "$combined_vali_ap" \
        "$combined_test_ap" \
        "$package_fpath" \
        "$detector_eval_dpath" \
        "$combined_vali_dpath" \
        "$combined_test_dpath" \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        "$phase" >> "$summary_fpath"
}

print_top_rows() {
    local summary_fpath="$1"
    local limit="${2:-20}"
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
        for key in ['detector_vali_ap', 'combined_vali_ap', 'combined_test_ap']:
            try:
                row[f'_{key}'] = float(row[key]) if row[key] not in {'', 'NA'} else float('-inf')
            except Exception:
                row[f'_{key}'] = float('-inf')
        rows.append(row)
rows.sort(key=lambda r: (r['_combined_test_ap'], r['_combined_vali_ap'], r['_detector_vali_ap']), reverse=True)
print('phase\tversion\tcandidate_id\tdet_vali\tcomb_vali\tcomb_test\tscore\tnms\tapprox_epoch\tinternal_epoch\tinternal_ap')
for row in rows[:limit]:
    print(
        f"{row['phase']}\t{row['version']}\t{row['candidate_id']}\t"
        f"{row['detector_vali_ap']}\t{row['combined_vali_ap']}\t{row['combined_test_ap']}\t"
        f"{row['score_thresh']}\t{row['nms_thresh']}\t{row['approx_epoch']}\t"
        f"{row['internal_epoch']}\t{row['internal_ap']}"
    )
PY
}

emit_candidate_rows() {
    local version="$1"
    local workdir="$2"
    local log_fpath="$3"
    local candidate_mode="$4"
    local top_internal_epochs="$5"
    "$PYTHON_BIN" - "$version" "$workdir" "$log_fpath" "$candidate_mode" "$top_internal_epochs" <<'PY'
import json
import pathlib
import re
import sys

version = sys.argv[1]
workdir = pathlib.Path(sys.argv[2])
log_fpath = pathlib.Path(sys.argv[3])
candidate_mode = sys.argv[4]
top_internal_epochs = int(sys.argv[5])

periodic = []
for ckpt in sorted(workdir.glob('checkpoint*.pth')):
    match = re.search(r'checkpoint0*(\d+)\.pth$', ckpt.name)
    if match:
        periodic.append((int(match.group(1)), ckpt))

internal_rows = []
internal_lookup = {}
for line in log_fpath.read_text().splitlines():
    line = line.strip()
    if not line.startswith('{'):
        continue
    try:
        row = json.loads(line)
    except Exception:
        continue
    vals = row.get('test_coco_eval_bbox', None)
    if vals:
        epoch = int(row['epoch'])
        ap = float(vals[0])
        internal_rows.append((epoch, ap))
        internal_lookup[epoch] = ap
internal_rows.sort(key=lambda t: t[1], reverse=True)

chosen = {}

def add_candidate(candidate_id, ckpt_path, approx_epoch, internal_epoch, internal_ap, reason):
    chosen[candidate_id] = {
        'candidate_id': candidate_id,
        'ckpt_path': str(ckpt_path),
        'approx_epoch': approx_epoch,
        'internal_epoch': internal_epoch,
        'internal_ap': internal_ap,
        'reason': reason,
    }

best_stg1 = workdir / 'best_stg1.pth'
last = workdir / 'last.pth'
if best_stg1.exists():
    add_candidate('best_stg1', best_stg1, 'NA', 'NA', 'NA', 'special')
if last.exists():
    final_epoch = max((epoch for epoch, _ in internal_rows), default='NA')
    final_ap = next((ap for epoch, ap in internal_rows if epoch == final_epoch), 'NA')
    add_candidate('last', last, str(final_epoch), str(final_epoch), f'{final_ap:.6f}' if final_ap != 'NA' else 'NA', 'special')

if candidate_mode == 'all':
    for periodic_epoch, periodic_ckpt in periodic:
        exact_ap = internal_lookup.get(periodic_epoch, None)
        add_candidate(
            periodic_ckpt.stem,
            periodic_ckpt,
            str(periodic_epoch),
            str(periodic_epoch),
            f'{exact_ap:.6f}' if exact_ap is not None else 'NA',
            'periodic_all',
        )
elif candidate_mode == 'internal_top':
    for internal_epoch, internal_ap in internal_rows[:top_internal_epochs]:
        if not periodic:
            continue
        nearest_epoch, nearest_ckpt = min(periodic, key=lambda item: abs(item[0] - internal_epoch))
        candidate_id = nearest_ckpt.stem
        add_candidate(candidate_id, nearest_ckpt, str(nearest_epoch), str(internal_epoch), f'{internal_ap:.6f}', 'nearest_internal_top')
else:
    raise KeyError(f'Unknown candidate_mode={candidate_mode!r}')

rows = sorted(chosen.values(), key=lambda row: (row['approx_epoch'] == 'NA', int(row['approx_epoch']) if row['approx_epoch'] != 'NA' else 10**9, row['candidate_id']))
for row in rows:
    print('\t'.join([
        row['candidate_id'],
        row['ckpt_path'],
        row['approx_epoch'],
        row['internal_epoch'],
        row['internal_ap'],
        row['reason'],
    ]))
PY
}

build_candidate_package() {
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

evaluate_detector_vali() {
    local package_fpath="$1"
    local out_dpath="$2"
    local score_thresh="$3"
    local nms_thresh="$4"
    mkdir -p "$out_dpath/eval"
    "$PYTHON_BIN" -m shitspotter.algo_foundation_v3.cli_predict_boxes \
        "$VALI_FPATH" \
        --package_fpath "$package_fpath" \
        --dst "$out_dpath/pred_boxes.kwcoco.zip" \
        --score_thresh "$score_thresh" \
        --nms_thresh "$nms_thresh"
    "$PYTHON_BIN" -m kwcoco eval \
        --true_dataset "$VALI_FPATH" \
        --pred_dataset "$out_dpath/pred_boxes.kwcoco.zip" \
        --out_dpath "$out_dpath/eval" \
        --out_fpath "$out_dpath/eval/detect_metrics.json" \
        --confusion_fpath "$out_dpath/eval/confusion.kwcoco.zip" \
        --draw False \
        --iou_thresh 0.5 >/dev/null
}

evaluate_combined_raw() {
    local package_fpath="$1"
    local split="$2"
    local out_dpath="$3"
    local src_fpath
    if [ "$split" = "vali" ]; then
        src_fpath="$VALI_FPATH"
    elif [ "$split" = "test" ]; then
        src_fpath="$TEST_FPATH"
    else
        echo "Unknown split=$split" >&2
        exit 1
    fi
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

select_top_candidates() {
    local summary_fpath="$1"
    local version="$2"
    local score_thresh="$3"
    local topk="$4"
    "$PYTHON_BIN" - "$summary_fpath" "$version" "$score_thresh" "$topk" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
version = sys.argv[2]
score_thresh = sys.argv[3]
topk = int(sys.argv[4])
rows = []
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        if row['phase'] != 'detector':
            continue
        if row['version'] != version:
            continue
        if row['score_thresh'] != score_thresh:
            continue
        try:
            row['_ap'] = float(row['detector_vali_ap'])
        except Exception:
            continue
        rows.append(row)
rows.sort(key=lambda r: r['_ap'], reverse=True)
seen = set()
for row in rows:
    candidate_id = row['candidate_id']
    if candidate_id in seen:
        continue
    seen.add(candidate_id)
    print(candidate_id)
    if len(seen) >= topk:
        break
PY
}

REPO_DPATH="$(canonical_existing_path "${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}")"
DATA_DPATH="$(canonical_existing_path "${DVC_DATA_DPATH:?Set DVC_DATA_DPATH or install geowatch_dvc}")"
EXPT_DPATH="$(canonical_existing_path "${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}")"

VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip"
SAM2_INIT_CKPT="$REPO_DPATH/tpl/segment-anything-2/checkpoints/sam2.1_hiera_base_plus.pt"

VERSIONS=(${VERSIONS:-v5 v3})
CANDIDATE_MODE="${CANDIDATE_MODE:-all}"
TOP_INTERNAL_EPOCHS="${TOP_INTERNAL_EPOCHS:-12}"
DETECTOR_SCORE_THRESHES=(${DETECTOR_SCORE_THRESHES:-0.2 0.0})
DETECTOR_NMS_THRESH="${DETECTOR_NMS_THRESH:-0.5}"
SELECT_SCORE_THRESH="${SELECT_SCORE_THRESH:-${DETECTOR_SCORE_THRESHES[0]}}"
TOP_COMBINED_K="${TOP_COMBINED_K:-2}"
RUN_COMBINED_TOPK="${RUN_COMBINED_TOPK:-True}"
DEFAULT_SWEEP_ROOT="$EXPT_DPATH/foundation_detseg_v3/checkpoint_select"
FALLBACK_SWEEP_ROOT="$REPO_DPATH/experiments/foundation_detseg_v3/_checkpoint_select"
if [ -n "${SWEEP_ROOT:-}" ]; then
    SWEEP_ROOT="$SWEEP_ROOT"
elif mkdir -p "$DEFAULT_SWEEP_ROOT" 2>/dev/null; then
    SWEEP_ROOT="$DEFAULT_SWEEP_ROOT"
else
    SWEEP_ROOT="$FALLBACK_SWEEP_ROOT"
fi
SUMMARY_FPATH="$SWEEP_ROOT/summary.tsv"

require_path "$REPO_DPATH"
require_path "$VALI_FPATH"
require_path "$TEST_FPATH"
mkdir -p "$SWEEP_ROOT/packages" "$SWEEP_ROOT/evals"

if [ ! -f "$SUMMARY_FPATH" ]; then
    printf 'phase\tversion\tcandidate_id\tckpt_fpath\tapprox_epoch\tinternal_epoch\tinternal_ap\tscore_thresh\tnms_thresh\tdetector_vali_ap\tcombined_vali_ap\tcombined_test_ap\tpackage_fpath\tdetector_eval_dpath\tcombined_vali_dpath\tcombined_test_dpath\ttimestamp\tphase_copy\n' > "$SUMMARY_FPATH"
fi

echo "foundation_v3 checkpoint selection"
printf '  %-24s %s\n' "VERSIONS" "${VERSIONS[*]}"
printf '  %-24s %s\n' "CANDIDATE_MODE" "$CANDIDATE_MODE"
printf '  %-24s %s\n' "TOP_INTERNAL_EPOCHS" "$TOP_INTERNAL_EPOCHS"
printf '  %-24s %s\n' "DETECTOR_SCORE_THRESHES" "${DETECTOR_SCORE_THRESHES[*]}"
printf '  %-24s %s\n' "DETECTOR_NMS_THRESH" "$DETECTOR_NMS_THRESH"
printf '  %-24s %s\n' "SELECT_SCORE_THRESH" "$SELECT_SCORE_THRESH"
printf '  %-24s %s\n' "TOP_COMBINED_K" "$TOP_COMBINED_K"
printf '  %-24s %s\n' "RUN_COMBINED_TOPK" "$RUN_COMBINED_TOPK"
printf '  %-24s %s\n' "SWEEP_ROOT" "$SWEEP_ROOT"

echo
echo "=== Detector checkpoint sweep ==="
for version in "${VERSIONS[@]}"; do
    root="$EXPT_DPATH/foundation_detseg_v3/$version"
    detector_workdir="$root/train_detector_deimv2_m"
    segmenter_ckpt="$root/train_segmenter_sam2_1_hiera_base_plus/checkpoints/checkpoint.pt"
    if [ ! -f "$segmenter_ckpt" ]; then
        segmenter_ckpt="$SAM2_INIT_CKPT"
    fi
    require_path "$detector_workdir"
    require_path "$detector_workdir/log.txt"
    require_path "$segmenter_ckpt"
    echo
    echo "--- $version candidates ---"
    while IFS=$'\t' read -r candidate_id ckpt_fpath approx_epoch internal_epoch internal_ap reason; do
        [ -n "$candidate_id" ] || continue
        require_path "$ckpt_fpath"
        package_fpath="$SWEEP_ROOT/packages/${version}_${candidate_id}.yaml"
        metadata_name="${version}_${candidate_id}"
        build_candidate_package "$package_fpath" "$ckpt_fpath" "$segmenter_ckpt" "$metadata_name"
        for score_thresh in "${DETECTOR_SCORE_THRESHES[@]}"; do
            detector_eval_dpath="$SWEEP_ROOT/evals/${version}_${candidate_id}/boxes_s${score_thresh}_n${DETECTOR_NMS_THRESH}"
            if have_metrics "$detector_eval_dpath"; then
                echo "Reusing detector metrics for $version $candidate_id score=$score_thresh"
            else
                evaluate_detector_vali "$package_fpath" "$detector_eval_dpath" "$score_thresh" "$DETECTOR_NMS_THRESH"
            fi
            detector_vali_ap="$(read_ap "$detector_eval_dpath/eval/detect_metrics.json")"
            append_summary_row "$SUMMARY_FPATH" "detector" "$version" "$candidate_id" \
                "$ckpt_fpath" "$approx_epoch" "$internal_epoch" "$internal_ap" \
                "$score_thresh" "$DETECTOR_NMS_THRESH" "$detector_vali_ap" "NA" "NA" \
                "$package_fpath" "$detector_eval_dpath" "NA" "NA"
            printf '  %-6s %-16s score=%-4s internal=%-8s det_vali_ap=%s\n' \
                "$version" "$candidate_id" "$score_thresh" "$internal_ap" "$detector_vali_ap"
        done
    done < <(emit_candidate_rows "$version" "$detector_workdir" "$detector_workdir/log.txt" "$CANDIDATE_MODE" "$TOP_INTERNAL_EPOCHS")
done

echo
echo "Top detector-only results:"
print_top_rows "$SUMMARY_FPATH" 24

if [ "${RUN_COMBINED_TOPK,,}" = "true" ] || [ "${RUN_COMBINED_TOPK,,}" = "yes" ] || [ "${RUN_COMBINED_TOPK}" = "1" ]; then
    echo
    echo "=== Combined raw eval for top detector checkpoints ==="
    for version in "${VERSIONS[@]}"; do
        mapfile -t top_candidates < <(select_top_candidates "$SUMMARY_FPATH" "$version" "$SELECT_SCORE_THRESH" "$TOP_COMBINED_K")
        for candidate_id in "${top_candidates[@]}"; do
            [ -n "$candidate_id" ] || continue
            package_fpath="$SWEEP_ROOT/packages/${version}_${candidate_id}.yaml"
            row="$("$PYTHON_BIN" - "$SUMMARY_FPATH" "$version" "$candidate_id" "$SELECT_SCORE_THRESH" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
version = sys.argv[2]
candidate_id = sys.argv[3]
score_thresh = sys.argv[4]
best = None
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        if row['phase'] != 'detector':
            continue
        if row['version'] != version or row['candidate_id'] != candidate_id or row['score_thresh'] != score_thresh:
            continue
        try:
            ap = float(row['detector_vali_ap'])
        except Exception:
            ap = float('-inf')
        if best is None or ap > best[0]:
            best = (ap, row)
if best is None:
    raise SystemExit('No matching detector row found')
row = best[1]
print('\t'.join([row['ckpt_fpath'], row['approx_epoch'], row['internal_epoch'], row['internal_ap'], row['detector_vali_ap']]))
PY
)"
            IFS=$'\t' read -r ckpt_fpath approx_epoch internal_epoch internal_ap detector_vali_ap <<< "$row"
            combined_vali_dpath="$SWEEP_ROOT/evals/${version}_${candidate_id}/combined_raw_vali"
            combined_test_dpath="$SWEEP_ROOT/evals/${version}_${candidate_id}/combined_raw_test"
            if have_metrics "$combined_vali_dpath"; then
                echo "Reusing combined validation metrics for $version $candidate_id"
            else
                evaluate_combined_raw "$package_fpath" "vali" "$combined_vali_dpath"
            fi
            if have_metrics "$combined_test_dpath"; then
                echo "Reusing combined test metrics for $version $candidate_id"
            else
                evaluate_combined_raw "$package_fpath" "test" "$combined_test_dpath"
            fi
            combined_vali_ap="$(read_ap "$combined_vali_dpath/eval/detect_metrics.json")"
            combined_test_ap="$(read_ap "$combined_test_dpath/eval/detect_metrics.json")"
            append_summary_row "$SUMMARY_FPATH" "combined_raw" "$version" "$candidate_id" \
                "$ckpt_fpath" "$approx_epoch" "$internal_epoch" "$internal_ap" \
                "$SELECT_SCORE_THRESH" "$DETECTOR_NMS_THRESH" "$detector_vali_ap" "$combined_vali_ap" "$combined_test_ap" \
                "$package_fpath" "NA" "$combined_vali_dpath" "$combined_test_dpath"
            printf '  %-6s %-16s det_vali=%s comb_vali=%s comb_test=%s\n' \
                "$version" "$candidate_id" "$detector_vali_ap" "$combined_vali_ap" "$combined_test_ap"
        done
    done
fi

echo
echo "=== Final summary ==="
print_top_rows "$SUMMARY_FPATH" 40
printf '  %-24s %s\n' "SUMMARY_FPATH" "$SUMMARY_FPATH"
