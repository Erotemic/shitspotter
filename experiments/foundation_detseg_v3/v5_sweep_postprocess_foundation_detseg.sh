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

have_metrics() {
    local dpath="$1"
    [ -f "$dpath/eval/detect_metrics.json" ]
}

read_ap() {
    local metrics_fpath="$1"
    python - "$metrics_fpath" <<'PY'
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

write_variant_package() {
    local src_fpath="$1"
    local dst_fpath="$2"
    local metadata_name="$3"
    local score_thresh="$4"
    local nms_thresh="$5"
    local crop_padding="$6"
    local polygon_simplify="$7"
    local min_component_area="$8"
    local keep_largest_component="$9"
    python - "$src_fpath" "$dst_fpath" "$metadata_name" "$score_thresh" "$nms_thresh" \
        "$crop_padding" "$polygon_simplify" "$min_component_area" "$keep_largest_component" <<'PY'
import pathlib
import sys
import yaml

src_fpath = pathlib.Path(sys.argv[1])
dst_fpath = pathlib.Path(sys.argv[2])
metadata_name = sys.argv[3]
score_thresh = float(sys.argv[4])
nms_thresh = float(sys.argv[5])
crop_padding = int(sys.argv[6])
polygon_simplify = float(sys.argv[7])
min_component_area = int(sys.argv[8])
keep_largest_component = sys.argv[9].lower() in {'1', 'true', 'yes', 'on'}

data = yaml.safe_load(src_fpath.read_text()) or {}
post = data.setdefault('postprocess', {})
post['score_thresh'] = score_thresh
post['nms_thresh'] = nms_thresh
post['crop_padding'] = crop_padding
post['polygon_simplify'] = polygon_simplify
post['min_component_area'] = min_component_area
post['keep_largest_component'] = keep_largest_component
meta = data.setdefault('metadata', {})
meta['name'] = metadata_name
dst_fpath.parent.mkdir(parents=True, exist_ok=True)
dst_fpath.write_text(yaml.safe_dump(data, sort_keys=False))
PY
}

append_summary_row() {
    local summary_fpath="$1"
    local phase="$2"
    local variant_id="$3"
    local score_thresh="$4"
    local nms_thresh="$5"
    local crop_padding="$6"
    local polygon_simplify="$7"
    local min_component_area="$8"
    local keep_largest_component="$9"
    local vali_ap="${10}"
    local test_ap="${11}"
    local package_fpath="${12}"
    local vali_eval_dpath="${13}"
    local test_eval_dpath="${14}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$phase" \
        "$variant_id" \
        "$score_thresh" \
        "$nms_thresh" \
        "$crop_padding" \
        "$polygon_simplify" \
        "$min_component_area" \
        "$keep_largest_component" \
        "$vali_ap" \
        "$test_ap" \
        "$package_fpath" \
        "$vali_eval_dpath" \
        "$test_eval_dpath" \
        "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$summary_fpath"
}

print_top_rows() {
    local summary_fpath="$1"
    local limit="${2:-10}"
    python - "$summary_fpath" "$limit" <<'PY'
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
            row['_vali_ap'] = float(row['vali_ap'])
        except Exception:
            row['_vali_ap'] = float('-inf')
        try:
            row['_test_ap'] = float(row['test_ap']) if row['test_ap'] not in {'', 'NA'} else float('-inf')
        except Exception:
            row['_test_ap'] = float('-inf')
        rows.append(row)

rows.sort(key=lambda r: (r['_vali_ap'], r['_test_ap']), reverse=True)
print('phase\tvariant_id\tvali_ap\ttest_ap\tscore\tnms\tcrop\tpoly\tmin_area\tkeep_largest')
for row in rows[:limit]:
    print(
        f"{row['phase']}\t{row['variant_id']}\t{row['vali_ap']}\t{row['test_ap']}\t"
        f"{row['score_thresh']}\t{row['nms_thresh']}\t{row['crop_padding']}\t"
        f"{row['polygon_simplify']}\t{row['min_component_area']}\t{row['keep_largest_component']}"
    )
PY
}

select_top_variant_ids() {
    local summary_fpath="$1"
    local phase="$2"
    local limit="$3"
    python - "$summary_fpath" "$phase" "$limit" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
phase = sys.argv[2]
limit = int(sys.argv[3])
rows = []
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        if row['phase'] != phase:
            continue
        try:
            row['_vali_ap'] = float(row['vali_ap'])
        except Exception:
            continue
        rows.append(row)

rows.sort(key=lambda r: r['_vali_ap'], reverse=True)
seen = set()
chosen = []
for row in rows:
    variant_id = row['variant_id']
    if variant_id in seen:
        continue
    seen.add(variant_id)
    chosen.append(variant_id)
    if len(chosen) >= limit:
        break
print('\n'.join(chosen))
PY
}

lookup_variant_row() {
    local summary_fpath="$1"
    local variant_id="$2"
    python - "$summary_fpath" "$variant_id" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
variant_id = sys.argv[2]
best = None
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        if row['variant_id'] != variant_id:
            continue
        try:
            score = float(row['vali_ap'])
        except Exception:
            score = float('-inf')
        if best is None or score > best[0]:
            best = (score, row)
if best is None:
    raise SystemExit(f'Unknown variant_id={variant_id!r}')
row = best[1]
print('\t'.join([
    row['score_thresh'],
    row['nms_thresh'],
    row['crop_padding'],
    row['polygon_simplify'],
    row['min_component_area'],
    row['keep_largest_component'],
]))
PY
}

evaluate_variant() {
    local package_fpath="$1"
    local out_dpath="$2"
    local split="$3"
    if [ "$split" = "vali" ]; then
        export VALI_FPATH
        export EVAL_PATH="$out_dpath"
        export PACKAGE_FPATH="$package_fpath"
        bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_vali.sh"
    elif [ "$split" = "test" ]; then
        export TEST_FPATH
        export EVAL_PATH="$out_dpath"
        export PACKAGE_FPATH="$package_fpath"
        bash "$REPO_DPATH/experiments/foundation_detseg_v3/run_deimv2_sam2_on_test.sh"
    else
        echo "Unknown split=$split" >&2
        exit 1
    fi
}

REPO_DPATH="$(canonical_existing_path "${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}")"
DATA_DPATH="$(canonical_existing_path "${DVC_DATA_DPATH:?Set DVC_DATA_DPATH or install geowatch_dvc}")"
EXPT_DPATH="$(canonical_existing_path "${DVC_EXPT_DPATH:?Set DVC_EXPT_DPATH or install geowatch_dvc}")"

VALI_FPATH="$DATA_DPATH/vali_imgs691_99b22ad0.kwcoco.zip"
TEST_FPATH="$DATA_DPATH/test_imgs121_6cb3b6ff.kwcoco.zip"
BASE_PACKAGE_FPATH="${BASE_PACKAGE_FPATH:-$REPO_DPATH/experiments/foundation_detseg_v3/packages/v5_deimv2_m_sam2_1_hiera_base_plus_tuned.yaml}"

SWEEP_ROOT="${SWEEP_ROOT:-$EXPT_DPATH/foundation_detseg_v3/v5_postprocess_sweep}"
VARIANT_DPATH="$SWEEP_ROOT/packages"
EVAL_DPATH="$SWEEP_ROOT/evals"
SUMMARY_FPATH="$SWEEP_ROOT/summary.tsv"

SCORE_THRESH_CANDIDATES=(${SCORE_THRESH_CANDIDATES:-0.20 0.25 0.30 0.35 0.40})
NMS_THRESH_CANDIDATES=(${NMS_THRESH_CANDIDATES:-0.40 0.50 0.60})
BASE_CROP_PADDING="${BASE_CROP_PADDING:-0}"
BASE_POLYGON_SIMPLIFY="${BASE_POLYGON_SIMPLIFY:-0}"
BASE_MIN_COMPONENT_AREA="${BASE_MIN_COMPONENT_AREA:-0}"
BASE_KEEP_LARGEST_COMPONENT="${BASE_KEEP_LARGEST_COMPONENT:-false}"
MIN_COMPONENT_AREA_CANDIDATES=(${MIN_COMPONENT_AREA_CANDIDATES:-0 16 32})
KEEP_LARGEST_COMPONENT_CANDIDATES=(${KEEP_LARGEST_COMPONENT_CANDIDATES:-false true})
CROP_PADDING_CANDIDATES=(${CROP_PADDING_CANDIDATES:-0})
POLYGON_SIMPLIFY_CANDIDATES=(${POLYGON_SIMPLIFY_CANDIDATES:-0})
TOPK_PHASE1="${TOPK_PHASE1:-4}"
TOPK_FINAL="${TOPK_FINAL:-3}"
RUN_TEST_FOR_TOPK="${RUN_TEST_FOR_TOPK:-True}"

require_path "$REPO_DPATH"
require_path "$DATA_DPATH"
require_path "$EXPT_DPATH"
require_path "$VALI_FPATH"
require_path "$TEST_FPATH"
require_path "$BASE_PACKAGE_FPATH"

mkdir -p "$SWEEP_ROOT" "$VARIANT_DPATH" "$EVAL_DPATH"

if [ ! -f "$SUMMARY_FPATH" ]; then
    printf 'phase\tvariant_id\tscore_thresh\tnms_thresh\tcrop_padding\tpolygon_simplify\tmin_component_area\tkeep_largest_component\tvali_ap\ttest_ap\tpackage_fpath\tvali_eval_dpath\ttest_eval_dpath\ttimestamp\n' > "$SUMMARY_FPATH"
fi

echo "v5 postprocess sweep"
printf '  %-28s %s\n' "BASE_PACKAGE_FPATH" "$BASE_PACKAGE_FPATH"
printf '  %-28s %s\n' "SWEEP_ROOT" "$SWEEP_ROOT"
printf '  %-28s %s\n' "TOPK_PHASE1" "$TOPK_PHASE1"
printf '  %-28s %s\n' "TOPK_FINAL" "$TOPK_FINAL"
printf '  %-28s %s\n' "RUN_TEST_FOR_TOPK" "$RUN_TEST_FOR_TOPK"
printf '  %-28s %s\n' "BASE_CROP_PADDING" "$BASE_CROP_PADDING"
printf '  %-28s %s\n' "BASE_POLYGON_SIMPLIFY" "$BASE_POLYGON_SIMPLIFY"
printf '  %-28s %s\n' "BASE_MIN_COMPONENT_AREA" "$BASE_MIN_COMPONENT_AREA"
printf '  %-28s %s\n' "BASE_KEEP_LARGEST_COMPONENT" "$BASE_KEEP_LARGEST_COMPONENT"

echo
echo "=== Phase 1: score/nms sweep on validation ==="
for score_thresh in "${SCORE_THRESH_CANDIDATES[@]}"; do
    for nms_thresh in "${NMS_THRESH_CANDIDATES[@]}"; do
        variant_id="thr_s${score_thresh}_n${nms_thresh}"
        variant_id="${variant_id//./p}"
        package_fpath="$VARIANT_DPATH/${variant_id}.yaml"
        vali_eval_dpath="$EVAL_DPATH/$variant_id/vali"
        write_variant_package \
            "$BASE_PACKAGE_FPATH" \
            "$package_fpath" \
            "$variant_id" \
            "$score_thresh" \
            "$nms_thresh" \
            "$BASE_CROP_PADDING" \
            "$BASE_POLYGON_SIMPLIFY" \
            "$BASE_MIN_COMPONENT_AREA" \
            "$BASE_KEEP_LARGEST_COMPONENT"
        if have_metrics "$vali_eval_dpath"; then
            echo "Reusing validation metrics for $variant_id"
        else
            evaluate_variant "$package_fpath" "$vali_eval_dpath" vali
        fi
        vali_ap="$(read_ap "$vali_eval_dpath/eval/detect_metrics.json")"
        append_summary_row "$SUMMARY_FPATH" "phase1" "$variant_id" \
            "$score_thresh" "$nms_thresh" \
            "$BASE_CROP_PADDING" "$BASE_POLYGON_SIMPLIFY" \
            "$BASE_MIN_COMPONENT_AREA" "$BASE_KEEP_LARGEST_COMPONENT" \
            "$vali_ap" "NA" "$package_fpath" "$vali_eval_dpath" "NA"
        printf '  %-24s vali_ap=%s\n' "$variant_id" "$vali_ap"
    done
done

echo
echo "Top phase1 validation results:"
print_top_rows "$SUMMARY_FPATH" 12

mapfile -t PHASE1_TOP_VARIANTS < <(select_top_variant_ids "$SUMMARY_FPATH" "phase1" "$TOPK_PHASE1")

echo
echo "=== Phase 2: mask/postprocess sweep around top phase1 variants ==="
for parent_variant_id in "${PHASE1_TOP_VARIANTS[@]}"; do
    if [ -z "$parent_variant_id" ]; then
        continue
    fi
    IFS=$'\t' read -r parent_score parent_nms _parent_crop _parent_poly _parent_area _parent_keep \
        < <(lookup_variant_row "$SUMMARY_FPATH" "$parent_variant_id")
    for crop_padding in "${CROP_PADDING_CANDIDATES[@]}"; do
        for polygon_simplify in "${POLYGON_SIMPLIFY_CANDIDATES[@]}"; do
            for min_component_area in "${MIN_COMPONENT_AREA_CANDIDATES[@]}"; do
                for keep_largest_component in "${KEEP_LARGEST_COMPONENT_CANDIDATES[@]}"; do
                    variant_id="${parent_variant_id}_cp${crop_padding}_ps${polygon_simplify}_ma${min_component_area}_kl${keep_largest_component}"
                    variant_id="${variant_id//./p}"
                    package_fpath="$VARIANT_DPATH/${variant_id}.yaml"
                    vali_eval_dpath="$EVAL_DPATH/$variant_id/vali"
                    write_variant_package \
                        "$BASE_PACKAGE_FPATH" \
                        "$package_fpath" \
                        "$variant_id" \
                        "$parent_score" \
                        "$parent_nms" \
                        "$crop_padding" \
                        "$polygon_simplify" \
                        "$min_component_area" \
                        "$keep_largest_component"
                    if have_metrics "$vali_eval_dpath"; then
                        echo "Reusing validation metrics for $variant_id"
                    else
                        evaluate_variant "$package_fpath" "$vali_eval_dpath" vali
                    fi
                    vali_ap="$(read_ap "$vali_eval_dpath/eval/detect_metrics.json")"
                    append_summary_row "$SUMMARY_FPATH" "phase2" "$variant_id" \
                        "$parent_score" "$parent_nms" "$crop_padding" "$polygon_simplify" \
                        "$min_component_area" "$keep_largest_component" \
                        "$vali_ap" "NA" "$package_fpath" "$vali_eval_dpath" "NA"
                    printf '  %-24s vali_ap=%s\n' "$variant_id" "$vali_ap"
                done
            done
        done
    done
done

echo
echo "Top overall validation results:"
print_top_rows "$SUMMARY_FPATH" 20

if [ "${RUN_TEST_FOR_TOPK,,}" = "true" ] || [ "${RUN_TEST_FOR_TOPK,,}" = "yes" ] || [ "${RUN_TEST_FOR_TOPK}" = "1" ]; then
    echo
    echo "=== Phase 3: test eval for top validation variants ==="
    mapfile -t FINAL_TOP_VARIANTS < <(python - "$SUMMARY_FPATH" "$TOPK_FINAL" <<'PY'
import csv
import pathlib
import sys

summary_fpath = pathlib.Path(sys.argv[1])
limit = int(sys.argv[2])
rows = []
with summary_fpath.open('r', newline='') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        if row['phase'] not in {'phase1', 'phase2'}:
            continue
        try:
            row['_vali_ap'] = float(row['vali_ap'])
        except Exception:
            continue
        rows.append(row)
rows.sort(key=lambda r: r['_vali_ap'], reverse=True)
seen = set()
for row in rows:
    variant_id = row['variant_id']
    if variant_id in seen:
        continue
    seen.add(variant_id)
    print(variant_id)
    if len(seen) >= limit:
        break
PY
)
    for variant_id in "${FINAL_TOP_VARIANTS[@]}"; do
        [ -n "$variant_id" ] || continue
        package_fpath="$VARIANT_DPATH/${variant_id}.yaml"
        vali_eval_dpath="$EVAL_DPATH/$variant_id/vali"
        test_eval_dpath="$EVAL_DPATH/$variant_id/test"
        if have_metrics "$test_eval_dpath"; then
            echo "Reusing test metrics for $variant_id"
        else
            evaluate_variant "$package_fpath" "$test_eval_dpath" test
        fi
        vali_ap="$(read_ap "$vali_eval_dpath/eval/detect_metrics.json")"
        test_ap="$(read_ap "$test_eval_dpath/eval/detect_metrics.json")"
        IFS=$'\t' read -r score_thresh nms_thresh crop_padding polygon_simplify min_component_area keep_largest_component \
            < <(lookup_variant_row "$SUMMARY_FPATH" "$variant_id")
        append_summary_row "$SUMMARY_FPATH" "phase3" "$variant_id" \
            "$score_thresh" "$nms_thresh" "$crop_padding" "$polygon_simplify" \
            "$min_component_area" "$keep_largest_component" \
            "$vali_ap" "$test_ap" "$package_fpath" "$vali_eval_dpath" "$test_eval_dpath"
        printf '  %-24s vali_ap=%s test_ap=%s\n' "$variant_id" "$vali_ap" "$test_ap"
    done
fi

echo
echo "=== Sweep summary ==="
print_top_rows "$SUMMARY_FPATH" 25
printf '  %-28s %s\n' "SUMMARY_FPATH" "$SUMMARY_FPATH"
