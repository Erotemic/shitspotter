#!/bin/bash
# v9 driver: pseudo-label with v9 OGDino bbox teacher, merge into human
# GT, then train via kit recipe-run.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

# Use v6.1's corrected tile bundle (built from train_imgs10671 = v4's
# source). v6.0 used the wrong simplified_train_imgs7350 bundle which
# capped training at 25% of v4's image count; see
# experiments/V4_VS_KIT_APPLES_TO_APPLES.md for the discovery.
V6_TILE_TRAIN=${V6_TILE_TRAIN:-/data/joncrall/kcd/v6_1/data/train_tile_g2.kwcoco.zip}

# The v9 OGDino+SAM2 deployment package YAML referenced in
# foundation_detseg_v3/v9/selected_detector_checkpoint.yaml isn't on
# disk (only the *_default template is). We generate it on the fly in
# the workspace below, filling in the three v9-specific paths
# (detector cfg + ckpt + segmenter ckpt) from the selected_detector
# YAML the v3 pipeline already produced. Override TEACHER_PACKAGE to
# point at a hand-authored YAML if you'd rather control it directly.
TEACHER_PACKAGE_TEMPLATE=${TEACHER_PACKAGE_TEMPLATE:-/root/code/shitspotter/experiments/foundation_detseg_v3/packages/opengroundingdino_sam2_default.yaml}
V9_SELECTED_YAML=${V9_SELECTED_YAML:-/data/joncrall/dvc-repos/shitspotter_expt_dvc/foundation_detseg_v3/v9/selected_detector_checkpoint.yaml}

V9_ROOT=${V9_ROOT:-/data/joncrall/kcd/v9}
V9_DATA=$V9_ROOT/data
mkdir -p "$V9_DATA"

PSEUDO_KWCOCO="$V9_DATA/train_tile_g2_teacher_pseudo.kwcoco.zip"
MERGED_KWCOCO="$V9_DATA/train_tile_g2_merged.kwcoco.zip"
TEACHER_PACKAGE=${TEACHER_PACKAGE:-$V9_DATA/v9_teacher_package.yaml}

if [ ! -f "$V6_TILE_TRAIN" ]; then
    echo "[v9] ERROR: v6.1 tiled train bundle missing: $V6_TILE_TRAIN" >&2
    echo "[v9]        run v6.1 first (or override V6_TILE_TRAIN to point elsewhere)" >&2
    exit 1
fi

# Generate the v9 teacher package YAML from the v3-pipeline-produced
# selected_detector_checkpoint.yaml (idempotent; skip if already there).
if [ ! -f "$TEACHER_PACKAGE" ]; then
    if [ ! -f "$TEACHER_PACKAGE_TEMPLATE" ]; then
        echo "[v9] ERROR: teacher template missing: $TEACHER_PACKAGE_TEMPLATE" >&2
        exit 1
    fi
    if [ ! -f "$V9_SELECTED_YAML" ]; then
        echo "[v9] ERROR: v9 selected_detector_checkpoint.yaml missing: $V9_SELECTED_YAML" >&2
        echo "[v9]        ensure the v3 foundation_detseg_v3 pipeline ran for v9" >&2
        exit 1
    fi
    echo "[v9] generating teacher package -> $TEACHER_PACKAGE"
    python3 - <<PY
import yaml
sel = yaml.safe_load(open("$V9_SELECTED_YAML"))
pkg = yaml.safe_load(open("$TEACHER_PACKAGE_TEMPLATE"))
pkg["detector"]["config_fpath"]    = sel["detector_config_fpath"]
pkg["detector"]["checkpoint_fpath"] = sel["selected_detector_checkpoint_fpath"]
pkg["segmenter"]["checkpoint_fpath"] = sel["tuned_segmenter_checkpoint_fpath"]
pkg["metadata"]["name"] = "v9_opengroundingdino_sam2_1_hiera_base_plus_tuned"
pkg["metadata"]["source_selected_yaml"] = "$V9_SELECTED_YAML"
pkg["metadata"]["selected_candidate_id"] = sel.get("selected_candidate_id", "")
pkg["metadata"]["detector_test_simplified_ap"] = sel.get("detector_test_simplified_ap", "")
with open("$TEACHER_PACKAGE","w") as f:
    yaml.safe_dump(pkg, f, sort_keys=False)
print("[v9] wrote teacher package with detector_test_simplified_ap=" + str(sel.get("detector_test_simplified_ap","")))
PY
fi

# Step 1: teacher predicts boxes over the v6 tiled training pool.
if [ ! -f "$PSEUDO_KWCOCO" ]; then
    echo "[v9] pseudo-label train tiles with v9 OGDino teacher"
    kwcoco-detector-kit pseudo-label "$TEACHER_PACKAGE" \
        --src "$V6_TILE_TRAIN" \
        --dst "$PSEUDO_KWCOCO" \
        --device "cuda:0" \
        --score_thresh 0.30 \
        --min_annotations 0
else
    echo "[v9] pseudo-label kwcoco already exists: $PSEUDO_KWCOCO"
fi

# Step 2: merge human GT + teacher pseudo-GT into one training kwcoco.
# Teacher boxes are tagged via annotation['from_teacher'] = True so the
# trainer can downweight them later if helpful. For v9 the default is
# weight=1.0 (treat as regular GT).
if [ ! -f "$MERGED_KWCOCO" ]; then
    echo "[v9] merging human + teacher annotations"
    python3 -c "
import kwcoco
human = kwcoco.CocoDataset('$V6_TILE_TRAIN')
teacher = kwcoco.CocoDataset('$PSEUDO_KWCOCO')

merged = human.copy()
merged.fpath = '$MERGED_KWCOCO'

# kwcoco union assigns new IDs; we want to keep human image IDs intact
# and just absorb the teacher's annotations on the same images.
human_imgs_by_name = {g['file_name']: g for g in merged.imgs.values()}
imported = 0
skipped = 0
for ann in teacher.anns.values():
    timg = teacher.imgs[ann['image_id']]
    hum = human_imgs_by_name.get(timg['file_name'])
    if hum is None:
        skipped += 1
        continue
    new = dict(ann)
    new.pop('id', None)
    new['image_id'] = hum['id']
    new['from_teacher'] = True
    new['weight'] = 1.0
    cat_name = teacher.cats[ann['category_id']]['name']
    new['category_id'] = merged.ensure_category(name=cat_name)
    merged.add_annotation(**new)
    imported += 1

merged.dump()
print(f'[v9] merged: imported {imported} teacher anns; skipped {skipped} no-match')
"
else
    echo "[v9] merged kwcoco already exists: $MERGED_KWCOCO"
fi

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
