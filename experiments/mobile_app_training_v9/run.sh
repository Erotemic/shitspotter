#!/bin/bash
# v9 driver: pseudo-label with v9 OGDino bbox teacher, merge into human
# GT, then train via kit recipe-run.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

V6_TILE_TRAIN=${V6_TILE_TRAIN:-/data/joncrall/kcd/v6/data/train_tile_g2.kwcoco.zip}
TEACHER_PACKAGE=${TEACHER_PACKAGE:-/data/joncrall/dvc-repos/shitspotter_expt_dvc/foundation_detseg_v3/v9/packages/v9_opengroundingdino_sam2_1_hiera_base_plus_tuned.yaml}

V9_ROOT=${V9_ROOT:-/data/joncrall/kcd/v9}
V9_DATA=$V9_ROOT/data
mkdir -p "$V9_DATA"

PSEUDO_KWCOCO="$V9_DATA/train_tile_g2_teacher_pseudo.kwcoco.zip"
MERGED_KWCOCO="$V9_DATA/train_tile_g2_merged.kwcoco.zip"

if [ ! -f "$V6_TILE_TRAIN" ]; then
    echo "[v9] ERROR: v6 tiled train bundle missing: $V6_TILE_TRAIN" >&2
    echo "[v9]        run v6 first (or override V6_TILE_TRAIN to point elsewhere)" >&2
    exit 1
fi
if [ ! -f "$TEACHER_PACKAGE" ]; then
    echo "[v9] ERROR: v9 OGDino teacher package missing: $TEACHER_PACKAGE" >&2
    exit 1
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
