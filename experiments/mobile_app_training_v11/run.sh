#!/bin/bash
# v11 driver: pico@640 resolution bump, two isolated arms.
#
#   bash run.sh baseline      # train pico@640 on human GT only
#   bash run.sh distill       # ensure teacher pseudo-GT merged bundle, then train
#   bash run.sh both           # baseline, then distill (default)
#   bash run.sh both --dry_run # extra flags are forwarded to recipe-run
#
# The distill arm reuses v9's offline pseudo-GT mechanism: the OGDino bbox
# teacher (AP=0.766) predicts boxes on the SAME train tiles, those boxes are
# tagged from_teacher=True and merged into the human GT. Pseudo-GT is on the
# tiles, so it is independent of the student's 640 input — the only thing
# that differs from the baseline arm is the training kwcoco.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_RECIPE="$SCRIPT_DPATH/recipe.baseline.yaml"
DISTILL_RECIPE="$SCRIPT_DPATH/recipe.distill.yaml"

ARM="${1:-both}"
[ $# -gt 0 ] && shift   # remaining args ($@) forwarded to recipe-run

# --- data + teacher paths (override via env) -------------------------------
TRAIN_TILES=${TRAIN_TILES:-/media/joncrall/flash1/kcd-ssd/v6_1/data/train_tile_g2.kwcoco.zip}
V11_DATA=${V11_DATA:-/media/joncrall/flash1/kcd-ssd/v11/data}
PSEUDO_KWCOCO="$V11_DATA/train_tile_g2_teacher_pseudo.kwcoco.zip"
MERGED_KWCOCO="$V11_DATA/train_tile_g2_merged.kwcoco.zip"

# The OGDino teacher is repackaged from the v3-pipeline
# selected_detector_checkpoint.yaml into a KIT-NATIVE package (trainer +
# artifacts layout) that `pseudo-label`/`predict_kwcoco` can consume — the
# foundation-format package (backend/detector.config_fpath) is NOT understood
# by predict_kwcoco (that mismatch is why v9 KeyError'd). See
# make_teacher_kit_package.py.
V9_SELECTED_YAML=${V9_SELECTED_YAML:-/data/joncrall/dvc-repos/shitspotter_expt_dvc/foundation_detseg_v3/v9/selected_detector_checkpoint.yaml}
TEACHER_KIT_PKG=${TEACHER_KIT_PKG:-$V11_DATA/teacher_kit_package}

ensure_merged() {
    mkdir -p "$V11_DATA"
    if [ ! -f "$TRAIN_TILES" ]; then
        echo "[v11] ERROR: train tile bundle missing: $TRAIN_TILES" >&2
        echo "[v11]        stage v6.1 tiles on the SSD (or override TRAIN_TILES)" >&2
        exit 1
    fi
    if [ -f "$MERGED_KWCOCO" ]; then
        echo "[v11] merged bundle already exists: $MERGED_KWCOCO"
        return
    fi

    # Build the kit-native OGDino teacher package (idempotent).
    if [ ! -f "$TEACHER_KIT_PKG/package.yaml" ]; then
        [ -f "$V9_SELECTED_YAML" ] || { echo "[v11] ERROR: v9 selected_detector_checkpoint.yaml missing: $V9_SELECTED_YAML" >&2; exit 1; }
        echo "[v11] repackaging OGDino teacher -> kit-native package $TEACHER_KIT_PKG"
        python3 "$SCRIPT_DPATH/make_teacher_kit_package.py" \
            --selected_yaml "$V9_SELECTED_YAML" \
            --out_pkg "$TEACHER_KIT_PKG" \
            --label poop
    fi

    # Step 1: teacher predicts boxes over the train tiles.
    # NOTE: needs $KCD_OPENGROUNDINGDINO_REPO_DPATH set + the OGDino repo
    # importable (the predictor imports groundingdino lazily).
    if [ ! -f "$PSEUDO_KWCOCO" ]; then
        echo "[v11] pseudo-label train tiles with OGDino teacher"
        kwcoco-detector-kit pseudo-label "$TEACHER_KIT_PKG" \
            --src "$TRAIN_TILES" \
            --dst "$PSEUDO_KWCOCO" \
            --device "cuda:0" \
            --score_thresh 0.30 \
            --min_annotations 0
    fi

    # Step 2: merge human GT + teacher pseudo-GT (teacher anns tagged
    # from_teacher=True, weight 1.0). Same logic as v9/run.sh.
    echo "[v11] merging human + teacher annotations -> $MERGED_KWCOCO"
    python3 -c "
import kwcoco
human = kwcoco.CocoDataset('$TRAIN_TILES')
teacher = kwcoco.CocoDataset('$PSEUDO_KWCOCO')
merged = human.copy()
human_imgs_by_name = {g['file_name']: g for g in merged.imgs.values()}
imported = skipped = 0
for ann in teacher.anns.values():
    timg = teacher.imgs[ann['image_id']]
    hum = human_imgs_by_name.get(timg['file_name'])
    if hum is None:
        skipped += 1; continue
    new = dict(ann); new.pop('id', None)
    new['image_id'] = hum['id']
    new['from_teacher'] = True
    new['weight'] = 1.0
    new['category_id'] = merged.ensure_category(name=teacher.cats[ann['category_id']]['name'])
    merged.add_annotation(**new)
    imported += 1
# Absolutize image paths (against the v6.1 bundle dir) BEFORE moving the
# bundle: the merged kwcoco is written to v11/data but the tile JPEGs live
# in v6_1/data/train_tile_g2_assets/. kwcoco stores file_names relative to
# the bundle dir, so without this the trainer looks for them under v11/data.
merged.reroot(absolute=True)
merged.fpath = '$MERGED_KWCOCO'
merged.dump()
print(f'[v11] merged: imported {imported} teacher anns; skipped {skipped} no-match')
"
}

case "$ARM" in
    baseline)
        exec kwcoco-detector-kit recipe-run "$BASELINE_RECIPE" "$@"
        ;;
    distill)
        ensure_merged
        exec kwcoco-detector-kit recipe-run "$DISTILL_RECIPE" "$@"
        ;;
    both)
        kwcoco-detector-kit recipe-run "$BASELINE_RECIPE" "$@"
        ensure_merged
        exec kwcoco-detector-kit recipe-run "$DISTILL_RECIPE" "$@"
        ;;
    *)
        echo "usage: run.sh [baseline|distill|both] [recipe-run flags...]" >&2
        exit 2
        ;;
esac
