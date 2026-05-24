#!/bin/bash
# v6 driver: tile the raw splits once, then run the kit recipe end-to-end.
#
# Assumes you are inside the shitspotter docker image (or have the kit
# installed locally with KCD_DEIMV2_REPO_DPATH set). The container's
# bind mounts should expose:
#
#   /data/joncrall/dvc-repos/shitspotter_dvc        (read-only, raw splits)
#   /data/joncrall/dvc-repos/shitspotter_expt_dvc   (read-only, v9 test GT)
#   /data/joncrall/kcd                              (writable, $KCD_ROOT)
#
# Pass --skip_checks to bypass the leading check-env probe, or any of
# --force_train / --force_export / --force_eval / --force_bench to
# re-run a specific stage.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

RAW_DPATH=${RAW_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}
RAW_TRAIN=${RAW_TRAIN:-$RAW_DPATH/simplified_train_imgs7350_4f0174d0.kwcoco.zip}
RAW_VALI=${RAW_VALI:-$RAW_DPATH/simplified_vali_imgs1258_07ec447d.kwcoco.zip}

V6_ROOT=${V6_ROOT:-/data/joncrall/kcd/v6}
DATA_DPATH=$V6_ROOT/data
mkdir -p "$DATA_DPATH"

echo "[v6] raw train -> $RAW_TRAIN"
echo "[v6] raw vali  -> $RAW_VALI"
echo "[v6] workspace -> $V6_ROOT"

# Tile (idempotent). quadrant mode @ grid=2 / overlap=0.20 / dim=640 mirrors v4.
if [ ! -f "$DATA_DPATH/train_tile_g2.kwcoco.zip" ]; then
    echo "[v6] tile train (quadrant g2)"
    kwcoco-detector-kit tile \
        "$RAW_TRAIN" "$DATA_DPATH/train_tile_g2.kwcoco.zip" \
        --mode quadrant \
        --category_names poop \
        --tile_grid 2 \
        --tile_overlap 0.20 \
        --tile_output_dim 640 \
        --full_dim 1280
else
    echo "[v6] skip tile train ($DATA_DPATH/train_tile_g2.kwcoco.zip exists)"
fi
if [ ! -f "$DATA_DPATH/vali_tile_g2.kwcoco.zip" ]; then
    echo "[v6] tile vali (quadrant g2)"
    kwcoco-detector-kit tile \
        "$RAW_VALI" "$DATA_DPATH/vali_tile_g2.kwcoco.zip" \
        --mode quadrant \
        --category_names poop \
        --tile_grid 2 \
        --tile_overlap 0.20 \
        --tile_output_dim 640 \
        --full_dim 1280
else
    echo "[v6] skip tile vali ($DATA_DPATH/vali_tile_g2.kwcoco.zip exists)"
fi

# Run the recipe (sweep + manifest). All remaining args forwarded.
exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
