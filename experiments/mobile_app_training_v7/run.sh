#!/bin/bash
# v7 driver. Same shape as v6's; reuses v6's tiled bundles if present.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

RAW_DPATH=${RAW_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}
RAW_TRAIN=${RAW_TRAIN:-$RAW_DPATH/simplified_train_imgs7350_4f0174d0.kwcoco.zip}
RAW_VALI=${RAW_VALI:-$RAW_DPATH/simplified_vali_imgs1258_07ec447d.kwcoco.zip}

V6_DATA=${V6_DATA:-/data/joncrall/kcd/v6/data}
V7_ROOT=${V7_ROOT:-/data/joncrall/kcd/v7}

mkdir -p "$V7_ROOT/data"

for bundle in train_tile_g2.kwcoco.zip vali_tile_g2.kwcoco.zip; do
    if [ ! -f "$V7_ROOT/data/$bundle" ]; then
        if [ -f "$V6_DATA/$bundle" ]; then
            ln -sf "$V6_DATA/$bundle" "$V7_ROOT/data/$bundle"
            echo "[v7] reusing v6 tile bundle -> $V7_ROOT/data/$bundle"
        fi
    fi
done

# If symlinking failed (no v6), tile from raw.
if [ ! -e "$V7_ROOT/data/train_tile_g2.kwcoco.zip" ]; then
    kwcoco-detector-kit tile "$RAW_TRAIN" "$V7_ROOT/data/train_tile_g2.kwcoco.zip" \
        --mode quadrant --category_names poop --tile_grid 2 \
        --tile_overlap 0.20 --tile_output_dim 640 --full_dim 1280
fi
if [ ! -e "$V7_ROOT/data/vali_tile_g2.kwcoco.zip" ]; then
    kwcoco-detector-kit tile "$RAW_VALI" "$V7_ROOT/data/vali_tile_g2.kwcoco.zip" \
        --mode quadrant --category_names poop --tile_grid 2 \
        --tile_overlap 0.20 --tile_output_dim 640 --full_dim 1280
fi

# The recipe currently points at /data/joncrall/kcd/v6/data/. If v6 hasn't
# run, override the data paths via env: V7_USE_OWN_DATA=1 will rewrite the
# recipe's data section in-flight via a small inline override.
if [ "${V7_USE_OWN_DATA:-0}" = "1" ]; then
    TMP_RECIPE=$(mktemp --suffix=.yaml)
    sed "s|/data/joncrall/kcd/v6/data/|$V7_ROOT/data/|g" "$RECIPE" > "$TMP_RECIPE"
    RECIPE="$TMP_RECIPE"
fi

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
