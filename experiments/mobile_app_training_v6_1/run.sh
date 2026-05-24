#!/bin/bash
# v6.1 driver -- same as v6.0 but with the CORRECT source bundle.
# Tiles the v4-equivalent input (10671 train + 1258 vali images) into
# v6.1's own workspace, then runs the kit recipe end-to-end.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

RAW_DPATH=${RAW_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}
# CORRECTED (vs v6.0): use the 10671-image bundle that v4 used, not the
# 2564-image "simplified" bundle. See V4_VS_KIT_APPLES_TO_APPLES.md.
RAW_TRAIN=${RAW_TRAIN:-$RAW_DPATH/train_imgs10671_b277c63d.kwcoco.zip}
RAW_VALI=${RAW_VALI:-$RAW_DPATH/vali_imgs1258_577e331c.kwcoco.zip}

V6_1_ROOT=${V6_1_ROOT:-/data/joncrall/kcd/v6_1}
DATA_DPATH=$V6_1_ROOT/data
mkdir -p "$DATA_DPATH"

echo "[v6.1] raw train -> $RAW_TRAIN"
echo "[v6.1] raw vali  -> $RAW_VALI"
echo "[v6.1] workspace -> $V6_1_ROOT"

# Tile (idempotent). quadrant mode @ grid=2 / overlap=0.20 / dim=640 mirrors v4.
if [ ! -f "$DATA_DPATH/train_tile_g2.kwcoco.zip" ]; then
    echo "[v6.1] tile train (quadrant g2)"
    kwcoco-detector-kit tile \
        "$RAW_TRAIN" "$DATA_DPATH/train_tile_g2.kwcoco.zip" \
        --mode quadrant \
        --category_names poop \
        --tile_grid 2 \
        --tile_overlap 0.20 \
        --tile_output_dim 640 \
        --full_dim 1280
else
    echo "[v6.1] skip tile train ($DATA_DPATH/train_tile_g2.kwcoco.zip exists)"
fi
if [ ! -f "$DATA_DPATH/vali_tile_g2.kwcoco.zip" ]; then
    echo "[v6.1] tile vali (quadrant g2)"
    kwcoco-detector-kit tile \
        "$RAW_VALI" "$DATA_DPATH/vali_tile_g2.kwcoco.zip" \
        --mode quadrant \
        --category_names poop \
        --tile_grid 2 \
        --tile_overlap 0.20 \
        --tile_output_dim 640 \
        --full_dim 1280
else
    echo "[v6.1] skip tile vali ($DATA_DPATH/vali_tile_g2.kwcoco.zip exists)"
fi

# Sanity-check: confirm the corrected bundle produces ~v4-scale tile output.
# v6.0 with the simplified bundle produced 12820 train tiles; v4 with the
# 10671-image bundle produced 53355. v6.1 should land near v4's count.
python3 - <<PY
import zipfile, json
for label, p in [("train", "$DATA_DPATH/train_tile_g2.kwcoco.zip"),
                 ("vali",  "$DATA_DPATH/vali_tile_g2.kwcoco.zip")]:
    with zipfile.ZipFile(p) as z:
        jn = next(n for n in z.namelist() if n.endswith(".json"))
        with z.open(jn) as f:
            d = json.load(f)
    print(f"[v6.1] tile-output {label}: {len(d['images'])} images, "
          f"{len(d['annotations'])} annotations")
PY

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
