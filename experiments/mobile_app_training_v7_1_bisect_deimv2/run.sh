#!/bin/bash
# v7.1 DEIMv2 bisect driver. Same shape as v7.1's run.sh but with a
# separate workspace and a single cell.
#
# Build the image with the bisect-pinned kit (commit a8ca45e on the
# kit's main branch -- tpl/DEIMv2 at 377e10a) before running this.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

V6_1_DATA=${V6_1_DATA:-/data/joncrall/kcd/v6_1/data}
V7_1_BISECT_ROOT=${V7_1_BISECT_ROOT:-/data/joncrall/kcd/v7_1_bisect_deimv2}
mkdir -p "$V7_1_BISECT_ROOT"

for fname in train_tile_g2.kwcoco.zip vali_tile_g2.kwcoco.zip; do
    if [ ! -f "$V6_1_DATA/$fname" ]; then
        echo "[bisect] ERROR: missing $V6_1_DATA/$fname" >&2
        echo "         Run v6.1 first to produce the corrected tile bundles." >&2
        exit 2
    fi
done
echo "[bisect] reusing v6.1 tile bundles -> $V6_1_DATA"
echo "[bisect] this run uses DEIMv2 377e10a (rolled back from aeabc7e)"

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
