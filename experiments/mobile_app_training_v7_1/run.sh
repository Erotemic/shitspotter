#!/bin/bash
# v7.1 driver. Reuses v6.1's corrected tile bundles (no re-tiling).
# If v6.1 hasn't run, the recipe's data path won't exist and the kit
# will fail with a clear error -- run v6.1 first in that case.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

V6_1_DATA=${V6_1_DATA:-/data/joncrall/kcd/v6_1/data}
V7_1_ROOT=${V7_1_ROOT:-/data/joncrall/kcd/v7_1}
mkdir -p "$V7_1_ROOT"

# Validate v6.1's tile bundles exist; v7.1 reuses them directly.
for fname in train_tile_g2.kwcoco.zip vali_tile_g2.kwcoco.zip; do
    if [ ! -f "$V6_1_DATA/$fname" ]; then
        echo "[v7.1] ERROR: missing $V6_1_DATA/$fname" >&2
        echo "       Run v6.1 first to produce the corrected tile bundles." >&2
        exit 2
    fi
done
echo "[v7.1] reusing v6.1 tile bundles -> $V6_1_DATA"

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
