#!/bin/bash
# v12 driver: pico@768 resolution push. Plain DEIMv2 — no teacher, no OGDino,
# no docker required. This is the same path that trained the v11 baseline
# (0.588) directly on the host.
#
#   bash run.sh                # train + export + eval + bench
#   bash run.sh --dry_run      # validate recipe, no GPU
#   bash run.sh --force_train  # retrain a completed cell
#
# Extra flags pass through to `kwcoco-detector-kit recipe-run`.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
