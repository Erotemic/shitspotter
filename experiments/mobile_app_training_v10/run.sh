#!/bin/bash
# v10 driver — runs the kit recipe end-to-end (sweep + manifest).
#
# Prerequisites:
#   1. shitspotter:latest is built off the current Dockerfile
#      (which installs kwcoco_dataloader from the kit's submodule).
#      Rebuild via:
#          bash reproduce/mobile_quality_push.sh build
#   2. v6.1's train kwcoco bundle exists at
#      /data/joncrall/kcd/v6_1/data/train_tile_g2.kwcoco.zip
#      (produced by v6.1's run.sh; v7.1 + v10 both reuse it).
#   3. WebDataset shards built once from that bundle:
#          bash experiments/mobile_app_training_v10/00_build_wds_shards.sh
#      Idempotent; lands under /data/joncrall/kcd/v6_1/data/shards/.
#
# The recipe runner pre-flights the shards directory (checks for
# __footer__.json files) before any GPU minute is spent, so a missing
# or partial shard build fails fast with a clear error message.
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$SCRIPT_DPATH/recipe.yaml"

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
