#!/bin/bash
# v13 STEP 1b (separate): scale-balance the train corpus after build_corpus.sh.
#
#   reproduce/in_docker.sh bash experiments/mobile_app_training_v13/balance_corpus.sh
#   bash experiments/mobile_app_training_v13/balance_corpus.sh        # (host)
#
# Draws a (apparent-scale x has-annotation) balanced set with the kit's
# generic balance-scale op (BalancedSampleForest), materializing a kwcoco that
# references the SAME tiles at balanced rates. No re-tiling. Vali is left as-is.
set -euo pipefail

DATA=${V13_DATA:-/media/joncrall/flash1/kcd-ssd/v13/data}

kwcoco-detector-kit balance-scale \
    "$DATA/train_corpus.kwcoco.zip" \
    "$DATA/train_corpus_balanced.kwcoco.zip" \
    --target_size "${TARGET_SIZE:-254000}" \
    --pos_fraction "${POS_FRACTION:-0.4}" \
    --rng 0

echo "[v13] balanced train corpus -> $DATA/train_corpus_balanced.kwcoco.zip"
