#!/bin/bash
# v13 STEP 1 (separate, invoke explicitly): build the multi-scale tile corpus
# via the generic kit `tile-corpus` builder. Pure CLI — no docker logic here;
# run it on the host, or in the container by prefixing the docker wrapper:
#
#     reproduce/in_docker.sh bash experiments/mobile_app_training_v13/build_corpus.sh
#     bash experiments/mobile_app_training_v13/build_corpus.sh        # (host, no docker)
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="$SCRIPT_DPATH/corpus_spec.yaml"

RAW_DPATH=${RAW_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}
RAW_TRAIN=${RAW_TRAIN:-$RAW_DPATH/train_imgs10671_b277c63d.kwcoco.zip}
RAW_VALI=${RAW_VALI:-$RAW_DPATH/vali_imgs1258_577e331c.kwcoco.zip}
DATA=${V13_DATA:-/media/joncrall/flash1/kcd-ssd/v13/data}
mkdir -p "$DATA"

if [ ! -f "$DATA/train_corpus.kwcoco.zip" ]; then
    echo "[v13] building train corpus via tile-corpus"
    kwcoco-detector-kit tile-corpus "$RAW_TRAIN" "$DATA/train_corpus.kwcoco.zip" --spec "$SPEC"
else
    echo "[v13] train corpus exists: $DATA/train_corpus.kwcoco.zip"
fi

if [ ! -f "$DATA/vali_corpus.kwcoco.zip" ]; then
    echo "[v13] building vali corpus via tile-corpus"
    kwcoco-detector-kit tile-corpus "$RAW_VALI" "$DATA/vali_corpus.kwcoco.zip" --spec "$SPEC"
else
    echo "[v13] vali corpus exists: $DATA/vali_corpus.kwcoco.zip"
fi

echo "[v13] corpus ready under $DATA"
