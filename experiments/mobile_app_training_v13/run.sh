#!/bin/bash
# v13 driver: build the multi-scale tile corpus with the GENERIC kit
# tile-corpus builder, then train pico@768 on it. DEIMv2 only — no OGDino,
# no teacher. Runs on the host like the v11 baseline, or in docker
# (--shm-size=16g) for the GPU.
#
#   bash run.sh                # build corpus (if missing) + train/export/eval/bench
#   bash run.sh --dry_run      # validate recipe, no GPU
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEC="$SCRIPT_DPATH/corpus_spec.yaml"

RAW_DPATH=${RAW_DPATH:-/data/joncrall/dvc-repos/shitspotter_dvc}
RAW_TRAIN=${RAW_TRAIN:-$RAW_DPATH/train_imgs10671_b277c63d.kwcoco.zip}
RAW_VALI=${RAW_VALI:-$RAW_DPATH/vali_imgs1258_577e331c.kwcoco.zip}
DATA=${V13_DATA:-/media/joncrall/flash1/kcd-ssd/v13/data}
mkdir -p "$DATA"

# Build corpora via the generic kit builder (shared with sealions / any project).
if [ ! -f "$DATA/train_corpus.kwcoco.zip" ]; then
    echo "[v13] building train corpus via tile-corpus"
    kwcoco-detector-kit tile-corpus "$RAW_TRAIN" "$DATA/train_corpus.kwcoco.zip" --spec "$SPEC"
fi
if [ ! -f "$DATA/vali_corpus.kwcoco.zip" ]; then
    echo "[v13] building vali corpus via tile-corpus"
    kwcoco-detector-kit tile-corpus "$RAW_VALI" "$DATA/vali_corpus.kwcoco.zip" --spec "$SPEC"
fi

exec kwcoco-detector-kit recipe-run "$SCRIPT_DPATH/recipe.yaml" "$@"
