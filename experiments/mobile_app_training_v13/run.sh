#!/bin/bash
# v13 STEP 2 (separate): train pico@768 on the multi-scale corpus. Corpus build
# is a SEPARATE prior step (build_corpus.sh) — run that first. Pure CLI, no
# docker logic; run on the host or via the wrapper:
#
#     reproduce/in_docker.sh bash experiments/mobile_app_training_v13/build_corpus.sh   # step 1
#     DETACH=1 NAME=v13 reproduce/in_docker.sh \
#         bash experiments/mobile_app_training_v13/run.sh                                # step 2
#     bash experiments/mobile_app_training_v13/run.sh        # (host, no docker)
#
# Extra flags pass through to recipe-run (e.g. --dry_run, --force_train).
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA=${V13_DATA:-/media/joncrall/flash1/kcd-ssd/v13/data}

if [ ! -f "$DATA/train_corpus.kwcoco.zip" ]; then
    echo "[v13] ERROR: $DATA/train_corpus.kwcoco.zip missing." >&2
    echo "[v13]        run build_corpus.sh first (separate step)." >&2
    exit 1
fi

exec kwcoco-detector-kit recipe-run "$SCRIPT_DPATH/recipe.yaml" "$@"
