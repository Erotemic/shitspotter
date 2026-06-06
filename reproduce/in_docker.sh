#!/bin/bash
# Generic "run a command inside the shitspotter docker image" wrapper.
#
# It is a transparent PREFIX: the rest of the argv is the command to run.
#
#     reproduce/in_docker.sh kwcoco-detector-kit tile-corpus ...     # in container
#                            kwcoco-detector-kit tile-corpus ...     # on the host
#
# i.e. drop the `reproduce/in_docker.sh` prefix to run the exact same command
# on the host instead. It bakes in the standard data mounts, GPU, and
# /dev/shm so every step (tile-corpus, recipe-run, predict, eval, ...) runs
# with one consistent environment. Code comes from the IMAGE by default
# (rebuild with `reproduce/mobile_quality_push.sh build`); set CODE_MOUNT=1 to
# bind-mount the live repos for quick iteration (the occasional exception).
#
# Env knobs:
#   IMAGE_TAG   image to run            (default shitspotter:latest)
#   GPUS        --gpus value, "" to disable (default all)
#   SHM_SIZE    /dev/shm                (default 32g)
#   DETACH=1    run -d (background) instead of -it; pair with NAME
#   NAME        container name          (default none)
#   CODE_MOUNT=1  bind-mount live kwcoco_detector_kit + shitspotter (dev only)
#   DVC_RO / DVC_EXPT_RO / KCD_HOST / SSD / V4_PRETRAINED_RO  override host paths
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: reproduce/in_docker.sh <command...>   (the command to run in the image)" >&2
    exit 2
fi

IMAGE_TAG=${IMAGE_TAG:-shitspotter:latest}
GPUS=${GPUS:-all}
SHM_SIZE=${SHM_SIZE:-32g}

DVC_RO=${DVC_RO:-/data/joncrall/dvc-repos/shitspotter_dvc}
DVC_EXPT_RO=${DVC_EXPT_RO:-/data/joncrall/dvc-repos/shitspotter_expt_dvc}
KCD_HOST=${KCD_HOST:-/data/joncrall/kcd}
SSD=${SSD:-/media/joncrall/flash1/kcd-ssd}
V4_PRETRAINED_RO=${V4_PRETRAINED_RO:-/data/joncrall/shitspotter_v4}
# The DVC bundles bake absolute asset paths under /home/joncrall/data/...;
# mount the real source at those legacy paths too.
DVC_LEGACY_RO=${DVC_LEGACY_RO:-/home/joncrall/data/dvc-repos/shitspotter_dvc}
DVC_EXPT_LEGACY_RO=${DVC_EXPT_LEGACY_RO:-/home/joncrall/data/dvc-repos/shitspotter_expt_dvc}

args=(docker run --rm)
[ -n "$GPUS" ] && args+=(--gpus="$GPUS")
if [ "${DETACH:-0}" = "1" ]; then args+=(-d); else args+=(-it); fi
[ -n "${NAME:-}" ] && args+=(--name "$NAME")
args+=(--shm-size="$SHM_SIZE")

# Data mounts (same path inside as out so recipe/spec paths are identical).
[ -d "$DVC_RO" ]           && args+=(-v "$DVC_RO:$DVC_RO:ro" -v "$DVC_RO:$DVC_LEGACY_RO:ro")
[ -d "$DVC_EXPT_RO" ]      && args+=(-v "$DVC_EXPT_RO:$DVC_EXPT_RO:ro" -v "$DVC_EXPT_RO:$DVC_EXPT_LEGACY_RO:ro")
[ -d "$KCD_HOST" ]         && args+=(-v "$KCD_HOST:$KCD_HOST")
[ -d "$SSD" ]              && args+=(-v "$SSD:$SSD")
[ -d "$V4_PRETRAINED_RO" ] && args+=(-v "$V4_PRETRAINED_RO:$V4_PRETRAINED_RO:ro")

# Code: from the image by default (rebuild to update). CODE_MOUNT=1 binds live.
if [ "${CODE_MOUNT:-0}" = "1" ]; then
    [ -d "$HOME/code/kwcoco_detector_kit" ] && args+=(-v "$HOME/code/kwcoco_detector_kit:/root/code/kwcoco_detector_kit")
    [ -d "$HOME/code/shitspotter" ]         && args+=(-v "$HOME/code/shitspotter:/root/code/shitspotter")
    echo "[in_docker] CODE_MOUNT=1: bind-mounting live repos (image code shadowed)" >&2
fi

args+=(-w /root/code/shitspotter "$IMAGE_TAG" "$@")

echo "[in_docker] ${args[*]}" >&2
exec "${args[@]}"
