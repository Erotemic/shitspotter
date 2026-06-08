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
# (rebuild with `reproduce/mobile_quality_push.sh build`); set a per-repo
# MOUNT_<REPO>=1 flag to bind-mount that live checkout for quick iteration
# without a rebuild (the occasional exception — e.g. testing a kit fix).
#
# Env knobs:
#   IMAGE_TAG   image to run            (default shitspotter:latest)
#   GPUS        --gpus value, "" to disable (default all)
#   SHM_SIZE    /dev/shm                (default 32g)
#   DETACH=1    run -d (background) instead of -it; pair with NAME
#   NAME        container name          (default none)
#   DVC_RO / DVC_EXPT_RO / KCD_HOST / SSD / V4_PRETRAINED_RO  override host paths
#
# Per-repo dev mounts (default "0"; set to "1" to bind-mount over the image's
# baked copy, avoiding a rebuild). Each works because the image installs the
# repo editable (-e) from /root/code/<repo>. Override the host path with the
# matching <REPO>_REPO_ROOT.
#   MOUNT_KCD=1          kwcoco_detector_kit  (host: KCD_REPO_ROOT)
#   MOUNT_SHITSPOTTER=1  shitspotter          (host: SHITSPOTTER_REPO_ROOT)
#   CODE_MOUNT=1         back-compat shortcut for MOUNT_KCD + MOUNT_SHITSPOTTER
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

# Code: from the image by default (rebuild to update). Per-repo MOUNT_<REPO>=1
# bind-mounts the live host checkout over the baked copy for rebuild-free
# iteration. Host paths default to $HOME/code/<repo>; override via <REPO>_REPO_ROOT.
KCD_REPO_ROOT=${KCD_REPO_ROOT:-$HOME/code/kwcoco_detector_kit}
SHITSPOTTER_REPO_ROOT=${SHITSPOTTER_REPO_ROOT:-$HOME/code/shitspotter}

# CODE_MOUNT=1 stays as a back-compat shortcut for "mount both code repos".
if [ "${CODE_MOUNT:-0}" = "1" ]; then
    MOUNT_KCD=${MOUNT_KCD:-1}
    MOUNT_SHITSPOTTER=${MOUNT_SHITSPOTTER:-1}
fi

# maybe_mount_repo <enabled 0|1> <host_path> <container_path> <label>
maybe_mount_repo() {
    [ "$1" = "1" ] || return 0
    if [ -d "$2" ]; then
        args+=(-v "$2:$3")
        echo "[in_docker] MOUNT $4: $2 -> $3 (image copy shadowed)" >&2
    else
        echo "[in_docker] WARNING: MOUNT $4 requested but host path missing: $2" >&2
    fi
}

# Add another mountable repo by copying one line (+ its <REPO>_REPO_ROOT default).
maybe_mount_repo "${MOUNT_KCD:-0}"         "$KCD_REPO_ROOT"         /root/code/kwcoco_detector_kit KCD
maybe_mount_repo "${MOUNT_SHITSPOTTER:-0}" "$SHITSPOTTER_REPO_ROOT" /root/code/shitspotter         SHITSPOTTER

args+=(-w /root/code/shitspotter "$IMAGE_TAG" "$@")

echo "[in_docker] ${args[*]}" >&2
exec "${args[@]}"
