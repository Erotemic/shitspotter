#!/bin/bash
# v10 driver — runs the kit recipe end-to-end (sweep + manifest).
#
# Self-wraps inside shitspotter:latest so it works directly from the
# host. Same detection pattern as 00_build_wds_shards.sh
# (/.dockerenv / IN_DOCKER / SKIP_DOCKER).
#
# Prerequisites:
#   1. shitspotter:latest is built off a current Dockerfile + current
#      kit checkout (DEIMv2 submodule pin must include 6b5a2ef /
#      img_folder accept). Rebuild via:
#          bash reproduce/mobile_quality_push.sh build
#   2. v6.1's train kwcoco bundle exists at
#      /data/joncrall/kcd/v6_1/data/train_tile_g2.kwcoco.zip
#   3. WebDataset shards built once from that bundle:
#          bash experiments/mobile_app_training_v10/00_build_wds_shards.sh
#
# Docker-only knobs (ignored once inside):
#   SHITSPOTTER_IMAGE   image tag (default: shitspotter:latest)
#   KCD_SSD_DPATH       SSD-backed workspace + tile bundle root (rw,
#                       default /media/joncrall/flash1/kcd-ssd). This
#                       is what recipe.yaml's data.* + workspace.kcd_root
#                       paths resolve under.
#   KCD_HOST_DPATH      legacy HDD-backed kcd root (rw). Mounted as a
#                       fallback so older recipes / data still resolve.
#                       Set to "" to skip the mount.
#   DVC_RO              shitspotter_dvc host path (ro)
#   DVC_EXPT_RO         shitspotter_expt_dvc host path (ro)
#   V4_PRETRAINED_RO    DEIMv2 COCO-pretrained .pth root (ro)
#   SHM_SIZE            /dev/shm size for DataLoader IPC (default 32g)
#   GPUS                docker --gpus flag value (default all)
#   DOCKER_BIN          docker binary (default docker)
#   SKIP_DOCKER         non-empty: run on host directly (assumes
#                       kwcoco-detector-kit is on PATH)
#
# Forwarded into the container:
#   $@                  forwarded verbatim to recipe-run (e.g. --dry_run,
#                       --force_train, --force_eval, ...)
set -euo pipefail

SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Host-side wrapper: re-exec inside docker if we're outside one.
if [ ! -f "/.dockerenv" ] && [ -z "${IN_DOCKER:-}" ] && [ -z "${SKIP_DOCKER:-}" ]; then
    SHITSPOTTER_REPO="$(cd "$SCRIPT_DPATH/../.." && pwd)"
    SCRIPT_REL="experiments/mobile_app_training_v10/$(basename "${BASH_SOURCE[0]}")"

    DOCKER_BIN="${DOCKER_BIN:-docker}"
    SHITSPOTTER_IMAGE="${SHITSPOTTER_IMAGE:-shitspotter:latest}"
    KCD_SSD_DPATH="${KCD_SSD_DPATH:-/media/joncrall/flash1/kcd-ssd}"
    KCD_HOST_DPATH="${KCD_HOST_DPATH:-/data/joncrall/kcd}"
    DVC_RO="${DVC_RO:-/data/joncrall/dvc-repos/shitspotter_dvc}"
    DVC_EXPT_RO="${DVC_EXPT_RO:-/data/joncrall/dvc-repos/shitspotter_expt_dvc}"
    # The DVC bundle has absolute asset paths baked in pointing at
    # /home/joncrall/data/dvc-repos/shitspotter_{,expt_}dvc/. Bind the
    # same source at BOTH paths the bundle might reference.
    DVC_LEGACY_RO="${DVC_LEGACY_RO:-/home/joncrall/data/dvc-repos/shitspotter_dvc}"
    DVC_EXPT_LEGACY_RO="${DVC_EXPT_LEGACY_RO:-/home/joncrall/data/dvc-repos/shitspotter_expt_dvc}"
    V4_PRETRAINED_RO="${V4_PRETRAINED_RO:-/data/joncrall/shitspotter_v4}"
    SHM_SIZE="${SHM_SIZE:-32g}"
    GPUS="${GPUS:-all}"

    if ! command -v "$DOCKER_BIN" >/dev/null 2>&1; then
        echo "[v10/run.sh] $DOCKER_BIN not found." >&2
        echo "             Run from inside the shitspotter image, or set" >&2
        echo "             SKIP_DOCKER=1 if kwcoco-detector-kit is on PATH." >&2
        exit 1
    fi

    # Pretrained dir is optional at the wrapper layer; the recipe
    # runner errors with a clear message if init_checkpoint paths
    # don't resolve inside the container.
    pretrained_mount=()
    if [ -d "$V4_PRETRAINED_RO" ]; then
        pretrained_mount=(-v "$V4_PRETRAINED_RO:$V4_PRETRAINED_RO:ro")
    fi
    # SSD mount is required for v10; bail early with a clear message
    # if the host path doesn't exist (likely means data hasn't been
    # rsync'd over yet).
    if [ ! -d "$KCD_SSD_DPATH" ]; then
        echo "[v10/run.sh] KCD_SSD_DPATH does not exist: $KCD_SSD_DPATH" >&2
        echo "             Copy v6.1's tiles to SSD first, e.g.:" >&2
        echo "               mkdir -p $KCD_SSD_DPATH/v6_1" >&2
        echo "               rsync -aP /data/joncrall/kcd/v6_1/data $KCD_SSD_DPATH/v6_1/" >&2
        exit 1
    fi
    # Legacy HDD kcd mount is optional — only mount when present and
    # non-empty so a host with that path unmounted still works.
    kcd_legacy_mount=()
    if [ -n "${KCD_HOST_DPATH:-}" ] && [ -d "$KCD_HOST_DPATH" ]; then
        kcd_legacy_mount=(-v "$KCD_HOST_DPATH:$KCD_HOST_DPATH")
    fi

    echo "[v10/run.sh] outside docker — re-exec inside $SHITSPOTTER_IMAGE"
    echo "[v10/run.sh] mounts: kcd_ssd=$KCD_SSD_DPATH(rw)  dvc=$DVC_RO(ro)  dvc_expt=$DVC_EXPT_RO(ro)"
    if [ "${#kcd_legacy_mount[@]}" -gt 0 ]; then
        echo "[v10/run.sh] also mounting legacy kcd=$KCD_HOST_DPATH(rw)"
    fi
    echo "[v10/run.sh] shm=$SHM_SIZE gpus=$GPUS"

    exec "$DOCKER_BIN" run --gpus="$GPUS" --rm -it \
        --shm-size="$SHM_SIZE" \
        -v "$DVC_RO:$DVC_RO:ro" \
        -v "$DVC_EXPT_RO:$DVC_EXPT_RO:ro" \
        -v "$DVC_RO:$DVC_LEGACY_RO:ro" \
        -v "$DVC_EXPT_RO:$DVC_EXPT_LEGACY_RO:ro" \
        -v "$KCD_SSD_DPATH:$KCD_SSD_DPATH" \
        "${kcd_legacy_mount[@]}" \
        -v "$SHITSPOTTER_REPO:/work/shitspotter:ro" \
        "${pretrained_mount[@]}" \
        -e IN_DOCKER=1 \
        "$SHITSPOTTER_IMAGE" \
        bash "/work/shitspotter/$SCRIPT_REL" "$@"
fi

# ---- In-container (or SKIP_DOCKER) path: actually run the recipe.
# Prefer the bind-mounted host source if we self-wrapped (so recipe
# edits are visible immediately); fall back to the script's own dpath
# when run from the baked-in image copy (e.g. via mobile_quality_push.sh).
if [ -f "/work/shitspotter/experiments/mobile_app_training_v10/recipe.yaml" ]; then
    RECIPE="/work/shitspotter/experiments/mobile_app_training_v10/recipe.yaml"
else
    RECIPE="$SCRIPT_DPATH/recipe.yaml"
fi

exec kwcoco-detector-kit recipe-run "$RECIPE" "$@"
