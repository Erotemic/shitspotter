#!/bin/bash
#
# Build WebDataset shards from a shitspotter tile bundle.
#
# This is the one-time data-prep step that produces the on-disk shard
# layout consumed by the v10 (and any later) recipe's
# `data.tile_store: webdataset` opt-in. Train-only — vali stays on the
# random-access kwcoco/JPEG path per ADR-0001.
#
# The build runs inside the shitspotter docker image (which owns the
# /data/joncrall/kcd/ tree as root). If invoked from the host the
# script self-wraps; if invoked from inside a docker container it
# runs the build directly.
#
# Bucketing: kwcoco_dataloader writes shards under
# `dominant_raw_class=<name>/` subdirs. shitspotter is single-class
# (poop) so this collapses to two buckets:
#   - dominant_raw_class=poop/       — tiles with >=1 annotation
#   - dominant_raw_class=<empty>/    — hard-negative tiles
# The detection reader's `load_bucket_streams` can weight those two
# streams independently — useful when we want to control the
# empty-tile ratio without rebuilding shards.
#
# Idempotency: writes a `.build_done` marker on completion. Subsequent
# runs reuse the existing shards. Set FORCE_RESHARD=1 to rebuild.
#
# Env knobs (read by both the outer host-side and the inner in-docker
# invocations; the wrapper forwards them through):
#   TRAIN_KWCOCO    input kwcoco bundle path
#                   default: /data/joncrall/kcd/v6_1/data/train_tile_g2.kwcoco.zip
#   SHARDS_DPATH    output directory for the shard tree
#                   default: <dirname TRAIN_KWCOCO>/shards
#   MAXCOUNT        max samples per shard (default 5000)
#   MAXSIZE_MB      max bytes per shard, MB (default 1024)
#   JPEG_QUALITY    re-encode quality for non-JPEG sources (default 95)
#   FORCE_RESHARD   non-empty: rebuild even if .build_done exists
#
# Docker-only knobs (ignored once inside the container):
#   SHITSPOTTER_IMAGE  image tag (default: shitspotter:latest)
#   KCD_HOST_DPATH     host-side path that holds /data/joncrall/kcd
#                      (default: /data/joncrall/kcd; mounted rw)
#   DOCKER_BIN         docker binary (default: docker)
#   SKIP_DOCKER        non-empty: assume we already have write access
#                      and run the build directly on the host
#
# Output also written: SHARDS_DPATH/_build_args.txt for provenance.
#
# After this script succeeds, point the v10 recipe at SHARDS_DPATH via:
#
#     data:
#       tile_store: webdataset
#       train_wds_shards: <SHARDS_DPATH>
#
# Source bundle context: the bit-identical-to-v4 train_imgs10671-
# derived bundle that v6.1 + v7.1 + the bisect all consumed.
# ~53K tiles / ~22K annotations.
set -euo pipefail

# ---- Host-side wrapper: re-exec inside docker if we're outside one.
# Detection prefers /.dockerenv (kernel-level marker) and falls back to
# an explicit IN_DOCKER=1 override for cases where the marker is
# missing (rootless docker, non-standard images).
if [ ! -f "/.dockerenv" ] && [ -z "${IN_DOCKER:-}" ] && [ -z "${SKIP_DOCKER:-}" ]; then
    SCRIPT_DPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SHITSPOTTER_REPO="$(cd "$SCRIPT_DPATH/../.." && pwd)"
    SCRIPT_REL="experiments/mobile_app_training_v10/$(basename "${BASH_SOURCE[0]}")"

    DOCKER_BIN="${DOCKER_BIN:-docker}"
    SHITSPOTTER_IMAGE="${SHITSPOTTER_IMAGE:-shitspotter:latest}"
    # Default to the SSD root used by v10's recipe; override
    # KCD_HOST_DPATH to point somewhere else if you're building
    # shards from a different bundle.
    KCD_HOST_DPATH="${KCD_HOST_DPATH:-/media/joncrall/flash1/kcd-ssd}"

    if ! command -v "$DOCKER_BIN" >/dev/null 2>&1; then
        echo "[00_build_wds_shards] $DOCKER_BIN not found." >&2
        echo "                       Re-run from inside the shitspotter image, or set" >&2
        echo "                       SKIP_DOCKER=1 if /data/joncrall/kcd is writable on host." >&2
        exit 1
    fi
    if [ ! -d "$KCD_HOST_DPATH" ]; then
        echo "[00_build_wds_shards] KCD_HOST_DPATH does not exist: $KCD_HOST_DPATH" >&2
        exit 1
    fi

    echo "[00_build_wds_shards] outside docker — re-exec inside $SHITSPOTTER_IMAGE"
    echo "[00_build_wds_shards] mount: $KCD_HOST_DPATH -> /data/joncrall/kcd (rw)"
    echo "[00_build_wds_shards] mount: $SHITSPOTTER_REPO -> /work/shitspotter (ro)"

    exec "$DOCKER_BIN" run --rm \
        -v "$KCD_HOST_DPATH":/data/joncrall/kcd \
        -v "$SHITSPOTTER_REPO":/work/shitspotter:ro \
        -e IN_DOCKER=1 \
        -e TRAIN_KWCOCO="${TRAIN_KWCOCO:-}" \
        -e SHARDS_DPATH="${SHARDS_DPATH:-}" \
        -e MAXCOUNT="${MAXCOUNT:-}" \
        -e MAXSIZE_MB="${MAXSIZE_MB:-}" \
        -e JPEG_QUALITY="${JPEG_QUALITY:-}" \
        -e FORCE_RESHARD="${FORCE_RESHARD:-}" \
        "$SHITSPOTTER_IMAGE" \
        bash "/work/shitspotter/$SCRIPT_REL"
fi

# ---- From here down: in-docker (or SKIP_DOCKER) build path.
TRAIN_KWCOCO="${TRAIN_KWCOCO:-/media/joncrall/flash1/kcd-ssd/v6_1/data/train_tile_g2.kwcoco.zip}"
SHARDS_DPATH="${SHARDS_DPATH:-$(dirname "$TRAIN_KWCOCO")/shards}"
MAXCOUNT="${MAXCOUNT:-5000}"
MAXSIZE_MB="${MAXSIZE_MB:-1024}"
JPEG_QUALITY="${JPEG_QUALITY:-95}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Sanity-check: the image must have a kwcoco_dataloader recent enough
# to expose build_detection_webdataset (dev/0.1.3 or later). Catches
# the "stale image" failure mode with a clear message instead of a
# confusing ModuleNotFoundError half a second later.
if ! "$PYTHON_BIN" -c "from kwcoco_dataloader.cli.build_detection_webdataset import BuildDetectionWebdatasetCLI" 2>/dev/null; then
    echo "[00_build_wds_shards] image does not expose kwcoco_dataloader.cli.build_detection_webdataset." >&2
    echo "                       Image likely pre-dates the dev/0.1.3 dataloader merge." >&2
    echo "                       Rebuild via:  bash reproduce/mobile_quality_push.sh build" >&2
    exit 3
fi

if [ ! -f "$TRAIN_KWCOCO" ]; then
    echo "[00_build_wds_shards] TRAIN_KWCOCO does not exist: $TRAIN_KWCOCO" >&2
    exit 1
fi

SHARDS_DONE_MARKER="$SHARDS_DPATH/.build_done"
if [ -f "$SHARDS_DONE_MARKER" ] && [ -z "${FORCE_RESHARD:-}" ]; then
    echo "[00_build_wds_shards] $SHARDS_DPATH already built (marker present)."
    echo "[00_build_wds_shards] Set FORCE_RESHARD=1 to rebuild."
    echo "[00_build_wds_shards] Recipe should set:"
    echo "  data:"
    echo "    tile_store: webdataset"
    echo "    train_wds_shards: $SHARDS_DPATH"
    exit 0
fi

mkdir -p "$SHARDS_DPATH"

echo "[00_build_wds_shards] in:  $TRAIN_KWCOCO"
echo "[00_build_wds_shards] out: $SHARDS_DPATH"
echo "[00_build_wds_shards] maxcount=$MAXCOUNT maxsize_mb=$MAXSIZE_MB jpeg_quality=$JPEG_QUALITY"

# Record what we ran, for provenance. Lives next to the shards so a
# future bug report (or a hash-drift audit) can reconstruct the
# build inputs without digging through shell history.
cat > "$SHARDS_DPATH/_build_args.txt" <<EOF
# Built by experiments/mobile_app_training_v10/00_build_wds_shards.sh
# at $(date -u +"%Y-%m-%dT%H:%M:%SZ")
TRAIN_KWCOCO=$TRAIN_KWCOCO
SHARDS_DPATH=$SHARDS_DPATH
MAXCOUNT=$MAXCOUNT
MAXSIZE_MB=$MAXSIZE_MB
JPEG_QUALITY=$JPEG_QUALITY
git_sha_shitspotter=$(git -C "$(dirname "$0")" rev-parse HEAD 2>/dev/null || echo unknown)
EOF

# bucket_attr stays at the default `dominant_raw_class`; for
# shitspotter that produces poop/ and <empty>/ buckets. Empty-tile
# weighting at train time happens via the reader's bucket weight_fn,
# NOT here.
# kwcoco_dataloader doesn't ship a top-level __main__.py; invoke the
# CLI module path directly. Same fix as the kit applied to sealions
# in kwcoco_detector_kit commit 79e28dd.
"$PYTHON_BIN" -m kwcoco_dataloader.cli.build_detection_webdataset \
    --in_fpath "$TRAIN_KWCOCO" \
    --out_dpath "$SHARDS_DPATH" \
    --bucket_attr dominant_raw_class \
    --maxcount "$MAXCOUNT" \
    --maxsize_mb "$MAXSIZE_MB" \
    --jpeg_quality "$JPEG_QUALITY" \
    --drop_provenance false \
    --progress true

# Confirm the writer actually produced footer files; a crashed run
# would leave the dir but no __footer__.json — the recipe runner's
# pre-flight checks for the same condition so catch it here too.
N_FOOTERS=$(find "$SHARDS_DPATH" -name "__footer__.json" | wc -l)
if [ "$N_FOOTERS" -eq 0 ]; then
    echo "[00_build_wds_shards] writer produced no __footer__.json files;" >&2
    echo "                       NOT touching .build_done. Investigate." >&2
    exit 2
fi

touch "$SHARDS_DONE_MARKER"

N_SHARDS=$(find "$SHARDS_DPATH" -name "*.tar" | wc -l)
TOTAL_MB=$(du -sm "$SHARDS_DPATH" | awk '{print $1}')
echo
echo "[00_build_wds_shards] done: $N_SHARDS shards, $N_FOOTERS bucket footers, ${TOTAL_MB} MB"
echo "[00_build_wds_shards] Recipe should set:"
echo "  data:"
echo "    tile_store: webdataset"
echo "    train_wds_shards: $SHARDS_DPATH"
