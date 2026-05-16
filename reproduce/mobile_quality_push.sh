#!/bin/bash
# End-to-end driver for the v6 -> v10 mobile-quality push.
#
# Source-tree assumptions (on the host machine):
#   ~/code/shitspotter                         this repo
#   ~/code/kwcoco_detector_kit                 the kit (cloned beside this repo)
#   ~/code/Open-GroundingDino                  the v3 OGDino fork (cloned)
#   ~/code/YOLO-v9                             (cloned, listed in repos.yaml)
#
# Host paths the docker image bind-mounts:
#   /data/joncrall/dvc-repos/shitspotter_dvc          (raw splits, read-only)
#   /data/joncrall/dvc-repos/shitspotter_expt_dvc     (v9 OGDino package + test GT, ro)
#   /data/joncrall/kcd                                (writable workspace)
#
# Sub-commands (positional):
#   pull        clone or fast-forward the four source repos
#   build       stage + build the docker image (~20 min cold, ~5 min warm)
#   doctor      run `check-env --runtime` inside the image
#   v6 v7 v8 v9 v10
#               run one experiment cell
#   compare     print the v4 vs latest-vN manifest delta
#   all         build + doctor + v6 + v7 + v8 + v9 + (manual v10)
#
# Example:
#   ./reproduce/mobile_quality_push.sh pull
#   ./reproduce/mobile_quality_push.sh build
#   ./reproduce/mobile_quality_push.sh doctor
#   ./reproduce/mobile_quality_push.sh v6
#   ./reproduce/mobile_quality_push.sh compare       # check v6 vs v4
#   ./reproduce/mobile_quality_push.sh v7            # only when v6 looks good
#
set -euo pipefail

SHITSPOTTER_DPATH=${SHITSPOTTER_DPATH:-$HOME/code/shitspotter}
KIT_DPATH=${KIT_DPATH:-$HOME/code/kwcoco_detector_kit}
OGDINO_DPATH=${OGDINO_DPATH:-$HOME/code/Open-GroundingDino}
YOLO_DPATH=${YOLO_DPATH:-$HOME/code/YOLO-v9}

DVC_RO=${DVC_RO:-/data/joncrall/dvc-repos/shitspotter_dvc}
DVC_EXPT_RO=${DVC_EXPT_RO:-/data/joncrall/dvc-repos/shitspotter_expt_dvc}
KCD_HOST=${KCD_HOST:-/data/joncrall/kcd}

IMAGE_TAG=${IMAGE_TAG:-shitspotter:latest}
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.6"}     # RTX 3090
UV_VERSION=${UV_VERSION:-0.8.4}
PYTHON_VERSION=${PYTHON_VERSION:-3.11}

# ---- helpers ----

_ensure_workspace() {
    if [ ! -d "$KCD_HOST" ]; then
        echo "[setup] creating workspace $KCD_HOST"
        mkdir -p "$KCD_HOST" 2>/dev/null || \
            { echo "ERROR: cannot create $KCD_HOST (need sudo or override KCD_HOST=...)" >&2; exit 1; }
    fi
}

_run_in_image() {
    _ensure_workspace
    # All training/eval runs inside the image with the three bind mounts.
    docker run --gpus=all -it --rm \
        -v "$DVC_RO:$DVC_RO:ro" \
        -v "$DVC_EXPT_RO:$DVC_EXPT_RO:ro" \
        -v "$KCD_HOST:$KCD_HOST" \
        "$IMAGE_TAG" "$@"
}

_clone_or_pull() {
    local url="$1"
    local dst="$2"
    if [ -d "$dst/.git" ] || [ -f "$dst/.git" ]; then
        echo "[pull] $dst (fast-forward)"
        git -C "$dst" fetch --tags --quiet
        git -C "$dst" pull --ff-only --quiet || \
            echo "  (skipping ff-only pull -- $dst has diverged or is on a non-tracking branch)"
    else
        echo "[pull] cloning $url -> $dst"
        git clone --recurse-submodules "$url" "$dst"
    fi
    # If the repo has submodules, make sure they're synced.
    if [ -f "$dst/.gitmodules" ]; then
        git -C "$dst" submodule update --init --recursive --quiet
    fi
}

# ---- subcommands ----

cmd_pull() {
    _clone_or_pull https://gitlab.kitware.com/computer-vision/shitspotter "$SHITSPOTTER_DPATH"
    _clone_or_pull https://github.com/Erotemic/kwcoco-detector-kit       "$KIT_DPATH"
    _clone_or_pull git@github.com:Erotemic/Open-GroundingDino.git        "$OGDINO_DPATH"
    _clone_or_pull git@github.com:Erotemic/YOLO.git                      "$YOLO_DPATH"
    echo "[pull] done."
}

cmd_build() {
    cd "$SHITSPOTTER_DPATH"
    python3 ./dockerfiles/setup_staging.py
    local repo_hash
    repo_hash=$(git rev-parse --short=12 HEAD)
    echo "[build] image=$IMAGE_TAG  repo=$repo_hash  arch=$TORCH_CUDA_ARCH_LIST"
    DOCKER_BUILDKIT=1 docker build --progress=plain \
        -t "$IMAGE_TAG" \
        -t "shitspotter:${repo_hash}-uv${UV_VERSION}-python${PYTHON_VERSION}" \
        --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
        --build-arg UV_VERSION="$UV_VERSION" \
        --build-arg REPO_GIT_HASH="$repo_hash" \
        --build-arg TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
        -f ./dockerfiles/shitspotter.dockerfile .
    echo "[build] done. tags: $IMAGE_TAG, shitspotter:${repo_hash}-uv${UV_VERSION}-python${PYTHON_VERSION}"
}

cmd_doctor() {
    _run_in_image kwcoco-detector-kit check-env --runtime \
        --groups core,onnx,deimv2
}

cmd_v6()  { _run_in_image bash experiments/mobile_app_training_v6/run.sh "$@"; }
cmd_v7()  { _run_in_image bash experiments/mobile_app_training_v7/run.sh "$@"; }
cmd_v8()  { _run_in_image bash experiments/mobile_app_training_v8/run.sh "$@"; }
cmd_v9()  { _run_in_image bash experiments/mobile_app_training_v9/run.sh "$@"; }
cmd_v10() { _run_in_image bash experiments/mobile_app_training_v10/run.sh "$@"; }

cmd_compare() {
    # Compare each vN/manifest.tsv against v4's manifest and print the AP
    # delta per cell. Runs inside the image so Python + pandas-ish parsing
    # is available regardless of host Python.
    _run_in_image bash -lc '
        python3 -c "
import csv, sys, os
from pathlib import Path

V4 = Path(\"/data/joncrall/shitspotter_v4/manifest.tsv\")
if not V4.exists():
    print(\"no v4 manifest; skip\"); sys.exit(0)

def load(path):
    rows = {}
    if not Path(path).exists():
        return rows
    with open(path, newline=\"\") as f:
        for r in csv.DictReader(f, delimiter=\"\\t\"):
            key = (r.get(\"variant\",\"\"), r.get(\"export_input_h\",\"\"))
            try:
                ap = float(r.get(\"test_ap\") or r.get(\"test_ap_simplified\") or \"nan\")
            except ValueError:
                ap = float(\"nan\")
            rows[key] = ap
    return rows

v4_rows = load(V4)
print(\"%-40s %10s %10s %10s\" % (\"cell\", \"v4_AP\", \"latest_AP\", \"delta\"))
for vN in (\"v6\",\"v7\",\"v8\",\"v9\",\"v10\"):
    mpath = Path(f\"/data/joncrall/kcd/{vN}/manifest.tsv\")
    rows = load(mpath)
    if not rows:
        continue
    for key, ap in rows.items():
        v4ap = v4_rows.get(key, float(\"nan\"))
        delta = ap - v4ap
        print(\"%-40s %10.4f %10.4f %+10.4f\" % (f\"{vN}:{key[0]}@{key[1]}\", v4ap, ap, delta))
"
    '
}

cmd_all() {
    cmd_build
    cmd_doctor
    cmd_v6
    cmd_compare
    cmd_v7
    cmd_compare
    cmd_v8
    cmd_compare
    cmd_v9
    cmd_compare
    echo
    echo "[all] v10 is not auto-run -- it needs recipe.yaml filled in"
    echo "[all] from v7/v8/v9 results. Run cmd_v10 manually after."
}

# ---- dispatch ----

cmd=${1:-help}
shift || true
case "$cmd" in
    pull)     cmd_pull ;;
    build)    cmd_build ;;
    doctor)   cmd_doctor ;;
    v6|v7|v8|v9|v10) "cmd_$cmd" "$@" ;;
    compare)  cmd_compare ;;
    all)      cmd_all ;;
    help|-h|--help|*)
        echo "Usage: $0 {pull|build|doctor|v6|v7|v8|v9|v10|compare|all} [args...]"
        echo "See the file header for details."
        exit 1
        ;;
esac
