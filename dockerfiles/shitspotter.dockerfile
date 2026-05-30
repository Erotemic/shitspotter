# syntax=docker/dockerfile:1.5
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


# ------------------------------------
# Step 1: Install System Prerequisites
# ------------------------------------

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt/lists <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    curl \
    wget \
    git \
    unzip \
    ca-certificates \
    build-essential \
    ninja-build
# Cleanup for smaller image sizes
apt clean
rm -rf /var/lib/apt/lists/*
EOF

# Set the shell to bash to auto-activate environments
SHELL ["/bin/bash", "-l", "-c"]


# Step 2: Install uv
# ------------------
# Here we take a few extra steps to pin to a verified version of the uv
# installer. This increases reproducibility and security against the main
# astral domain, but not against those linked in the main installer.
# The "normal" way to install the latest uv is:
# curl -LsSf https://astral.sh/uv/install.sh | bash

# Control the version of uv
ARG UV_VERSION=0.8.4

RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
set -e
mkdir /bootstrap
cd /bootstrap
# For new releases see: https://github.com/astral-sh/uv/releases
declare -A UV_INSTALL_KNOWN_HASHES=(
    ["0.8.4"]="601321180a10e0187c99d8a15baa5ccc11b03494c2ca1152fc06f5afeba0a460"
    ["0.7.20"]="3b7ca115ec2269966c22201b3a82a47227473bef2fe7066c62ea29603234f921"
    ["0.7.19"]="e636668977200d1733263a99d5ea66f39d4b463e324bb655522c8782d85a8861"
)
EXPECTED_SHA256="${UV_INSTALL_KNOWN_HASHES[${UV_VERSION}]}"
DOWNLOAD_PATH=uv-install-v${UV_VERSION}.sh
if [[ -z "$EXPECTED_SHA256" ]]; then
    echo "No hash known for UV_VERSION '$UV_VERSION'; no known hash. Aborting."
    exit 1
fi
curl -LsSf https://astral.sh/uv/$UV_VERSION/install.sh > $DOWNLOAD_PATH
report_bad_checksum(){
    echo "Got unexpected checksum"
    sha256sum "$DOWNLOAD_PATH"
    exit 1
}
echo "$EXPECTED_SHA256  $DOWNLOAD_PATH" | sha256sum --check || report_bad_checksum
# Run the install script
bash /bootstrap/uv-install-v${UV_VERSION}.sh
EOF


# ------------------------------------------
# Step 3: Setup a Python virtual environment
# ------------------------------------------
# This step mirrors a normal virtualenv development environment inside the
# container, which can prevent subtle issues due when running as root inside
# containers. 

# Control which python version we are using
ARG PYTHON_VERSION=3.10

ENV PIP_ROOT_USER_ACTION=ignore

RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
# Use uv to install the requested python version and seed the venv
uv venv "/root/venv$PYTHON_VERSION" --python=$PYTHON_VERSION --seed
BASHRC_CONTENTS='
# setup a user-like environment, even though we are root
export HOME="/root"
export PATH="$HOME/.local/bin:$PATH"
# Auto-activate the venv on login
source $HOME/venv'$PYTHON_VERSION'/bin/activate
'
# It is important to add the content to both so
# subsequent run commands use the context we setup here.
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile
echo "$BASHRC_CONTENTS" >> $HOME/.bash_profile
EOF


# -----------------------------------------------------------------------
# Step 3.5: Pre-install CUDA-matched torch wheels (BEFORE any other pip)
# -----------------------------------------------------------------------
# As of 2026 PyPI's default torch wheels are built against CUDA 13.0. Our
# base image provides CUDA 12.4, so loading the default torch breaks any
# C++/CUDA extension that calls torch.utils.cpp_extension._check_cuda_version
# (DEIMv2's MSDeformAttn, OGDino's MSDeformAttn). Pin the cu124-built
# wheels here so every downstream `uv pip install` finds torch already at
# the right version and skips the resolve. Override --build-arg
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 if you bump the
# BASE_IMAGE to a CUDA 13.x base.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
# Promote ARG -> ENV so bash heredocs below can see it. Docker only
# substitutes ${VAR} in RUN exec-form / shell-form (NOT in heredoc
# bodies); bash does the heredoc expansion using its own env.
ENV TORCH_INDEX_URL=${TORCH_INDEX_URL}

RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
source $HOME/venv$PYTHON_VERSION/bin/activate
# Fail loudly if the ARG->ENV promotion didn't take effect; otherwise an
# empty --index-url would silently fall back to default PyPI and install
# the wrong-CUDA torch wheel.
if [ -z "${TORCH_INDEX_URL:-}" ]; then
    echo "ERROR: TORCH_INDEX_URL is empty inside the bash heredoc; the" >&2
    echo "       ARG->ENV promotion above must be wrong." >&2
    exit 1
fi
echo "[torch-pin] using TORCH_INDEX_URL=${TORCH_INDEX_URL}"
# No --extra-index-url: uv's resolver may pick a higher version from
# PyPI even when --index-url points at the CUDA-pinned wheel index.
# Single index keeps us on the cu-suffixed wheels.
uv pip install \
    --index-url ${TORCH_INDEX_URL} \
    torch torchvision torchaudio
# Assert nvcc and torch agree on the CUDA major.minor BEFORE we waste
# any time building extensions. This is the same guard the kit's
# docker/opengroundingdino/Dockerfile uses.
python - <<'PY'
import re, shutil, subprocess
import torch
torch_cuda = torch.version.cuda
nvcc = shutil.which("nvcc")
text = subprocess.check_output([nvcc, "--version"], text=True)
match = re.search(r"release\s+([0-9]+\.[0-9]+)", text)
nvcc_cuda = match.group(1) if match else None
print(f"torch={torch.__version__}  torch.version.cuda={torch_cuda}  nvcc={nvcc_cuda}")
if not torch_cuda or not nvcc_cuda or torch_cuda != nvcc_cuda:
    raise SystemExit(
        f"CUDA ABI mismatch: torch.version.cuda={torch_cuda!r}, nvcc={nvcc_cuda!r}. "
        f"Match BASE_IMAGE and TORCH_INDEX_URL: cuda 12.4 -> cu124, 13.0 -> cu130, etc."
    )
PY
EOF


# -----------------------------------
# Step 4: Ensure venv auto-activation
# -----------------------------------
# This step creates an entrypoint script that ensures any command passed to
# `docker run` is executed inside a login shell where the virtual environment
# is auto-activated. It handles complex cases like multi-arg commands and
# ensures quoting is preserved accurately.
RUN <<EOF
#!/bin/bash
set -e

# We use a quoted heredoc to write the entrypoint script literally, with no variable expansion.
cat <<'__EOSCRIPT__' > /entrypoint.sh
#!/bin/bash
set -e

# Reconstruct the full command line safely, quoting each argument
args=()
for arg in "$@"; do
  args+=("$(printf "%q" "$arg")")
done

# Join arguments into a command string that can be executed by bash -c
# This preserves exact argument semantics (including quotes, spaces, etc.)
cmd="${args[*]}"

# Execute the reconstructed command inside a login shell
# This ensures virtualenv activation via .bash_profile
exec bash -l -c "$cmd"
__EOSCRIPT__

# Print the script at build time for visibility/debugging
cat /entrypoint.sh

chmod +x /entrypoint.sh
EOF

# Set the entrypoint to our script that activates the virtual environment first
ENTRYPOINT ["/entrypoint.sh"]


# ---------------------------------
# Step 5: Checkout and install REPO
# ---------------------------------
# Based on the state of the repo this copies the host .git data over and then
# checks out the extact version of REPO requested by REPO_GIT_HASH. It then
# performs a basic install of shitspotter into the virtual environment.

RUN mkdir -p /root/code/shitspotter

# Control the version of REPO (by default uses the current branch)
ARG REPO_GIT_HASH=HEAD

# NOTE: our .dockerignore file prevents us from copying in populated secrets /
# credentials
COPY .git /root/code/shitspotter/.git
RUN <<EOF
#!/bin/bash
set -e

cd  /root/code/shitspotter

# Checkout the requested branch 
git checkout "$REPO_GIT_HASH"
git reset --hard "$REPO_GIT_HASH"

# TODO: cleanup once we determine the best way to 
# install the REPO package for reproducibility. 

# TODO: add lock file for reproducibility
uv pip install -r requirements.txt

uv pip install -e .[headless,optional,tests,lint] 
#--resolution lowest-direct

# Handle special dependencies
geowatch finish_install

# Cleanup for smaller cache
rm -rf /root/.cache/
EOF

# ---------------------------------
# Step NEW: add other repos
# ---------------------------------

COPY .staging/Open-GroundingDino /root/code/Open-GroundingDino
COPY .staging/YOLO-v9 /root/code/YOLO-v9

RUN <<EOF
#!/bin/bash
set -e
cd  /root/code/YOLO-v9
uv pip install -e .
EOF

# ---------------------------------------------------------------
# Step NEW: install kwcoco_detector_kit (the v6+ training interface)
# ---------------------------------------------------------------
# The kit owns its own DEIMv2 + Open-GroundingDino submodules under
# tpl/. We point KCD_DEIMV2_REPO_DPATH at the kit's DEIMv2 so the
# trainer dispatch picks up the pinned SHA (kept in lockstep by
# setup_staging.py's recurse_submodules path). We install the [dev]
# + [deimv2] extras at build time; the [opengroundingdino] extras are
# only needed for v9 distillation and are installed lazily there.
COPY .staging/kwcoco_detector_kit /root/code/kwcoco_detector_kit

RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
set -e
cd /root/code/kwcoco_detector_kit
# kwcoco_dataloader is staged as a submodule under
# tpl/kwcoco_dataloader (pinned to dev/0.1.3; setup_staging.py recurses
# submodules for kwcoco_detector_kit per repos.yaml). Install it from
# the local checkout because the [kwcoco-dataloader] extra pins
# kwcoco-dataloader>=0.1.3 from PyPI, which doesn't exist yet.
#
# v10 needs this for the WebDataset training-input path
# (data.tile_store: webdataset in recipe.v1). v6-v9 don't touch
# kwcoco_dataloader at runtime, so the install is additive — adding it
# doesn't perturb their behavior.
test -f tpl/kwcoco_dataloader/kwcoco_dataloader/__init__.py || \
    { echo "kwcoco_dataloader submodule missing from staged kit." >&2;
      echo "Re-run setup_staging.py (it should recurse_submodules)." >&2;
      exit 1; }
uv pip install -e ./tpl/kwcoco_dataloader
# WebDataset + wids deps (the [webdataset] kit extra brings webdataset
# and braceexpand; we also need wids for the random-access read path
# and line_profiler for the bench harness). Mirrors the kit's own
# Dockerfile pattern at docker/opengroundingdino/Dockerfile.
uv pip install "webdataset>=0.2" "wids>=0.1" braceexpand "line_profiler>=4.0"

uv pip install -e ".[dev,deimv2]"

# Bake provenance into the image so every artifact produced inside is
# self-describing. kwcoco_detector_kit/_provenance.py reads this file
# at runtime; failing that it falls back to live git rev-parse of the
# installed source. Storing it in /etc/ means: even if a future user
# mutates the kit source (via -v bind-mount for iteration), the
# image's intended provenance is preserved.
mkdir -p /etc
python3 - <<PY
import json, subprocess, os
def _sha(p):
    try:
        return subprocess.check_output(
            ["git", "-C", p, "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None
prov = {
    "kit_sha": _sha("/root/code/kwcoco_detector_kit"),
    "deimv2_sha": _sha("/root/code/kwcoco_detector_kit/tpl/DEIMv2"),
    "opengroundingdino_sha":
        _sha("/root/code/kwcoco_detector_kit/tpl/Open-GroundingDino"),
    "image_built_at": subprocess.check_output(["date","-u","+%Y-%m-%dT%H:%M:%SZ"], text=True).strip(),
}
with open("/etc/kcd_provenance.json","w") as f:
    json.dump(prov, f, indent=2)
print("provenance baked:", prov)
PY
EOF

# Compile DEIMv2's MultiScaleDeformableAttention CUDA extension. Builds
# for the RTX 3090 architecture (SM 8.6) by default; override with
# --build-arg TORCH_CUDA_ARCH_LIST="..." for other hardware. The build
# only needs nvcc + CUDA headers (provided by cuda-devel base); no GPU
# is required at image-build time.
ARG TORCH_CUDA_ARCH_LIST="8.6"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV KCD_DEIMV2_REPO_DPATH=/root/code/kwcoco_detector_kit/tpl/DEIMv2

# Build the kit's Open-GroundingDino MSDeformAttention extension. This
# is only required for v9 distillation (the OGDino bbox teacher), but
# we build it at image time so the v9 step doesn't surprise users with
# a long CUDA compile when they're trying to start a training run.
RUN --mount=type=cache,target=/root/.cache <<EOF
#!/bin/bash
set -e
cd /root/code/kwcoco_detector_kit/tpl/Open-GroundingDino
# Lightweight deps to drive setup.py; the trainer plugin's full deps
# are gated behind kit's [opengroundingdino] extras.
uv pip install addict yapf colorlog pycocotools timm 'transformers>=4.35,<4.47' jsonlines
cd models/GroundingDINO/ops

# Two flavors of OGDino setup.py exist in our submodule history:
#
#   pre-9ddf1037: gate is `if torch.cuda.is_available() and CUDA_HOME is not None:`
#                 -- requires a sed patch so headless docker builds (no GPU
#                 visible at build time) still take the CUDAExtension path.
#
#   9ddf1037+:    gate is `if (torch.cuda.is_available() or force_cuda) and ...`
#                 -- built-in FORCE_CUDA=1 env-var path, no patch needed.
#
# Detect the new gate first and skip the patch if it's there; otherwise
# apply the legacy patch.
if grep -q "force_cuda" setup.py; then
    echo "[ogdino-build] setup.py supports FORCE_CUDA env var natively"
elif grep -q "if torch.cuda.is_available() and CUDA_HOME is not None:" setup.py; then
    echo "[ogdino-build] applying legacy sed patch to setup.py CUDA gate"
    sed -i 's/if torch.cuda.is_available() and CUDA_HOME is not None:/if CUDA_HOME is not None:/' setup.py
    grep -q "if CUDA_HOME is not None:" setup.py || { echo "patch failed"; exit 1; }
else
    echo "[ogdino-build] WARN: setup.py has neither known CUDA gate; build may fail" >&2
fi

# FORCE_CUDA=1 is the modern flavor's switch; harmless for the legacy
# (patched) flavor since the gate no longer checks for it.
export FORCE_CUDA=1

python setup.py build_ext --inplace -v

# The forked OGDino expects the .so next to the package root, not under
# models/GroundingDINO/ops/. Copy + smoke-import (matches hacky_setup.sh).
# We deliberately do NOT execute the .so here -- importing a CUDA
# extension on a no-GPU host fails -- the check-env --runtime probe
# imports it at runtime when a GPU is present.
TORCH_LIB_DPATH=$(dirname $(find $(python -c "import torch; print(torch.__path__[0])") -name "libc10.so" | head -1))
export LD_LIBRARY_PATH=$TORCH_LIB_DPATH:$LD_LIBRARY_PATH
cp MultiScaleDeformableAttention.*.so ../../../
ls -la ../../../MultiScaleDeformableAttention.*.so
echo "OGDino MSDeformAttention .so built (runtime import deferred to check-env --runtime)"
EOF

# Convenience env so the kit's tools see the OGDino .so at runtime.
# (Entrypoint sources .bashrc which extends PYTHONPATH if a user has
# their own additions; this is the base.)
ENV PYTHONPATH=/root/code/kwcoco_detector_kit/tpl/Open-GroundingDino

# Set the default workdir to the shitspotter code repo
WORKDIR /root/code/shitspotter

# ---------------------------------------------------------------
# End of dockerfile logic. The following lines are documentation.
# ---------------------------------------------------------------

################
### __DOCS__ ###
################
RUN <<EOF
echo 'HEREDOC:
# https://www.docker.com/blog/introduction-to-heredocs-in-dockerfiles/

# The following are instructions to build and test this docker image

# cd into a local clone of the shitspotter repo
cd ~/code/shitspotter/

# Determine which shitspotter version to use
REPO_GIT_HASH=$(git rev-parse --short=12 HEAD)

python ./dockerfiles/setup_staging.py

# Determine version of repo, uv, and python to use
export REPO_GIT_HASH=$(git rev-parse --short=12 HEAD)
export UV_VERSION=0.8.4
export PYTHON_VERSION=3.11

# Pick the CUDA arch(s) to compile MSDeformAttention for. RTX 3090 = 8.6,
# 4090 = 8.9, A100 = 8.0, H100 = 9.0. Multiple are allowed: "8.0;8.6;8.9".
export TORCH_CUDA_ARCH_LIST="8.6"

# Torch wheel index URL. MUST match the BASE_IMAGE CUDA version:
#   nvidia/cuda:12.4.*  -> https://download.pytorch.org/whl/cu124
#   nvidia/cuda:12.6.*  -> https://download.pytorch.org/whl/cu126
#   nvidia/cuda:12.8.*  -> https://download.pytorch.org/whl/cu128
#   nvidia/cuda:13.0.*  -> https://download.pytorch.org/whl/cu130
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

# Build the image with version-specific tags
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t shitspotter:${REPO_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION} \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg UV_VERSION=$UV_VERSION \
    --build-arg REPO_GIT_HASH=$REPO_GIT_HASH \
    --build-arg TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST \
    --build-arg TORCH_INDEX_URL=$TORCH_INDEX_URL \
    -f ./dockerfiles/shitspotter.dockerfile .

# Add concise tags for easier reuse
export IMAGE_QUALNAME=shitspotter:${REPO_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION}
export NAME1=shitspotter:latest-uv${UV_VERSION}-python${PYTHON_VERSION}
export NAME2=shitspotter:latest-python${PYTHON_VERSION}
export NAME3=shitspotter:latest
docker tag $IMAGE_QUALNAME $NAME1
docker tag $IMAGE_QUALNAME $NAME2
docker tag $IMAGE_QUALNAME $NAME3

# Verify that GPUs are visible and the kit imports
docker run --gpus=all --rm shitspotter:latest nvidia-smi
docker run --gpus=all --rm shitspotter:latest \
    kwcoco-detector-kit check-env --runtime

# Start a shell and run any custom tests
# (See reproduce/mobile_quality_push.sh for the v6-v10 driver.)
docker run --gpus=all -it shitspotter:latest bash

# 1) Authenticate (recommended: use a Docker Hub access token)
#    Create a token in Docker Hub -> Account Settings -> Security
#    Then run:
# echo "<your-access-token>" | docker login --username "$DOCKERHUB_USER" --password-stdin
#
# If you must, you can use interactive login:
docker login

export DH_USER="erotemic"

# 3) Create remote-qualified tags
docker tag $IMAGE_QUALNAME $DH_USER/$IMAGE_QUALNAME
docker tag $NAME1  $DH_USER/$NAME1
docker tag $NAME2  $DH_USER/$NAME2
docker tag $NAME3  $DH_USER/$NAME3

# 4) Push the tags
docker push $DH_USER/$IMAGE_QUALNAME
docker push $DH_USER/$NAME1
docker push $DH_USER/$NAME2
docker push $DH_USER/$NAME3
docker push $DH_USER:latest-uv0.7.29-python3.11
docker push $DH_USER:latest


' > /dev/null

EOF
