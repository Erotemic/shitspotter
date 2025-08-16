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
    build-essential 
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

# Build the image with version-specific tags
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t shitspotter:${REPO_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION} \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg UV_VERSION=$UV_VERSION \
    --build-arg REPO_GIT_HASH=$REPO_GIT_HASH \
    -f ./dockerfiles/shitspotter.dockerfile .

# Add concise tags for easier reuse
export IMAGE_QUALNAME=shitspotter:${REPO_GIT_HASH}-uv${UV_VERSION}-python${PYTHON_VERSION}
export NAME1=shitspotter:latest-uv${UV_VERSION}-python${PYTHON_VERSION}
export NAME2=shitspotter:latest-python${PYTHON_VERSION}
export NAME3=shitspotter:latest
docker tag $IMAGE_QUALNAME $NAME1
docker tag $IMAGE_QUALNAME $NAME2
docker tag $IMAGE_QUALNAME $NAME3

# Verify that GPUs are visible and that each shitspotter command works
docker run --gpus=all -it shitspotter:latest nvidia-smi

# Start a shell and run any custom tests
# (TODO: show how to replicate experiments)
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
