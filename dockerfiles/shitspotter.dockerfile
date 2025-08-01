# syntax=docker/dockerfile:1.5
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV PIP_ROOT_USER_ACTION=ignore

# Control which python version we are using
ARG PYTHON_VERSION=3.13

# Control the version of uv
ARG UV_VERSION=0.7.19

# ------------------------------------
# Step 1: Install System Prerequisites
# ------------------------------------
RUN <<EOF
#!/bin/bash
set -e
apt update -q
DEBIAN_FRONTEND=noninteractive apt install -q -y --no-install-recommends \
    bzip2 \
    rsync \
    tmux \
    fd-find jq htop tree \
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

# Set the shell to bash to auto-activate enviornments
SHELL ["/bin/bash", "-l", "-c"]

# ------------------
# Step 2: Install uv
# ------------------
# Here we take a few extra steps to pin to a verified version of the uv
# installer. This increases reproducibility and security against the main
# astral domain, but not against those linked in the main installer.
# The "normal" way to install the latest uv is:
# curl -LsSf https://astral.sh/uv/install.sh | bash
RUN <<EOF
#!/bin/bash
set -e
mkdir /bootstrap
cd /bootstrap
declare -A UV_INSTALL_KNOWN_HASHES=(
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
echo "$EXPECTED_SHA256  $DOWNLOAD_PATH" | sha256sum --check
# Run the install script
bash /bootstrap/uv-install-v${UV_VERSION}.sh
# Cleanup for smaller images
rm -rf /root/.cache/
EOF


# ------------------------------------------
# Step 3: Setup a Python virtual environment
# ------------------------------------------
# This step mirrors a normal virtualenv development environment inside the
# container, which can prevent subtle issues due when running as root inside
# containers. 
RUN <<EOF
#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
# Use uv to install the requested python version and seed the venv
uv venv "/root/venv$PYTHON_VERSION" --python=$PYTHON_VERSION --seed
BASHRC_CONTENTS='
# setup a user-like environment, even though we are root
export HOME="/root"
export PATH="$HOME/.local/bin:$PATH"
# Auto-activate the venv on login
source /root/venv'$PYTHON_VERSION'/bin/activate
'
# It is important to add the content to both so 
# subsequent run commands use the the context we setup here.
echo "$BASHRC_CONTENTS" >> $HOME/.bashrc
echo "$BASHRC_CONTENTS" >> $HOME/.profile
EOF


RUN mkdir -p /root/code/shitspotter

# Control the version of REPO (by default uses the current branch)
ARG REPO_GIT_HASH=HEAD

# ---------------------------------
# Step 4: Checkout and install REPO
# ---------------------------------
# Based on the state of the repo this copies the host .git data over and then
# checks out the extact version of REPO requested by REPO_GIT_HASH. It then
# performs a basic install of shitspotter into the virtual environment.

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



# -----------------------------------
# Step 5: Ensure venv auto-activation
# -----------------------------------
# This final steps ensures that commands the user provides to docker run
# will always run in in the context of the virtual environment. 
RUN  <<EOF
#!/bin/bash
set -e
# write the entrypoint script
echo '#!/bin/bash
set -e
# Build the escaped command string
cmd=""
for arg in "$@"; do
  # Use printf %q to properly escape each argument for bash
  cmd+=$(printf "%q " "$arg")
done
# Remove trailing space
cmd=${cmd% }
exec bash -lc "$cmd"
' > entrypoint.sh
chmod +x /entrypoint.sh
EOF

# Set the entrypoint to our script that activates the virtual enviornment first
ENTRYPOINT ["/entrypoint.sh"]

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

# Build REPO in a reproducible way.
DOCKER_BUILDKIT=1 docker build --progress=plain \
    -t shitspotter:$REPO_GIT_HASH-uv0.7.29-python3.11 \
    --build-arg PYTHON_VERSION=3.11 \
    --build-arg UV_VERSION=0.7.19 \
    --build-arg REPO_GIT_HASH=$REPO_GIT_HASH \
    -f ./dockerfiles/shitspotter.dockerfile .


# Add latest tags for convinience
docker tag shitspotter:$REPO_GIT_HASH-uv0.7.29-python3.11 shitspotter:latest-uv0.7.29-python3.11
docker tag shitspotter:$REPO_GIT_HASH-uv0.7.29-python3.11 shitspotter:latest

# Verify that GPUs are visible and that each shitspotter command works
docker run --gpus=all -it shitspotter:latest nvidia-smi

# Start a shell and run any custom tests
# (TODO: show how to replicate experiments)
docker run --gpus=all -it shitspotter:latest bash

' > /dev/null

EOF
