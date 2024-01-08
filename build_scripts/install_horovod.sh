#!/bin/bash

# Copyright © 2023-2024 Apple Inc.

# Install dependencies through setuptools.

set -o nounset -o pipefail -o errexit
set -x

echo '====== Starting install_horovod.sh'
date

declare -i false=0 true=1

# Default install is both TF and PyTorch and 
# don't install required Linux packages.
INSTALL_TF=${1:-true}
INSTALL_PYTORCH=${2:-true}
INSTALL_PKGS=${3:-false}

if "$INSTALL_PKGS"; then
    # Use sudo only if needed.
    # Our docker image does not have sudo because already running as root.
    if command -v sudo &>/dev/null && [ "$(id -u)" -ne 0 ]; then
        SUDO=sudo
    else
        SUDO=
    fi

    # Install Horovod dependencies.
    while true; do
        # This can fail sometimes when run with tox because apt-get can't be run in
        # parallel. The solution here is to simply retry until it works.
        if $SUDO apt-get update && $SUDO apt-get install -y \
            cmake \
            openmpi-bin \
            openmpi-common \
            libopenmpi-dev \
            mpich \
            libstdc++6 \
            libcupti-dev; then
          break
        fi
        sleep 10
    done
fi


if (($INSTALL_TF && $INSTALL_PYTORCH)); then
    # Install Horovod for both PyTorch and TF.
    HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_GLOO=1 python -m pip install \
        --progress-bar off --no-cache-dir horovod[tensorflow,pytorch]

elif "$INSTALL_TF"; then
    # Install Horovod for any TF version.
    HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_GLOO=1 python -m pip install \
        --progress-bar off --no-cache-dir horovod[tensorflow]

elif "$INSTALL_PYTORCH"; then
    # Install Horovod for PyTorch.
    HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_GLOO=1 python -m pip install \
        --progress-bar off --no-cache-dir horovod[pytorch]
fi


date
echo '====== Finished install_horovod.sh'
