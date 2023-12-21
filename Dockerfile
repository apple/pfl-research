# syntax=docker/dockerfile:1
# See here for CUDA versions supported by particular PyTorch versions:
# https://download.pytorch.org/whl/torch_stable.html
# See here for CUDA versions supported by particular TensorFlow versions:
# https://www.tensorflow.org/install/source#gpu

# Can be `runtime` or `devel`
ARG RUNTIME_TYPE=runtime
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-${RUNTIME_TYPE}-ubuntu22.04 as base

ENV LANG C.UTF-8

RUN apt-get update && apt-get install --quiet -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /pfl/

# Prevent dpkg prompts on installation.
ENV DEBIAN_FRONTEND=noninteractive
# Packages needed to execute rest of build.
RUN apt-get update && apt-get install --quiet -y --no-install-recommends \
    git \
    openssh-client \
    openssh-server \
    openssl \
    lsof \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Anaconda.
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install default conda Python env.
ENV PATH /opt/conda/bin:$PATH
RUN conda install -y python=3.10

COPY ./pyproject.toml ./
COPY ./poetry.lock ./
COPY ./pfl ./pfl
COPY ./build_scripts ./build_scripts
COPY ./README.md ./README.md

# Install Poetry.
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH /root/.local/bin:$PATH

# Install dependencies excluding deep learning frameworks.
RUN poetry config virtualenvs.create false
RUN poetry build && python -m pip install dist/*.whl
RUN poetry install

# docker build -f Dockerfile -t docker.apple.com/dodml/oss-pfl:latest-tf . --target tf
FROM base as tf
WORKDIR /pfl/
COPY --from=base /pfl /pfl
RUN poetry install -E tf -E trees
RUN poetry run ./build_scripts/install_horovod.sh true false true
RUN rm -rf /pfl

# docker build -f Dockerfile -t docker.apple.com/dodml/oss-pfl:latest-pytorch . --build-arg CUDA_VERSION=11.8.0 --build-arg RUNTIME_TYPE=devel --target pytorch
FROM base as pytorch
WORKDIR /pfl/
COPY --from=base /pfl /pfl
RUN poetry install -E pytorch -E trees
RUN poetry run ./build_scripts/install_horovod.sh false true true
RUN rm -rf /pfl

# docker build -f Dockerfile -t docker.apple.com/dodml/oss-pfl:latest-ci . --build-arg CUDA_VERSION=11.8.0 --build-arg RUNTIME_TYPE=devel --target ci
FROM base as ci
WORKDIR /pfl/
COPY --from=base /pfl /pfl
RUN poetry install -E tf -E pytorch -E trees
RUN poetry run ./build_scripts/install_horovod.sh true true true
RUN rm -rf /pfl
