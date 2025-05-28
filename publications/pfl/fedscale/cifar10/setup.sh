#!/bin/bash

# Copyright © 2024 Apple Inc.
nvidia-smi -c 0

(
  git clone git@github.com:grananqvist/FedScale.git -b benchmark || true
  cd FedScale
  conda env create -f environment.yml
  source /opt/conda/etc/profile.d/conda.sh
  conda activate fedscale
  python -m pip install .
)

# Benchmark was performed with Cu117
#python -m pip install torchvision==0.14.0+cu117 torch==1.13.0+cu117 --find-links https://download.pytorch.org/whl/torch_stable.html
# No GPU
python -m pip install torchvision==0.14.0 torch==1.13.0 --find-links https://download.pytorch.org/whl/torch_stable.html

mkdir -p ./FedScale/benchmark/dataset/data/
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o ./FedScale/benchmark/dataset/data/cifar-10-python.tar.gz
(
    cd ./FedScale/benchmark/dataset/data/ 
    tar xzf cifar-10-python.tar.gz
)
