#!/bin/bash
# Copyright © 2024 Apple Inc.
set -ex

(
    git clone git@github.com:grananqvist/FedML.git || true
    cd FedML/python
    python -m pip install .
)

# Benchmark was performed with Cu118
python -m pip install torchvision==0.15.2+cu118 torch==2.0.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html
# No GPU
#python -m pip install torchvision==0.15.2 torch==2.0.1 --find-links https://download.pytorch.org/whl/torch_stable.html
