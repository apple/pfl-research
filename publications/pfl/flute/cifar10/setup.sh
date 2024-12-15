#!/bin/bash
#conda install -y python=3.9
nvidia-smi -c 0

conda install -c conda-forge mpi4py -y --force-reinstall
(
  git clone git@github.com:grananqvist/msrflute.git -b benchmark || true
  cd ./msrflute
  python -m pip install -r requirements.txt
)
python -m pip install torchvision==0.14.0+cu117 torch==1.13.0+cu117 --find-links https://download.pytorch.org/whl/torch_stable.html

# Code needs to be copied over to
# flute experiments to be able to run.
apt-get update && apt-get install rsync -y
rsync -av --exclude='msrflute' ./ ./msrflute/experiments/cifar10

mkdir ./data/
# download pickle versions using pfl-research instead.
#curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o ./data/cifar-10-python.tar.gz 
#(
#    cd ./data
#    tar xzf cifar-10-python.tar.gz
#)

