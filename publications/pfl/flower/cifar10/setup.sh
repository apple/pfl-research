#!/bin/bash
#conda install -y python=3.9
python -m pip install poetry
python -m poetry config virtualenvs.create false

(
  git clone git@github.com:adap/flower.git -b v1.7.0 || true
  cd flower/baselines/flwr_baselines/
  python -m poetry install
)

#python -m pip install "grpcio==1.51.0" ray==2.9
python -m pip install ray==2.9 flwr==1.7.0
python -m pip install torchvision==0.14.0+cu117 torch==1.13.0+cu117 --find-links https://download.pytorch.org/whl/torch_stable.html

nvidia-smi -c 0

mkdir data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz -o ./data/cifar-10-python.tar.gz 
(
    cd data
    tar xzf cifar-10-python.tar.gz
)

