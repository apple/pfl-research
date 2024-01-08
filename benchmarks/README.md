# PFL benchmarks

## Setup environment

Prerequisite 1: you need to have [poetry](https://python-poetry.org/docs/#installation) installed.
```
curl -sSL https://install.python-poetry.org | python3 -
```

Prerequisite 2: you need to have compatible Python version available in poetry.
You only need to have this activated the first time you install the poetry environment.
```
conda create -n py310 python=3.10
conda activate py310
```

Install environment:

```
git clone git@github.com:apple/pfl-research.git
cd pfl-research/benchmarks/
# If you have the new Python 3.10 environment active, it should be cloned.
poetry env use `which python`
# Install to run tf, pytorch and tests
poetry install -E pytorch -E tf
# Activate environment
poetry shell
# Add root directory to `PYTHONPATH` such that the utility modules can be imported
export PYTHONPATH=`pwd`:$PYTHONPATH
```

This default setup should enable you to run any of the official benchmarks.

## Quickstart

1. Complete setup as above.
2. Download CIFAR10 data:
```
python -m dataset.cifar10.download_preprocess --output_dir data/cifar10
```
3. Train a small CNN on CIFAR10 IID data:
```
python image_classification/pytorch/train.py --args_config image_classification/configs/baseline.yaml
```

## Official benchmarks

There are multiple official benchmarks for `pfl` to simulate various scenarios, split into categories:
* [image_classification](./image_classification) - train small CNN on CIFAR10.
* [lm](./lm) - train transformer model on StackOverflow
* [flair](./flair) - train ResNet18 on [FLAIR](https://proceedings.neurips.cc/paper_files/paper/2022/file/f64e55d03e2fe61aa4114e49cb654acb-Paper-Datasets_and_Benchmarks.pdf) dataset.

## Run distributed simulations

Each benchmark can run in distributed mode with multiple cores, GPUs and machines.
See the [distributed simulation guide](https://pages.github.apple.com/apple/pfl-research/tutorials/simulation_distributed.html) on how it works.
In summary, to quickly get started running distributed simulations:
1. Install [Horovod](https://horovod.readthedocs.io/en/stable/install_include.html). We have a helper script [here](https://github.com/apple/pfl-research/blob/main/build_scripts/install_horovod.sh).
2. Invoke your Python script with the `horovodrun` command. E.g. to run the same CIFAR10 training as described above in the quickstart, but train with 2 processes on the same machine, the command will look like this:
```
horovodrun --gloo -np 2 -H localhost:2 python image_classification/pytorch/train.py --args_config image_classification/configs/baseline.yaml
```
If you have 2 GPUs on the machine, each process will be allocated 1 GPU. If you only have 1 GPU, both processes will share the GPU. Sharing GPU can still result in speedup depending on the use case because of the inevitable overhead of FL.

## Other Poetry commands

Alternatively, you can install a subset of dependencies:

```
# Install to run tf examples
poetry install -E tf --no-dev
# Install to run pytorch examples
poetry install -E pytorch --no-dev
# Install to run tf, pytorch and tests
poetry install -E pytorch -E tf
```
