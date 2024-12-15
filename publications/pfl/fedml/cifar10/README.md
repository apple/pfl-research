# CIFAR10 benchmark for FedML

We used 1 NVIDIA A100 GPU (40GB) and 10CPU for this benchmark.
For more information, see Section 4.1, C.5 and D.1 in [pfl-research paper](https://arxiv.org/abs/2404.06430).

## Installation
```
./setup.sh
```

## Run

```
# Make sure compare_utils is in PYTHONPATH
PYTHONPATH=.:FedML/python/:../../ python -m main --cf fedml_config.yaml
```

## Modifications

* Code was adopted from https://github.com/FedML-AI/FedML/tree/28cd33b2d64fda1533c1bac6109c14f13c1012f7/python/examples/federate/simulation/sp_fedavg_cifar10_resnet56_example
* Disabled random crop and random horizontal flip in CIFAR10 data loader.
* For some reason, each user has all the test data, so we modified to only evaluate 1 user as central evaluation.
