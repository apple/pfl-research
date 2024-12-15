# CIFAR10 benchmark for Flower

We used 1 NVIDIA A100 GPU (40GB) and 10CPU for this benchmark.
For more information, see Section 4.1, C.5 and D.1 in [pfl-research paper](https://arxiv.org/abs/2404.06430).

This code is originally from the fedopt baseline https://github.com/adap/flower/tree/main/baselines/flwr_baselines

## Installation

```
./setup.sh
```

## Run

```
PYTHONPATH=../../:. python -m main --config-path conf/cifar10
```

## Notes

to run locally on m1, I had to do:
```
GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation" pip install grpcio --no-binary :all:
GRPC_PYTHON_LDFLAGS=" -framework CoreFoundation" pip install grpcio-tools --no-binary :all:
```
