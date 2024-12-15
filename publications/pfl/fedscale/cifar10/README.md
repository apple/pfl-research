# CIFAR10 benchmark for FedScale

We used 1 NVIDIA A100 GPU (40GB) and 10CPU for this benchmark.
For more information, see Section 4.1, C.5 and D.1 in [pfl-research paper](https://arxiv.org/abs/2404.06430).

## Installation

````
./setup.sh
```

## Run

```
export LOG_DIR=./logs
export FEDSCALE_HOME=.
python FedScale/docker/driver.py start fedscale_config.yaml
```

## Notes

Central evaluation is not added in this benchmark because of time constraints.
Adding this slow down simulation even more, but FedScale is already the slowest on the CIFAR10 benchmark.
