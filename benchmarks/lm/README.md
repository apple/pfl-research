# Federated language models

## Setup environment

1. Same as the [default setup](../README.md).
2. To download dataset, you need to have `tensorflow_federated` installed. Use either a different environment to install this package and prepare dataset, or re-run step (1) after dataset is processed to maintain the correct environment for running the benchmarks.
```
pip install tensorflow_federated
```

> :warning: As of 2023-10-20, `tensorflow-federated` is not installable on MacOS M1, hence preprocessing the data will only work on a Linux machine.

## Download and preprocess StackOverflow dataset

The script has arguments for vocabulary size and maximum sequence length.
The official benchmarks use the default `vocab_size=10000` and `max_sequence_length=20`.

```
python -m dataset.stackoverflow.download_preprocess --output_dir data/stackoverflow
```

## Run benchmarks

Benchmarks are implemented for both PyTorch and TensorFlow, available in `lm/pytorch` and `lm/tf` respectively. The commands are the same.

```
# pytorch or tf
framework=pytorch
```

StackOverflow non-IID no DP:
```
python -m lm.${framework}.train --args_config lm/configs/baseline.yaml
```

StackOverflow non-IID Central DP:
```
python -m lm.${framework}.train --args_config lm/configs/baseline.yaml --central_privacy_mechanism gaussian_moments_accountant
```
