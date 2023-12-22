# Single class image classification

## Setup environment

1. Same as the [default setup](../README.md).

## Download and preprocess CIFAR10 dataset

Downloads CIFAR10 dataset from the original source and preprocess into pickles.

```
python -m dataset.cifar10.download_preprocess --output_dir data/cifar10
```

## Run benchmarks

Benchmarks are implemented for both PyTorch and TensorFlow, available in `image_classification/pytorch` and `image_classification/tf` respectively. The commands are the same.

```
# pytorch or tf
framework=pytorch
```

CIFAR10 IID no DP:
```
python image_classification/${framework}/train.py --args_config image_classification/configs/baseline.yaml
```
CIFAR10 non-IID no DP:
```
python image_classification/${framework}/train.py --args_config image_classification/configs/baseline.yaml --dataset cifar10 --central_num_iterations 3000
```
CIFAR10 IID central DP:
```
python image_classification/${framework}/train.py --args_config image_classification/configs/baseline.yaml --central_privacy_mechanism gaussian_moments_accountant --central_num_iterations 3000
```
CIFAR10 non-IID no DP:
```
python image_classification/${framework}/train.py --args_config image_classification/configs/baseline.yaml --dataset cifar10 --central_privacy_mechanism gaussian_moments_accountant --central_num_iterations 3000
```
