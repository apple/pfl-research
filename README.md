# `pfl`: Python framework for Private Federated Learning simulations

[![GitHub License](https://img.shields.io/github/license/apple/pfl-research)](https://github.com/apple/pfl-research/blob/main/LICENSE)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/apple/pfl-research/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/apple/pfl-research/tree/main)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pfl)](https://github.com/apple/pfl-research/blob/main/pyproject.toml#L18)

**Documentation website:** https://apple.github.io/pfl-research

`pfl` is a Python framework developed at Apple to empower researchers to run efficient simulations with privacy-preserving federated learning (FL) and disseminate the results of their research in FL. We are a team comprising engineering and research expertise, and we encourage researchers to publish their papers, with this code, with confidence.

The framework is `not` intended to be used for third-party FL deployments but the results of the simulations can be tremendously useful in actual FL deployments.
We hope that `pfl` will promote open research in FL and its effective dissemination.

``pfl`` provides several useful features, including the following:

* Get started quickly trying out PFL for your use case with your existing model and data.
* Iterate quickly with fast simulations utilizing multiple levels of distributed training (multiple processes, GPUs and machines).
* Flexibility and expressiveness - when a researcher has a PFL idea to try, ``pfl`` has flexible APIs to express these ideas.
* Scalable simulations for large experiments with state-of-the-art algorithms and models.
* Support both PyTorch and TensorFlow.
* Unified benchmarks for datasets that have been vetted for both PyTorch and TensorFlow.
* Support other models in addition to neural networks, e.g. GBDTs. Switching between types of models is seamless.
* Tight integration with privacy features, including common mechanisms for local and central differential privacy.

Results from benchmarks are maintained in [this Weights & Biases report](https://api.wandb.ai/links/pfl/5scd5f66).

## Installation

Installation instructions can be found [here](http://apple.github.io/pfl-research/installation.html).
`pfl` is available on PyPI and a full installation be done with pip:

```
pip install 'pfl[tf,pytorch,trees]'
```

## Getting started - tutorial notebooks

To try out `pfl` immediately without installation, we provide several colab notebooks for learning the different components in `pfl` hands-on.

[![Introduction to Federated Learning with CIFAR10 and TensorFlow](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apple/pfl-research/blob/develop/tutorials/Introduction%20to%20Federated%20Learning%20with%20CIFAR10%20and%20TensorFlow.ipynb)
[![Introduction to PFL research with FLAIR and PyTorch](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apple/pfl-research/blob/develop/tutorials/Introduction%20to%20PFL%20research%20with%20FLAIR.ipynb)
[![Introduction to Differential Privacy (DP) with Federated Learning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apple/pfl-research/blob/develop/tutorials/Introduction%20to%20Differential%20Privacy%20with%20Federated%20Learning.ipynb)
[![Creating Federated Dataset for PFL Experiment](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apple/pfl-research/blob/develop/tutorials/Creating%20Federated%20Dataset%20for%20PFL%20Experiment.ipynb)

We also support MLX!
* [https://github.com/apple/pfl-research/blob/develop/tutorials/Introduction%20to%20Federated%20Learning%20with%20CIFAR10%20and%20MLX.ipynb](https://github.com/apple/pfl-research/blob/develop/tutorials/Introduction%20to%20Federated%20Learning%20with%20CIFAR10%20and%20MLX.ipynb)
But you have to run this notebook locally on your Apple silicon, see all Jupyter notebooks available [here](https://github.com/apple/pfl-research/tree/develop/tutorials).

## Getting started - benchmarks

`pfl` aims to streamline the benchmarking process of testing hypotheses in the Federated Learning paradigm. The official benchmarks are available in the [benchmarks](./benchmarks) directory, using a variety of realistic dataset-model combinations with and without differential privacy (yes, we do also have CIFAR10).

**Copying these examples is a great starting point for doing your own research.**
[See the quickstart](./benchmarks#quickstart) on how to start converging a model on the simplest benchmark (CIFAR10) in just a few minutes.

## Contributing

Researchers are invited to contribute to the framework. Please, see [here](http://apple.github.io/pfl-research/support/contributing.html) for more details.

## Citing pfl-research

```
@software{pfl2024,
  author = {Filip Granqvist and Congzheng Song and Áine Cahill and Rogier van Dalen and Martin Pelikan and Yi Sheng Chan and Xiaojun Feng and Natarajan Krishnaswami and Mona Chitnis and Vojta Jina},
  title = {{pfl}: simulation framework for accelerating research in Private Federated Learning},
  url = {https://github.com/apple/pfl-research},
  version = {0.0},
  year = {2024},
}
```
