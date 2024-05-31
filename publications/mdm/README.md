# Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomials (MDMs)

This software project accompanies the research paper, "Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomials".

Mixture-of-Dirichlet-Multinomial (MDM) models allow one to model heterogeneous federated datasets, and such MDM models can be trained with privacy preserving federated learning.

## Documentation

This repo contains the code to run all experiments in the paper "Improved Modelling of Federated Datasets using Mixtures-of-Dirichlet-Multinomials", and to process the results to produce the plots shown in the paper are available in the `mdm-paper` directory on this fork of the `pfl-research` framework, for running simulations using Private Federated Learning.

The structure of the `mdm` repo is:
- `mdm/`: This directory contains the algorithmic code, implementing the MDM model algorithm in the pfl-research framework. 
- `mdm_paper/`: This directory contains subdirectories: `training/` contains the python training scripts to run inference of MDM parameters on the CIFAR-10 and FEMNIST datasets; `notebooks/` contains Jupyter notebooks used to visualise results and create plots shown in the paper.
- `mdm_utils/`: This directory contains utilities to help with training setup, e.g. argument parsers, dataset helper functions, etc.


## Setup for experiments

It is assumed you first follow the default setup for benchmarks in the pfl-research framework. The details to follow for this default setup are available [here](https://github.com/apple/pfl-research/blob/develop/benchmarks/README.md).

It is next assumed that you have the FEMNIST and CIFAR-10 datasets downloaded locally in directories data/femnist/ and data/cifar10/ respectively. To download the data and ensure it is preprocessed correctly, please follow the pfl-research instructions for data setup [here](https://github.com/apple/pfl-research/tree/develop/benchmarks/image_classification).

## Running experiments in paper
To run MDM parameter inference on CIFAR-10: ` bash publications/mdm/run_cifar10_mse_alpha_phi_experiments.sh`.

To run MDM parameter inference on FEMNIST, for the experiments in the original paper version where users are not split between server-side dataset and live dataset: `bash publications/mdm/run_femnist.sh`.

To run MDM parameter inference on FEMNIST as in the rebuttal, where the users are split between server-side dataset and live dataset: `bash publications/mdm/run_femnist_rebuttal.sh`.
