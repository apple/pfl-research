# Change Log

## Unreleased

### Breaking change!

* Removed Horovod support (#130).
* Upgrading minimum dependencies in `docs` and `dev` install-extras (#130).

### New features

* 

### Tasks completed

* Support up to `torch==2.6.0` and `tensorflow==2.17.0` (#130).

### Bug fixes

* 


## v0.4.0

### Breaking change!

* Callbacks have been split up into sub-modules under the module `pfl.callback` (#124).

### Tasks completed

* Added `packaging` as a dependency, which was missing (#121).


## v0.3.1

### Tasks completed

* Allow `Saveable` to checkpoint itself on-demand (#115).


## v0.3.0

### New features

* Implemented `MLXModel` to support training with MLX (#80).
* Implemented Horovod-compatible `Barrier` class (#100).
* Implemented `JointMechanism` to enable applying different DP noise on subsets of the statistics (#105).
* Implemented `JointPrivacyAccountant` for joint privacy accounting in the case of using `JointMechanism` (#111).
* Implemented policy-based model checkpointing callbacks (#106).

### Tasks completed

* Added MLX [image classification example](https://github.com/apple/pfl-research/blob/develop/benchmarks/image_classification/mlx/train.py) (#80).
* Added MLX [language model example](https://github.com/apple/pfl-research/blob/develop/benchmarks/lm/mlx/train.py) (#82).
* Added [notebook tutorial](https://github.com/apple/pfl-research/blob/develop/tutorials/Introduction%20to%20Federated%20Learning%20with%20CIFAR10%20and%20MLX.ipynb) for training with MLX (#81).
* Updated notebooks to work on Colab (#95).
* Added CITATION.cff (#99).
* Don't hardcode for CUDA 11.8 (#108).

### Bug fixes

* Fixed bug in `PyTorchFederatedDataset` where it sometimes hang (#98).
* Fix `PyTorchSeedScope` for non-CPU random states (#100).
* Respect `CUDA_VISIBLE_DEVICES` if it is set for process (#100).
* Fixed algorithm states not working properly when restored from `Saveable` (#103).
* Fixed edge case for having `HyperParams` inside `HyperParams` (#112).


## v0.2.0

### Breaking change!

* `EMMGMMHyperParams` is renamed to `EMGMMHyperParams` (#55)

### New features

* Return local metadata from model training to algorithm (#71).

### Tasks completed

* Update FLAIR preprocessing script to download dataset from HuggingFace, available at https://huggingface.co/datasets/apple/flair (#72).
* Update LLM Benchmark Configs (#63).
* New improved worker scheduling in distributed simulations. Speeds up FLAIR benchmark by 19% (#73).
* Don't pin PyTorch version to 2.0.1 (#69).
* Move `--noise_cohort_size` to `add_mechanism_arguments` (#70).

### Bug fixes

* 


## v0.1.0

2024-03-01

* Initial release!
