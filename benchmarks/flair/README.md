# Federated Learning Annotated Image Repository (FLAIR): A large labelled image dataset for benchmarking in federated learning

FLAIR is a large dataset of images that captures a number of characteristics encountered in federated learning and privacy-preserving ML tasks. 
This dataset comprises approximately 430,000 images from 51,000 Flickr users, which will better reflect federated learning problems arising in practice, and it is being released to aid research in the field.

![alt text](images/FLAIR_sample.jpeg)


## Setup environment

Same as the [default setup](../README.md).

## Download and preprocess FLAIR dataset

```
python -m dataset.flair.download_preprocess --output_file data/flair/flair_federated.hdf5
```

## Run benchmarks

Benchmarks for FLAIR are only available in PyTorch at this time.

FLAIR IID no DP:
```
python -m flair.train --args_config flair/configs/baseline.yaml --dataset flair_iid
```
FLAIR non-IID no DP:
```
python -m flair.train --args_config flair/configs/baseline.yaml
```
FLAIR IID central DP:
```
python -m flair.train --args_config flair/configs/baseline.yaml --dataset flair_iid --central_privacy_mechanism gaussian_moments_accountant 
```
FLAIR non-IID no DP:
```
python -m flair.train --args_config flair/configs/baseline.yaml --central_privacy_mechanism gaussian_moments_accountant 
```

## Image Labels
These images have been annotated by humans and assigned labels from a taxonomy of more than 1,600 fine-grained labels. 
All main subjects present in the images have been labeled, so images may have multiple labels. 
The taxonomy is hierarchical where the fine-grained labels can be mapped to 17 coarse-grained categories.
The dataset includes both fine-grained and coarse-grained labels so researchers can vary the complexity of a machine learning task.

## User Labels and their use for Federated Learning
We have used image metadata to extract artist names/IDs for the purposes of creating user datasets for federated learning. 
While optimization algorithms for machine learning are often designed under the assumption that each example is an independent sample from the distribution, federated learning applications deviate from this assumption in a few different ways that are reflected in our user-annotated examples. 
Different users differ in the number of images they have, as well as the number of classes represented in their image collection. 
Further, images of the same class but taken by different users are likely to have some distribution shift. 
These properties of the dataset better reflect federated learning applications, and we expect that benchmark tasks on this dataset will benefit from algorithms designed to handle such data heterogeneity.

## Dataset split
We include a standard train/val/test split.
The partition is based on user ids with ratio 8:1:1, i.e. train, val and test sets have disjoint users.
Below are the numbers for each partition:

| Partition        | Train   | Val    | Test   |
| ---------------- | ------- | ------ | ------ |
| Number of users  | 41,131  | 5,141  | 5,142  |
| Number of images | 345,879 | 39,239 | 43,960 |


## Open-sourced Version
FLAIR paper and benchmarks are published at NeurIPS 2022.
* Paper link: https://machinelearning.apple.com/research/flair
* Benchmark link (implemented with TensorFlow Federated): https://github.com/apple/ml-flair
