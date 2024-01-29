# Copyright © 2023-2024 Apple Inc.
import os
import pickle
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from pfl.data import Dataset, FederatedDataset
from pfl.data.partition import partition_by_dirichlet_class_distribution
from pfl.data.sampling import get_data_sampler, get_user_sampler


def load_and_preprocess(pickle_file_path: str,
                        channel_means: Optional[np.ndarray] = None,
                        channel_stddevs: Optional[np.ndarray] = None,
                        exclude_classes=None):
    images, labels = pickle.load(open(pickle_file_path, 'rb'))
    images = images.astype(np.float32)

    # Normalize per-channel.
    if channel_means is None:
        channel_means = images.mean(axis=(0, 1, 2), dtype='float64')
    if channel_stddevs is None:
        channel_stddevs = images.std(axis=(0, 1, 2), dtype='float64')
    images = (images - channel_means) / channel_stddevs

    if exclude_classes is not None:
        for exclude_class in exclude_classes:
            mask = (labels != exclude_class).reshape(-1)
            labels = labels[mask]
            images = images[mask]

    return images, labels, channel_means, channel_stddevs


def make_federated_dataset(images: np.ndarray,
                           labels: np.ndarray,
                           user_dataset_len_sampler: Callable,
                           numpy_to_tensor: Callable = lambda x: x,
                           alpha: float = 0.1) -> FederatedDataset:
    """
    Create a federated dataset from the CIFAR10 dataset.

    Users are created as proposed by Hsu et al. https://arxiv.org/abs/1909.06335,
    by sampling each user's class distribution from Dir(0.1).
    """
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    users_to_indices = partition_by_dirichlet_class_distribution(
        labels, alpha, user_dataset_len_sampler)
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)
    users_to_data = [(images[indices], labels[indices])
                     for indices in users_to_indices]

    user_sampler = get_user_sampler('random', range(len(users_to_data)))
    return FederatedDataset.from_slices(users_to_data, user_sampler)


def make_iid_federated_dataset(
        images: np.ndarray,
        labels: np.ndarray,
        user_dataset_len_sampler: Callable,
        numpy_to_tensor: Callable = lambda x: x) -> FederatedDataset:
    """
    Create a federated dataset with IID users from the CIFAR10 dataset.

    Users are created by first sampling the dataset length from
    ``user_dataset_len_sampler`` and then sampling the datapoints IID.
    """
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)
    start_ix = 0
    users_to_data = {}
    while True:
        dataset_len = user_dataset_len_sampler()
        user_slice = slice(start_ix, start_ix + dataset_len)
        users_to_data[len(users_to_data)] = (images[user_slice],
                                             labels[user_slice])
        start_ix += dataset_len
        if start_ix >= len(images):
            break

    user_sampler = get_user_sampler('random', range(len(users_to_data)))
    return FederatedDataset.from_slices(users_to_data, user_sampler)


def make_central_dataset(images: np.ndarray, labels: np.ndarray) -> Dataset:
    """
    Create central dataset (represented as a ``Dataset``) from CIFAR10.
    This ``Dataset`` can be used for central evaluation with
    ``CentralEvaluationCallback``.
    """
    return Dataset(raw_data=[images, labels])


def make_cifar10_datasets(
    data_dir: str,
    user_dataset_len_sampler: Callable,
    numpy_to_tensor: Callable,
    alpha: float = 0.1,
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and val ``FederatedDataset`` as well as a
    central dataset from the CIFAR10 dataset.

    Here, users are created as proposed by Hsu et al. https://arxiv.org/abs/1909.06335,
    by sampling each user's class distribution from Dir(0.1).
    """
    train_images, train_labels, channel_means, channel_stddevs = (
        load_and_preprocess(os.path.join(data_dir, 'cifar10_train.p')))
    val_images, val_labels, _, _ = load_and_preprocess(
        os.path.join(data_dir, 'cifar10_test.p'), channel_means,
        channel_stddevs)

    # create artificial federated training and val datasets
    # from central training and val data.
    training_federated_dataset = make_federated_dataset(
        train_images, train_labels, user_dataset_len_sampler, numpy_to_tensor,
        alpha)
    val_federated_dataset = make_federated_dataset(val_images, val_labels,
                                                   user_dataset_len_sampler,
                                                   numpy_to_tensor, alpha)
    central_data = make_central_dataset(val_images, val_labels)

    return training_federated_dataset, val_federated_dataset, central_data, {}


def make_cifar10_iid_datasets(
    data_dir: str, user_dataset_len_sampler: Callable,
    numpy_to_tensor: Callable
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and val ``FederatedDataset`` with IID users as well as a
    central dataset from the CIFAR10 dataset.

    Here, infinite users are created by continously sampling datapoints iid
    from full dataset whenever next user is requested.
    """
    train_images, train_labels, channel_means, channel_stddevs = (
        load_and_preprocess(os.path.join(data_dir, 'cifar10_train.p')))
    val_images, val_labels, _, _ = load_and_preprocess(
        os.path.join(data_dir, 'cifar10_test.p'), channel_means,
        channel_stddevs)

    # create artificial federated training and val datasets
    # from central training and val data.
    training_federated_dataset = make_iid_federated_dataset(
        train_images, train_labels, user_dataset_len_sampler, numpy_to_tensor)
    val_federated_dataset = make_iid_federated_dataset(
        val_images, val_labels, user_dataset_len_sampler, numpy_to_tensor)
    central_data = make_central_dataset(val_images, val_labels)

    return training_federated_dataset, val_federated_dataset, central_data, {}
