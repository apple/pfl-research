import os
import pickle
from typing import Callable, List, Optional, Tuple

import numpy as np

from pfl.data import ArtificialFederatedDataset, FederatedDataset, FederatedDatasetBase
from pfl.data.dataset import Dataset
from pfl.data.sampling import get_data_sampler, get_user_sampler

from .mixture_dataset import ArtificialFederatedDatasetMixture, partition_by_dirichlet_mixture_class_distribution
from .sampler import DirichletDataSampler


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


def make_artificial_federated_dataset(
        images: np.ndarray,
        labels: np.ndarray,
        user_dataset_len_samplers: List[Callable],
        phi: np.ndarray = None,
        alphas: np.ndarray = None,
        numpy_to_tensor: Callable = lambda x: x) -> FederatedDatasetBase:
    """
    Create an artificial federated dataset from the CIFAR10 dataset
    """
    data = [numpy_to_tensor(images), numpy_to_tensor(labels)]
    if alphas is not None:
        data_samplers = [
            DirichletDataSampler(alpha, labels) for alpha in alphas
        ]
        return ArtificialFederatedDatasetMixture.from_slices(
            phi, data, data_samplers, user_dataset_len_samplers)
    else:
        data_sampler = get_data_sampler('random', len(labels))
        return ArtificialFederatedDataset.from_slices(
            data, data_sampler, user_dataset_len_samplers[0])


def make_federated_dataset(
        images: np.ndarray,
        labels: np.ndarray,
        user_dataset_len_samplers: List[Callable],
        phi: np.ndarray = None,
        alphas: np.ndarray = None,
        numpy_to_tensor: Callable = lambda x: x) -> FederatedDataset:
    """
    Create a federated dataset from the CIFAR10 dataset.
    """

    if alphas is not None:
        user_idxs = partition_by_dirichlet_mixture_class_distribution(
            labels, phi, alphas, user_dataset_len_samplers)
    else:
        all_idxs = np.arange(len(labels)).astype(int)
        np.random.shuffle(all_idxs)
        user_idxs = []
        while True:
            n = user_dataset_len_samplers[0]()
            if len(all_idxs) >= n:
                user_idxs.append(all_idxs[:n])
                all_idxs = all_idxs[n:]
            else:
                user_idxs.append(all_idxs)
                break

    user_sampler = get_user_sampler('random', list(range(len(user_idxs))))
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)

    data = {}
    for user_id in range(len(user_idxs)):
        data[user_id] = [
            images[user_idxs[user_id]], labels[user_idxs[user_id]]
        ]

    return FederatedDataset.from_slices(data, user_sampler)


def make_central_dataset(images: np.ndarray, labels: np.ndarray) -> Dataset:
    """
    Create central dataset (represented as a ``Dataset``) from CIFAR10.
    This ``Dataset`` can be used for central evaluation with
    ``CentralEvaluationCallback``.
    """
    return Dataset(raw_data=[images, labels])


def make_cifar10_datasets(
    dataset_type: str,
    data_dir: str,
    user_dataset_len_samplers: List[Callable],
    numpy_to_tensor: Callable,
    phi: np.ndarray = None,
    alphas: np.ndarray = None
) -> Tuple[FederatedDataset, FederatedDataset, Dataset]:
    """
    Create a train and val ``ArtificialFederatedDataset`` as well as a
    central dataset from the CIFAR10 dataset.

    The data files can be found at ``s3://pfl/data/cifar10/``.
    """
    train_images, train_labels, channel_means, channel_stddevs = (
        load_and_preprocess(os.path.join(data_dir, 'cifar10_train.p')))

    val_images, val_labels, _, _ = load_and_preprocess(
        os.path.join(data_dir, 'cifar10_test.p'), channel_means,
        channel_stddevs)

    # supports artificial federated dataset and federated dataset
    fed_dataset_fn = make_artificial_federated_dataset \
        if dataset_type == 'artificial_federated_dataset' \
        else make_federated_dataset

    # create federated training and val datasets
    # from central training and val data.
    training_federated_dataset = fed_dataset_fn(
        train_images,
        train_labels,
        user_dataset_len_samplers,
        phi=phi,
        alphas=alphas,
        numpy_to_tensor=numpy_to_tensor)
    val_federated_dataset = fed_dataset_fn(val_images,
                                           val_labels,
                                           user_dataset_len_samplers,
                                           phi=phi,
                                           alphas=alphas,
                                           numpy_to_tensor=numpy_to_tensor)
    central_val_data = make_central_dataset(val_images, val_labels)

    return training_federated_dataset, val_federated_dataset, central_val_data
