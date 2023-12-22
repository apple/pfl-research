# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

from typing import Callable
import h5py
import numpy as np

from pfl.data import FederatedDataset, ArtificialFederatedDataset
from pfl.data.dataset import Dataset
from pfl.data.sampling import get_user_sampler, get_data_sampler

from .common import (get_multi_hot_targets, get_label_mapping,
                     get_user_num_images)


def make_federated_dataset(
        hdf5_path: str,
        partition: str,
        use_fine_grained_labels: bool,
        max_num_user_images: int,
        numpy_to_tensor: Callable = lambda x: x) -> FederatedDataset:
    """
    Create federated dataset from the flair dataset, to use in simulations.
    The federated dataset samples user datasets. A user dataset is
    made from data points of one user.

    :param hdf5_path:
        A h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :param max_num_user_images:
        Maximum number of images each user can have.
    :param numpy_to_tensor:
        Function that convert numpy array to ML framework tensor.
    :return:
        Federated dataset from the HDF5 data file.
    """
    num_classes = len(get_label_mapping(hdf5_path, use_fine_grained_labels))
    user_num_images = get_user_num_images(hdf5_path, partition)
    user_ids = sorted(list(user_num_images.keys()))
    sampler = get_user_sampler('random', user_ids)

    def make_dataset_fn(user_id):
        with h5py.File(hdf5_path, 'r') as h5:
            inputs = np.array(h5[f'/{partition}/{user_id}/images'])
            targets = get_multi_hot_targets((len(inputs), num_classes), h5,
                                            partition, user_id,
                                            use_fine_grained_labels)
            user_slice = slice(0, max_num_user_images)
            inputs, targets = inputs[user_slice], targets[user_slice]
            data_order = np.random.permutation(len(inputs))
            inputs = numpy_to_tensor(inputs[data_order])
            targets = numpy_to_tensor(targets[data_order])
        return Dataset(
            raw_data=[inputs, targets],
            train_kwargs={"eval": False},
            eval_kwargs={"eval": True})

    return FederatedDataset(
        make_dataset_fn, sampler, user_id_to_weight=user_num_images)


def make_artificial_federated_dataset(
        hdf5_path: str,
        partition: str,
        use_fine_grained_labels: bool,
        user_dataset_len_sampler: Callable,
        numpy_to_tensor: Callable = lambda x: x) -> ArtificialFederatedDataset:
    """
    Create artificial I.I.D. federated dataset from the flair dataset,
    to use in simulations. I.I.D. dataset is created by sampling data from
    all data points and ignoring the existing user IDs.

    :param hdf5_path:
        A h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :param user_dataset_len_sampler:
        A callable that should sample the dataset length of a user, i.e.
        `callable() -> dataset_length`.
    :param numpy_to_tensor:
        Function that convert numpy array to ML framework tensor.
    :return:
        Artificial I.I.D. federated dataset from the HDF5 data file.
    """
    num_classes = len(get_label_mapping(hdf5_path, use_fine_grained_labels))
    user_num_images = get_user_num_images(hdf5_path, partition)
    user_image_id = []
    for user_id in sorted(user_num_images.keys()):
        num_images = user_num_images[user_id]
        for image_id in range(num_images):
            user_image_id.append((user_id, image_id))

    def make_dataset_fn(indices):
        inputs, targets = [], []
        with h5py.File(hdf5_path, 'r') as h5:
            for index in indices:
                user_id, image_id = user_image_id[index]
                image = h5[f'/{partition}/{user_id}/images'][image_id]
                targets_shape = (user_num_images[user_id], num_classes)
                target = get_multi_hot_targets(
                    targets_shape, h5, partition, user_id,
                    use_fine_grained_labels)[image_id]
                inputs.append(np.expand_dims(image, axis=0))
                targets.append(np.expand_dims(target, axis=0))
        inputs = numpy_to_tensor(np.vstack(inputs))
        targets = numpy_to_tensor(np.vstack(targets))
        return Dataset(
            raw_data=[inputs, targets],
            train_kwargs={"eval": False},
            eval_kwargs={"eval": True})

    data_sampler = get_data_sampler('random', len(user_image_id))
    return ArtificialFederatedDataset(make_dataset_fn, data_sampler,
                                      user_dataset_len_sampler)
