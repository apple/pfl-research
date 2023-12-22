# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import os
from typing import Any, Callable, Dict, Tuple, List

import h5py
import numpy as np

from pfl.data import FederatedDataset
from pfl.data.sampling import get_user_sampler
from pfl.data.dataset import Dataset


def _load_h5_into_dict(h5_file_path: str,
                       digits_only: bool,
                       numpy_to_tensor: Callable = lambda x: x
                       ) -> Dict[str, List[np.ndarray]]:
    """
    Load data into memory and create a mapping from user ids to that
    user's data.

    :returns:
        A dictionary mapping user ids to data. The data is a tuple
        `(pixels,labels)`, where `pixels` is a `BxWxHx1` `np.ndarray`
        (stacked images of a user) and `labels` is a `Bx1` vector of
        categorical labels.
    """
    user_id_to_data = {}
    with h5py.File(h5_file_path, "r") as f:
        for user, h5_group in f['examples'].items():
            images = np.expand_dims(h5_group['pixels'][()], axis=-1)
            labels = h5_group['label'][()]
            if digits_only:
                images = images[labels < 10]
                labels = labels[labels < 10]
            user_id_to_data[user] = [
                numpy_to_tensor(images),
                numpy_to_tensor(labels)
            ]

    return user_id_to_data


def make_federated_dataset(
        h5_file_path: str,
        digits_only: bool = False,
        numpy_to_tensor: Callable = lambda x: x) -> FederatedDataset:
    """
    Create federated dataset from a FEMNIST data file, to use in simulations.

    The federated dataset samples user datasets. A user dataset is
    made from datapoints of one user.

    :param h5_file_path:
        The FEMNIST data file (train or test).
    :returns:
        Federated dataset from the H5 data file.
    """
    user_id_to_data = _load_h5_into_dict(h5_file_path, digits_only,
                                         numpy_to_tensor)
    user_ids = list(user_id_to_data.keys())

    sampler = get_user_sampler('random', user_ids)
    federated_dataset = FederatedDataset.from_slices(user_id_to_data, sampler)
    return federated_dataset


def make_central_dataset(h5_file_path: str,
                         digits_only: bool = False) -> Dataset:
    """
    Create central dataset from a FEMNIST data file.

    :param h5_file_path:
        The FEMNIST data file (train or test).
    :returns:
        A central dataset (represented as a ``Dataset``) from the FEMNIST
        H5 data file. This ``Dataset`` can be used for central evaluation
        with ``CentralEvaluationCallback``.
    """
    user_id_to_data = _load_h5_into_dict(h5_file_path, digits_only)

    images = np.concatenate([data[0] for data in user_id_to_data.values()],
                            axis=0)
    labels = np.concatenate([data[1] for data in user_id_to_data.values()],
                            axis=0)

    return Dataset(raw_data=[images, labels])


def make_femnist_datasets(
        data_dir: str,
        digits_only: bool = False,
        numpy_to_tensor: Callable = lambda x: x,
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and val ``FederatedDataset`` as well as a central dataset
    from the FEMNIST data.
    """

    train_h5_file_path = os.path.join(data_dir, 'fed_emnist_train.h5')
    val_h5_file_path = os.path.join(data_dir, 'fed_emnist_test.h5')

    # create federated training and val datasets from central training and val
    # data
    training_federated_dataset = make_federated_dataset(
        train_h5_file_path, digits_only, numpy_to_tensor)
    val_federated_dataset = make_federated_dataset(
        val_h5_file_path, digits_only, numpy_to_tensor)
    central_data = make_central_dataset(val_h5_file_path, digits_only)

    return training_federated_dataset, val_federated_dataset, central_data, {}
