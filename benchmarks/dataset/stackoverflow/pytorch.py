# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from functools import partial
from typing import Any, Dict, List, Tuple

import torch

from pfl.data.dataset import Dataset
from pfl.data.sampling import get_user_sampler
from pfl.data.pytorch import PyTorchFederatedDataset

from .numpy import (get_metadata, get_fraction_of_users, get_user_weights,
                    make_tensors_fn, make_central_dataset as
                    make_central_dataset_numpy)


class _PTDatasetWrap(torch.utils.data.Dataset):
    """
    PyTorch native dataset to fetch StackOverflow user datasets.
    """

    def __init__(self, hdf5_path: str, partition: str, max_user_sentences: int,
                 user_ids: List):
        self._user_ids = user_ids
        self._make_tensors_fn = partial(make_tensors_fn, hdf5_path, partition,
                                        max_user_sentences)
        super().__init__()

    def __getitem__(self, user_id):
        return [torch.as_tensor(t) for t in self._make_tensors_fn(user_id)]

    def __len__(self):
        return len(self._user_ids)


def make_federated_dataset(
        hdf5_path: str,
        partition: str,
        max_user_sentences: int = 1000,
        data_fraction: float = 1.0) -> PyTorchFederatedDataset:
    """
    Create federated dataset from the StackOverflow dataset, to use in
    simulations. The federated dataset samples user datasets. A user dataset
    is made from datapoints of one user.

    :param hdf5_path:
        Path to HDF5 dataset on disk. This file can be created using
        `dataset/stackoverflow/download_preprocess.py` as a one-time
        procedure.
    :param partition:
        ``train``, ``val`` or ``test``.
    :param max_user_sentences:
        Maximum number of sentences (rows in the users features matrix).
        Users with more sentences will be trimmed.
    :param data_fraction:
        Use a subset of users, corresponding to `data_fraction` percent
        of the total data.
    :returns:
        A ``PyTorchFederatedDataset``, which uses PyTorch's ``Dataset``
        and ``DataLoader`` to parallelize preprocessing of users.
    """
    # Users with more data than the limit are split up into multiple users.
    user_ids = get_fraction_of_users(hdf5_path, partition, data_fraction)
    user_id_to_weight = get_user_weights(hdf5_path, partition)

    dataset = _PTDatasetWrap(hdf5_path, partition, max_user_sentences,
                             user_ids)
    sampler = get_user_sampler('random', user_ids)
    return PyTorchFederatedDataset(dataset,
                                   sampler,
                                   user_id_to_weight=user_id_to_weight,
                                   num_workers=4,
                                   pin_memory=True,
                                   prefetch_factor=4,
                                   persistent_workers=False)


def make_central_dataset(hdf5_path: str, partition: str,
                         data_fraction: float) -> Dataset:
    """
    Create central dataset of ``torch.Tensor``s from the
    StackOverflow dataset.
    All the datapoints from all users in the partition is pooled
    together.

    :param hdf5_path:
        Path to StackOverflow HDF5 dataset on disk.
    :param partition:
        ``train``, ``val`` or ``test``.
    :param data_fraction:
        Use a subset of users, corresponding to `data_fraction` percent
        of the total data.
    :returns:
        A central dataset (represented as a ``Dataset``) from the
        StackOverflow HDF5 data file. This ``Dataset`` can be used for
        central evaluation with ``CentralEvaluationCallback``.
    """
    data_tensors = make_central_dataset_numpy(hdf5_path, partition,
                                              data_fraction).raw_data
    data_tensors = [
        torch.as_tensor(t, dtype=torch.int32) for t in data_tensors
    ]
    return Dataset(raw_data=data_tensors)


def make_stackoverflow_datasets(
    data_path: str,
    max_user_sentences: int = 1000,
    data_fraction: float = 1.0,
    central_data_fraction: float = 0.01,
) -> Tuple[PyTorchFederatedDataset, PyTorchFederatedDataset, Dataset, Dict[
        str, Any]]:
    """
    Create a train and test ``PyTorchFederatedDataset`` as well as a
    central dataset for StackOverflow dataset.
    """
    metadata = get_metadata(data_path)
    training_federated_dataset = make_federated_dataset(
        data_path, 'train', max_user_sentences, data_fraction)
    val_federated_dataset = make_federated_dataset(data_path, 'val',
                                                   max_user_sentences, 1.0)

    # Federated evaluation with `val_cohort_size>=200` is reliable enough.
    # This central evaluation is solely to compare with the same validation set
    # as a centrally trained model.
    central_data = make_central_dataset(data_path, 'val',
                                        central_data_fraction)

    return (training_federated_dataset, val_federated_dataset, central_data,
            metadata)
