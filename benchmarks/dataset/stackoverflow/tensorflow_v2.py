# Copyright © 2023-2024 Apple Inc.
from functools import partial
from typing import Any, Dict, Tuple

import tensorflow as tf

from pfl.data.dataset import Dataset
from pfl.data.sampling import get_user_sampler
from pfl.data.tensorflow import TFFederatedDataset

from .numpy import get_fraction_of_users, get_metadata, get_user_weights, make_tensors_fn
from .numpy import make_central_dataset as make_central_dataset_numpy


def make_federated_dataset(hdf5_path: str,
                           partition: str,
                           max_user_sentences: int = 1000,
                           data_fraction: float = 1.0) -> TFFederatedDataset:
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
        A ``TFFederatedDataset``, which uses ``tf.data.Dataset``
        to parallelize preprocessing of users.
    """

    # Users with more data than the limit are split up into multiple users.
    user_ids = get_fraction_of_users(hdf5_path, partition, data_fraction)
    user_id_to_weight = get_user_weights(hdf5_path, partition)
    make_dataset_fn_ = lambda user_id: partial(
        make_tensors_fn, hdf5_path, partition, max_user_sentences)(
            # Need to convert from bytestring tensor to string.
            user_id.numpy().decode('UTF-8'))
    sampler = get_user_sampler('random', user_ids)

    def make_tf_dataset_fn(data):
        data = data.map(lambda i: tuple(
            tf.py_function(make_dataset_fn_, [i], [tf.int32, tf.int32])))
        data = data.prefetch(10)
        return data

    return TFFederatedDataset(make_tf_dataset_fn,
                              sampler,
                              user_id_dtype=tf.string,
                              user_id_to_weight=user_id_to_weight)


def make_central_dataset(hdf5_path: str, partition: str,
                         data_fraction: float) -> Dataset:
    """
    Create central dataset of ``tf.Tensor``s from the
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
    data_tensors_np = make_central_dataset_numpy(hdf5_path, partition,
                                                 data_fraction).raw_data
    data_tensors = [
        tf.convert_to_tensor(t, dtype=tf.int32) for t in data_tensors_np
    ]
    return Dataset(raw_data=data_tensors)


def make_stackoverflow_datasets(
    data_path: str,
    max_user_sentences: int = 1000,
    data_fraction: float = 1.0,
    central_data_fraction: float = 0.01,
) -> Tuple[TFFederatedDataset, TFFederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and test ``TFFederatedDataset`` as well as a
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
