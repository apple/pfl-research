# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from functools import partial
import json
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from pfl.data import FederatedDataset
from pfl.data.dataset import Dataset
from pfl.data.sampling import get_user_sampler


def get_metadata(data_path: str) -> Dict[str, int]:
    """
    :return:
        A tuple with the maximum sequence length, size of vocabulary, symbol
        for UNK and symbol for padding.
    """
    with h5py.File(data_path, 'r') as h5:
        return {
            'vocab_size': int(h5['metadata/vocabulary_size'][()]),
            'unk_symbol': int(h5['metadata/unk_symbol'][()]),
            'pad_symbol': int(h5['metadata/pad_symbol'][()]),
            'max_sequence_length': int(h5['metadata/max_sequence_length'][()])
        }


def get_fraction_of_users(hdf5_path: str, partition: str,
                          data_fraction: float) -> List[str]:
    """
    Get a subset of users representing a fraction of `data_fraction` of the
    data.

    :param hdf5_path:
        Path to StackOverflow HDF5 dataset on disk.
    :param partition:
        ``train``, ``val`` or ``test``.
    :param data_fraction:
        Fraction of total dataset the returned list of users should represent.
    :returns:
        A subset of users representing `data_fraction` of the dataset.
    """
    with h5py.File(hdf5_path, 'r') as h5:
        user_num_tokens = json.loads(
            h5[f'metadata/num_tokens/{partition}'][()])
        users = list(user_num_tokens.keys())
        if data_fraction == 1.0:
            return users

        tokens_cumsum = np.cumsum(list(user_num_tokens.values()))

    cutoff_index = np.searchsorted(tokens_cumsum,
                                   tokens_cumsum[-1] * data_fraction,
                                   side='left')
    selected_users = users[:cutoff_index]
    print(
        f'Using {int(tokens_cumsum[cutoff_index-1])}/{int(tokens_cumsum[-1])}'
        ' tokens.')
    print(f'Using {len(selected_users)}/{len(users)} users.')
    return selected_users


def get_user_weights(hdf5_path, partition):
    with h5py.File(hdf5_path, 'r') as h5:
        return json.loads(h5[f'/metadata/num_tokens/{partition}'][()])


def make_tensors_fn(hdf5_path, partition, max_user_sentences, user_id):
    """
    Main function to load a user's dataset given its id.
    """
    with h5py.File(hdf5_path, 'r') as h5:
        inputs = np.array(h5[f'{partition}/{user_id}/inputs'], dtype=np.int32)
        targets = np.array(h5[f'{partition}/{user_id}/targets'],
                           dtype=np.int32)
        data_order = np.random.permutation(len(inputs))
        inputs = inputs[data_order][:max_user_sentences]
        targets = targets[data_order][:max_user_sentences]
    return [inputs, targets]


def make_dataset_fn(hdf5_path, partition, max_user_sentences, user_id):
    tensors = make_tensors_fn(hdf5_path, partition, max_user_sentences,
                              user_id)
    return Dataset(raw_data=tensors,
                   train_kwargs={"eval": False},
                   eval_kwargs={"eval": True})


def make_federated_dataset(hdf5_path: str,
                           partition: str,
                           max_user_sentences: int = 1000,
                           data_fraction: float = 1.0) -> FederatedDataset:
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
        Federated dataset from the HDF5 data file.
    """
    # Users with more data than the limit are split up into multiple users.
    user_ids = get_fraction_of_users(hdf5_path, partition, data_fraction)
    user_id_to_weight = get_user_weights(hdf5_path, partition)
    make_dataset_fn_ = partial(make_dataset_fn, hdf5_path, partition,
                               max_user_sentences)
    sampler = get_user_sampler('random', user_ids)
    return FederatedDataset(make_dataset_fn_,
                            sampler,
                            user_id_to_weight=user_id_to_weight)


def make_central_dataset(hdf5_path: str, partition: str,
                         data_fraction: float) -> Dataset:
    """
    Create central dataset of ``numpy.ndarray``s from the StackOverflow
    dataset.
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
    inputs_all_list, targets_all_list = [], []
    users = get_fraction_of_users(hdf5_path, partition, data_fraction)

    with h5py.File(hdf5_path, 'r') as h5:
        for user_id in users:
            inputs = np.array(h5[f'{partition}/{user_id}/inputs'])
            targets = np.array(h5[f'{partition}/{user_id}/targets'])
            # Fast way of concatenating the user datasets.
            inputs_all_list.extend(inputs.tolist())
            targets_all_list.extend(targets.tolist())
        inputs_all, targets_all = (np.array(inputs_all_list, dtype=np.int32),
                                   np.array(targets_all_list, dtype=np.int32))
    return Dataset(raw_data=[inputs_all, targets_all],
                   eval_kwargs={"eval": True})


def make_stackoverflow_datasets(
    data_path: str,
    max_user_sentences: int = 1000,
    data_fraction: float = 1.0,
    central_data_fraction: float = 0.01,
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and test ``FederatedDataset`` as well as a
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
