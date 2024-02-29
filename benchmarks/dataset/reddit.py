# Copyright © 2023-2024 Apple Inc.
import json
from collections import namedtuple
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from pfl.data import FederatedDataset
from pfl.data.dataset import Dataset
from pfl.data.sampling import get_user_sampler
from pfl.internal.platform import get_platform

UserId = namedtuple('UserId', ['name', 'start_index', 'end_index'])


def _pad_batch(inputs, targets, masks, local_batch_size, max_sequence_len,
               pad_symbol):
    # Extend dataset with placeholder sentences to make divisible by local_batch_size.
    if len(inputs) % local_batch_size != 0:
        num_placeholder = local_batch_size - len(inputs) % local_batch_size
        placeholder_sentences = np.tile(pad_symbol,
                                        (num_placeholder, max_sequence_len))
        inputs = np.concatenate([inputs, placeholder_sentences], axis=0)
        targets = np.concatenate([targets, placeholder_sentences], axis=0)
        placeholder_masks = np.zeros((num_placeholder, max_sequence_len))
        masks = np.concatenate([masks, placeholder_masks], axis=0)
    return inputs, targets, masks


def get_metadata(data_path: str) -> Dict[str, int]:
    """
    :return:
        A tuple with the maximum sequence length, size of vocabulary, symbol
        for UNK and symbol for padding.
    """
    metadata = {'max_sequence_length': 10, 'pad_symbol': 0}
    with h5py.File(data_path, 'r') as h5:
        # The dataset is hardcoded to max 10 tokens per sentence.
        # This is used in all federated learning papers as well.
        metadata['vocab_size'] = int(h5['metadata/vocabulary_size'][()])
        metadata['unk_symbol'] = int(h5['metadata/unk_symbol'][()])
    return metadata


def get_fraction_of_users(hdf5_path: str, partition: str,
                          data_fraction: float) -> List[str]:
    """
    Get a subset of users representing a fraction of `data_fraction` of the
    data.

    :param hdf5_path:
        A h5py dataset object.
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
    print(f'{partition} dataset size (tokens): '
          f'{int(tokens_cumsum[cutoff_index - 1])}')
    return selected_users


def make_federated_dataset(
        hdf5_path: str,
        partition: str,
        include_mask: bool,
        local_batch_size: int,
        max_sequence_len: int = 10,
        max_user_tokens: int = 1600,
        data_fraction: float = 1.0,
        minimum_num_datapoints_per_user: int = 1) -> FederatedDataset:
    """
    Create federated dataset from the Reddit dataset, to use in simulations.
    The federated dataset samples user datasets. A user dataset is
    made from datapoints of one user.

    :param h5:
        A h5py dataset object.
    :param partition:
        ``train``, ``val`` or ``test``.
    :param local_batch_size:
        Batch size for local training. User datasets will be padded
        to be divisible by this number.
    :param max_sequence_len:
        Maximum number of tokens for one sentence.
        Any remaining tokens will be discarded.
        The value used by many publications is 10.
    :returns:
        Federated dataset from the HDF5 data file.
    """
    max_data_size = max_user_tokens // max_sequence_len

    def make_dataset_fn(user_id):
        with h5py.File(hdf5_path, 'r') as h5:
            inputs = np.array(h5[f'{partition}/{user_id.name}/inputs'])
            targets = np.array(h5[f'{partition}/{user_id.name}/targets'])
            user_slice = slice(user_id.start_index, user_id.end_index)
            inputs, targets = inputs[user_slice], targets[user_slice]
            data_order = np.random.permutation(len(inputs))
            inputs, targets = inputs[data_order], targets[data_order]

            if include_mask:
                # Only TF1 needs a mask tensor and manual batch padding.
                masks = (inputs
                         != h5['metadata/pad_symbol']).astype(np.float32)

                inputs, targets, masks = _pad_batch(inputs, targets, masks,
                                                    local_batch_size,
                                                    max_sequence_len,
                                                    h5['metadata/pad_symbol'])
                X_lengths = np.sum(masks, axis=1)
                data_tensors = [inputs, targets, X_lengths, masks]
            else:
                data_tensors = [inputs, targets]
        return Dataset(raw_data=data_tensors)

    # Users with more data than the limit are split up into multiple users.
    user_ids = []
    with h5py.File(hdf5_path, 'r') as h5:
        user_num_tokens = json.loads(
            h5[f'metadata/num_tokens/{partition}'][()])
    for user_name in get_fraction_of_users(hdf5_path, partition,
                                           data_fraction):
        data_size = int(user_num_tokens[user_name] // max_sequence_len)
        for start_index in range(0, data_size, max_data_size):
            end_index = min(data_size + 1, start_index + max_data_size)
            user_id = UserId(name=user_name,
                             start_index=start_index,
                             end_index=end_index)
            if (user_id.end_index -
                    user_id.start_index) >= minimum_num_datapoints_per_user:
                user_ids.append(user_id)
    sampler = get_user_sampler('random', user_ids)

    user_id_to_weight = {
        u: u.end_index - u.start_index
        for i, u in enumerate(user_ids)
    }
    return FederatedDataset(make_dataset_fn,
                            sampler,
                            user_id_to_weight=user_id_to_weight)


def make_central_dataset(hdf5_path: str,
                         partition: str,
                         include_mask: bool,
                         data_fraction: float,
                         local_batch_size: int,
                         max_sequence_len: int = 10) -> Dataset:
    """
    Create central dataset from the Reddit dataset.
    The federated dataset samples user datasets. A user dataset is
    made from datapoints of one user.

    :param h5:
        A h5py dataset object.
    :param partition:
        ``train``, ``val`` or ``test``.
    :param local_batch_size:
        Batch size for local training. User datasets will be padded
        to be divisible by this number.
    :param max_sequence_len:
        Maximum number of tokens for one sentence.
        Any remaining tokens will be discarded.
        The value used by many publications is 10.
    :returns:
        A central dataset (represented as a ``Dataset``) from the Reddit
        HDF5 data file. This ``Dataset`` can be used for central evaluation
        with ``CentralEvaluationCallback``.
    """
    inputs_all_list, targets_all_list, masks_all_list = [], [], []
    users = get_fraction_of_users(hdf5_path, partition, data_fraction)

    with h5py.File(hdf5_path, 'r') as h5:
        for user_id in users:
            inputs = np.array(h5[f'{partition}/{user_id}/inputs'])
            targets = np.array(h5[f'{partition}/{user_id}/targets'])
            masks = (inputs != h5['metadata/pad_symbol']).astype(np.float32)
            # Fast way of concatenating the user datasets.
            inputs_all_list.extend(inputs.tolist())
            targets_all_list.extend(targets.tolist())
            masks_all_list.extend(masks.tolist())
        inputs_all, targets_all, masks_all = (np.array(inputs_all_list),
                                              np.array(targets_all_list),
                                              np.array(masks_all_list))

        inputs_all, targets_all, masks_all = _pad_batch(
            inputs_all, targets_all, masks_all, local_batch_size,
            max_sequence_len, h5['metadata/pad_symbol'])

        if include_mask:
            X_all_lengths = np.sum(inputs_all != h5['metadata/pad_symbol'],
                                   axis=1)
            data_tensors = [inputs_all, targets_all, X_all_lengths, masks_all]
        else:
            data_tensors = [inputs_all, targets_all]
    return Dataset(raw_data=data_tensors, eval_kwargs={"eval": True})


def make_reddit_datasets(
    data_path: str,
    include_mask: bool,
    local_batch_size: int,
    max_user_tokens: int = 1600,
    data_fraction: float = 1.0,
    central_data_fraction: float = 0.01,
    minimum_num_datapoints_per_user: int = 1
) -> Tuple[FederatedDataset, FederatedDataset, Dataset, Dict[str, Any]]:
    """
    Create a train and val ``ArtificialFederatedDataset`` as well as a
    central dataset from the Reddit dataset.

    `minimum_num_datapoints_per_user` is only used to filter users in
    train partition.
    """
    metadata = get_metadata(data_path)

    training_federated_dataset = make_federated_dataset(
        data_path, 'train', include_mask, local_batch_size,
        metadata['max_sequence_length'], max_user_tokens, data_fraction,
        minimum_num_datapoints_per_user)
    val_federated_dataset = make_federated_dataset(
        data_path,
        'val',
        include_mask,
        local_batch_size,
        metadata['max_sequence_length'],
        max_user_tokens,
        1.0,
        # Always evaluate on all val users.
        minimum_num_datapoints_per_user=1)

    # Federated evaluation with `val_cohort_size>=200` is reliable enough.
    # This central evaluation is solely to compare with the same validation set
    # as a centrally trained model.
    central_data = make_central_dataset(data_path, 'val', include_mask,
                                        central_data_fraction,
                                        local_batch_size,
                                        metadata['max_sequence_length'])

    return (training_federated_dataset, val_federated_dataset, central_data,
            metadata)
