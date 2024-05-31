# -*- coding: utf-8 -*-

import os
from typing import Callable, Dict, Tuple, List, Optional

import h5py
import numpy as np
import torch

from pfl.data import ArtificialFederatedDataset, FederatedDataset
from pfl.data.sampling import get_user_sampler, get_data_sampler
from pfl.data.dataset import Dataset

from .mixture_dataset import (ArtificialFederatedDatasetMixture,
                              partition_by_dirichlet_mixture_class_distribution
                              )
from .sampler import DirichletDataSampler


def _sample_users(user_id_to_data: Dict[str, List[np.ndarray]],
                  filter_method: Optional[str] = None,
                  sample_fraction: float = None,
                  start_idx: int = None,
                  end_idx: int = None,
                  include_sampled: bool = True):

    user_ids = list(user_id_to_data.keys())

    if filter_method is None:
        print('\nKEEP ALL USERS')
        # no change. Use all users
        return user_id_to_data

    elif filter_method == 'index':
        assert start_idx is not None
        assert end_idx is not None
        print(f'\nKEEP ALL USERS WITH IDS IN RANGE {start_idx}-{end_idx}')

        selected_user_ids = user_ids[start_idx:end_idx]

    elif filter_method == 'sample':
        assert sample_fraction >= 0 and sample_fraction <= 1

        sample_number = int(sample_fraction * len(user_ids))

        original_state = np.random.get_state()
        np.random.seed(0)
        sampled_ids = np.random.choice(len(user_ids),
                                       sample_number,
                                       replace=False)
        np.random.set_state(original_state)

        if not include_sampled:
            sampled_ids = np.setdiff1d(np.arange(len(user_ids)), sampled_ids)
        print(
            f'\nKEEP {len(sampled_ids)} SAMPLED USERS WITH IDS: {sampled_ids}')
        selected_user_ids = [user_ids[i] for i in sampled_ids]

    else:
        raise ValueError(f'filter_method {filter_method} is not valid')

    return {user_id: user_id_to_data[user_id] for user_id in selected_user_ids}


def _load_h5_into_dict(
        h5_file_path: str,
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


def make_federated_dataset(user_id_to_data: Dict[str, List[np.ndarray]],
                           use_existing_partition: bool = True,
                           phi=None,
                           alphas=None,
                           user_dataset_len_samplers=None) -> FederatedDataset:
    """
    Create federated dataset from a FEMNIST data file.
    """

    user_ids = list(user_id_to_data.keys())

    if not use_existing_partition:
        images = torch.cat([data[0] for data in user_id_to_data.values()])
        labels = torch.cat([data[1] for data in user_id_to_data.values()])

        if alphas is not None:
            user_idxs = partition_by_dirichlet_mixture_class_distribution(
                labels.cpu().numpy(), phi, alphas, user_dataset_len_samplers)
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

        user_ids = [str(i) for i in range(len(user_idxs))]
        user_id_to_data = dict(
            zip(user_ids, [(images[idx], labels[idx]) for idx in user_idxs]))

    sampler = get_user_sampler('random', user_ids)
    federated_dataset = FederatedDataset.from_slices(user_id_to_data, sampler)

    return federated_dataset


def make_special_federated_dataset(
        user_id_to_data: Dict[str, List[np.ndarray]]) -> FederatedDataset:
    """
    Create federated dataset from a FEMNIST data file.
    Keep same label distribution per user, but mix up the datapoints.
    """

    user_ids = list(user_id_to_data.keys())
    images = torch.cat([data[0] for data in user_id_to_data.values()])

    labels = torch.cat([data[1]
                        for data in user_id_to_data.values()]).cpu().numpy()
    unique_labels = np.unique(labels)
    indices_per_class = {
        i: np.random.permutation(np.nonzero(labels == i)[0])
        for i in unique_labels
    }
    #from collections import Counter
    #print('labels count',Counter(labels))
    #for k,v in indices_per_class.items():
    #    print('indices_per_class', k, len(v))

    new_user_id_to_data = dict()
    start_id_per_class = {i: 0 for i in unique_labels}
    #print('start_id_per_class', start_id_per_class)
    for user_id, data in user_id_to_data.items():
        #print('user_id', user_id)
        user_labels = data[1].cpu().numpy()
        #print('labels', labels)

        # sample images based off labels.
        sampled_data_idx = []
        for label in user_labels:
            #print('label', label)
            #print('start_id_per_class[label]', start_id_per_class[label])
            #print(' indices_per_class[label]', type(indices_per_class[label]), len(indices_per_class[label]))
            sampled_data_idx.append(
                indices_per_class[label][start_id_per_class[label]])
            start_id_per_class[label] += 1

        # TODO might need to ensure labels in not on cpu any more.
        new_user_id_to_data[user_id] = [images[sampled_data_idx], data[1]]

    #new_labels = torch.cat([data[1] for data in new_user_id_to_data.values()]).cpu().numpy()
    #print('new label counts', Counter(new_labels))

    sampler = get_user_sampler('random', user_ids)
    federated_dataset = FederatedDataset.from_slices(new_user_id_to_data,
                                                     sampler)

    return federated_dataset


def make_artificial_federated_dataset(
        user_id_to_data: Dict[str, List[np.ndarray]],
        user_dataset_len_samplers: List[Callable],
        phi: np.ndarray = None,
        alphas: np.ndarray = None) -> Tuple[ArtificialFederatedDataset, dict]:
    """
    Create artificial federated dataset from a FEMNIST data file.
    """
    images = torch.cat([data[0] for data in user_id_to_data.values()])
    labels = torch.cat([data[1] for data in user_id_to_data.values()])

    data = [images, labels]

    if alphas is not None:
        data_samplers = [
            DirichletDataSampler(alpha,
                                 labels.cpu().numpy()) for alpha in alphas
        ]
        return ArtificialFederatedDatasetMixture.from_slices(
            phi, data, data_samplers, user_dataset_len_samplers)
    else:
        data_sampler = get_data_sampler('random', len(labels))
        return ArtificialFederatedDataset.from_slices(
            data, data_sampler, user_dataset_len_samplers[0])


def make_central_dataset(
        user_id_to_data: Dict[str, List[np.ndarray]]) -> Dataset:
    """
    Create central dataset from a FEMNIST data file.
    """
    images = np.concatenate([data[0].cpu() for data in user_id_to_data.values()],
                            axis=0)
    labels = np.concatenate([data[1].cpu() for data in user_id_to_data.values()],
                            axis=0)

    return Dataset(raw_data=[images, labels])


def make_femnist_datasets(
    data_dir: str,
    digits_only: bool = False,
    numpy_to_tensor: Callable = lambda x: x,
    dataset_type: str = 'original',
    phi=None,
    alphas=None,
    user_dataset_len_samplers=None,
    filter_method: Optional[str] = None,
    sample_fraction: float = None,
    start_idx: int = None,
    end_idx: int = None,
    include_sampled: bool = True
) -> Tuple[FederatedDataset, FederatedDataset, Dataset]:
    """
    Create a train and val ``FederatedDataset`` as well as a central dataset
    from the FEMNIST data.
    """

    train_h5_file_path = os.path.join(data_dir, 'fed_emnist_train.h5')
    val_h5_file_path = os.path.join(data_dir, 'fed_emnist_test.h5')

    train_user_id_to_data = _load_h5_into_dict(train_h5_file_path, digits_only,
                                               numpy_to_tensor)
    train_user_id_to_data = _sample_users(train_user_id_to_data, filter_method,
                                          sample_fraction, start_idx, end_idx,
                                          include_sampled)

    val_user_id_to_data = _load_h5_into_dict(val_h5_file_path, digits_only,
                                             numpy_to_tensor)
    val_user_id_to_data = _sample_users(val_user_id_to_data, filter_method,
                                        sample_fraction, start_idx, end_idx,
                                        include_sampled)

    # create federated training and val datasets from central training and val
    # data
    if dataset_type == 'original':
        training_federated_dataset = make_federated_dataset(
            train_user_id_to_data)
        val_federated_dataset = make_federated_dataset(val_user_id_to_data)

    elif dataset_type == 'original_labels_uniform_datapoints':
        training_federated_dataset = make_special_federated_dataset(
            train_user_id_to_data)
        val_federated_dataset = make_special_federated_dataset(
            val_user_id_to_data)

    elif dataset_type == 'polya_mixture_federated':
        training_federated_dataset = make_federated_dataset(
            user_id_to_data=train_user_id_to_data,
            use_existing_partition=False,
            phi=phi,
            alphas=alphas,
            user_dataset_len_samplers=user_dataset_len_samplers)
        val_federated_dataset = make_federated_dataset(
            user_id_to_data=val_user_id_to_data,
            use_existing_partition=False,
            phi=phi,
            alphas=alphas,
            user_dataset_len_samplers=user_dataset_len_samplers)

    elif dataset_type == 'polya_mixture_artificial_federated':
        training_federated_dataset = make_artificial_federated_dataset(
            train_user_id_to_data,
            user_dataset_len_samplers,
            phi=phi,
            alphas=alphas)
        val_federated_dataset = make_artificial_federated_dataset(
            val_user_id_to_data,
            user_dataset_len_samplers,
            phi=phi,
            alphas=alphas)

    elif dataset_type == 'uniform_federated':
        training_federated_dataset = make_federated_dataset(
            train_user_id_to_data,
            use_existing_partition=False,
            user_dataset_len_samplers=user_dataset_len_samplers)
        val_federated_dataset = make_federated_dataset(
            val_user_id_to_data,
            use_existing_partition=False,
            user_dataset_len_samplers=user_dataset_len_samplers)

    elif dataset_type == 'uniform_artificial_federated':
        training_federated_dataset = make_artificial_federated_dataset(
            train_user_id_to_data, user_dataset_len_samplers)
        val_federated_dataset = make_artificial_federated_dataset(
            val_user_id_to_data, user_dataset_len_samplers)

    else:
        raise NotImplementedError(
            f'Dataset type {dataset_type} not recognized.')

    central_data = make_central_dataset(val_user_id_to_data)

    return training_federated_dataset, val_federated_dataset, central_data
