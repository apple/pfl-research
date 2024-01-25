# Copyright © 2023-2024 Apple Inc.

from typing import List

import h5py
import numpy as np
import torch.utils.data  # type: ignore

from pfl.data import FederatedDataset
from pfl.data.pytorch import PyTorchFederatedDataset  # type: ignore
from pfl.data.sampling import get_user_sampler

from .common import get_label_mapping, get_multi_hot_targets, get_user_num_images


class FLAIRDataset(torch.utils.data.Dataset):
    """
    FLAIR Pytorch Dataset, each item contains the data tensors for a user ID.
    """

    def __init__(self, hdf5_path: str, user_ids: List[str], partition: str,
                 num_classes: int, use_fine_grained_labels: bool,
                 max_num_user_images: int):
        self._hdf5_path = hdf5_path
        self._user_ids = user_ids
        self._partition = partition
        self._num_classes = num_classes
        self._use_fine_grained_labels = use_fine_grained_labels
        self._max_num_user_images = max_num_user_images

    def __len__(self):
        return len(self._user_ids)

    def __getitem__(self, user_id):
        with h5py.File(self._hdf5_path, 'r') as h5:
            inputs = np.array(h5[f'/{self._partition}/{user_id}/images'])
            targets = get_multi_hot_targets((len(inputs), self._num_classes),
                                            h5, self._partition, user_id,
                                            self._use_fine_grained_labels)
            user_slice = slice(0, self._max_num_user_images)
            inputs, targets = inputs[user_slice], targets[user_slice]
            data_order = np.random.permutation(len(inputs))
            inputs, targets = inputs[data_order], targets[data_order]
            inputs = torch.as_tensor(inputs)  # type: ignore
            targets = torch.as_tensor(targets)  # type: ignore
            return inputs, targets


def make_federated_dataset(hdf5_path: str, partition: str,
                           use_fine_grained_labels: bool,
                           max_num_user_images: int) -> FederatedDataset:
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
        Maximum number of images each user can have. Users with more images
        than this number will be splitted into multiple fake users.
    :return:
        Federated dataset from the HDF5 data file.
    """
    num_classes = len(get_label_mapping(hdf5_path, use_fine_grained_labels))
    user_num_images = get_user_num_images(hdf5_path, partition)
    user_ids = sorted(user_num_images.keys())
    sampler = get_user_sampler('random', user_ids)
    flair_dataset = FLAIRDataset(hdf5_path, user_ids, partition, num_classes,
                                 use_fine_grained_labels, max_num_user_images)
    return PyTorchFederatedDataset(flair_dataset,
                                   sampler,
                                   user_id_to_weight=user_num_images)
