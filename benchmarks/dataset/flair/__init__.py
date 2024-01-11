# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import logging

from typing import Callable

from .common import (make_central_datasets, get_channel_mean_stddevs,
                     get_label_mapping)

logger = logging.getLogger(name=__name__)


def get_central_data_and_metadata(data_path: str,
                                  use_fine_grained_labels: bool):
    central_data = make_central_datasets(data_path, 'val',
                                         use_fine_grained_labels)
    channel_mean, channel_stddevs = get_channel_mean_stddevs()
    label_mapping = get_label_mapping(data_path, use_fine_grained_labels)

    metadata = {
        "channel_mean": channel_mean,
        "channel_stddevs": channel_stddevs,
        "label_mapping": label_mapping
    }
    return central_data, metadata


def make_flair_datasets(data_path: str, use_fine_grained_labels: bool,
                        max_num_user_images: int, numpy_to_tensor: Callable):
    """
    Create a train and val ``FederatedDataset`` as well as a
    central dataset from the FLAIR dataset.
    """
    from .numpy import make_federated_dataset

    training_federated_dataset = make_federated_dataset(
        data_path, 'train', use_fine_grained_labels, max_num_user_images,
        numpy_to_tensor)
    val_federated_dataset = make_federated_dataset(data_path, 'val',
                                                   use_fine_grained_labels,
                                                   max_num_user_images,
                                                   numpy_to_tensor)

    central_data, metadata = get_central_data_and_metadata(
        data_path, use_fine_grained_labels)
    return (training_federated_dataset, val_federated_dataset, central_data,
            metadata)


def make_flair_iid_datasets(data_path: str, use_fine_grained_labels: bool,
                            user_dataset_len_sampler: Callable,
                            numpy_to_tensor: Callable):
    """
    Create a train and val ``ArtificialFederatedDataset`` as well as a
    central dataset from the FLAIR dataset.
    """
    from .numpy import make_artificial_federated_dataset

    logger.info("Creating FLAIR with artificial I.I.D. data distribution")
    training_federated_dataset = make_artificial_federated_dataset(
        data_path, 'train', use_fine_grained_labels, user_dataset_len_sampler,
        numpy_to_tensor)
    val_federated_dataset = make_artificial_federated_dataset(
        data_path, 'val', use_fine_grained_labels, user_dataset_len_sampler,
        numpy_to_tensor)
    central_data, metadata = get_central_data_and_metadata(
        data_path, use_fine_grained_labels)
    return (training_federated_dataset, val_federated_dataset, central_data,
            metadata)


def make_flair_pytorch_datasets(data_path: str, use_fine_grained_labels: bool,
                                max_num_user_images: int):
    """
    Create a train and val ``FederatedDataset`` as well as a
    central dataset from the FLAIR dataset.
    """
    from .pytorch import make_federated_dataset

    training_federated_dataset = make_federated_dataset(
        data_path, 'train', use_fine_grained_labels, max_num_user_images)
    val_federated_dataset = make_federated_dataset(data_path, 'val',
                                                   use_fine_grained_labels,
                                                   max_num_user_images)

    central_data, metadata = get_central_data_and_metadata(
        data_path, use_fine_grained_labels)
    return (training_federated_dataset, val_federated_dataset, central_data,
            metadata)
