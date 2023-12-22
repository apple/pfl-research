# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import json
import random
from typing import Dict, List, Tuple, Optional

import h5py
import numpy as np
import tqdm

from pfl.data.dataset import Dataset


def get_multi_hot_targets(shape: Tuple[int, int], h5: h5py.File,
                          partition: str, use_id: str,
                          use_fine_grained_labels: bool) -> np.ndarray:
    prefix = "fine_grained_labels" if use_fine_grained_labels else "labels"
    row_indices = np.array(h5[f'/{partition}/{use_id}/{prefix}_row'])
    col_indices = np.array(h5[f'/{partition}/{use_id}/{prefix}_col'])
    vec = np.zeros(shape, dtype=np.float32)
    vec[row_indices, col_indices] = 1
    return vec


def get_label_mapping(hdf5_path: str,
                      use_fine_grained_labels: bool) -> Dict[str, int]:
    """
    Get the mapping of labels to indices.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :return:
        A dictionary with label as key and index as value
    """
    with h5py.File(hdf5_path, 'r') as h5:
        if use_fine_grained_labels:
            return json.loads(h5['/metadata/fine_grained_label_mapping'][()])
        else:
            return json.loads(h5['/metadata/label_mapping'][()])


def get_training_channel_mean_stddevs(hdf5_path: str,
                                      data_fraction: float = 0.1,
                                      seed: Optional[int] = None
                                      ) -> Tuple[List[float], List[float]]:
    """
    Get the averaged channel mean and std from training data, following ImageNet
    implementation from https://github.com/soumith/imagenet-multiGPU.torch/blob/master/donkey.lua # pylint: disable=line-too-long
    Technically there is a small privacy leakage if no noise added.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param data_fraction:
        Fraction of users for estimating the statistics.
    :param seed:
        Random seed for sampling a fraction of users.
    :return:
        Two list of three numbers representing channel mean and std.
    """
    num_channels = 3
    mean, std = np.zeros(num_channels), np.zeros(num_channels)
    n = 0
    if seed is not None:
        random.seed(seed)

    with h5py.File(hdf5_path, 'r') as h5:
        user_ids = h5['/train'].keys()
        num_users = int(len(user_ids) * data_fraction)

        random.shuffle(user_ids)
        for user_id in tqdm.tqdm(user_ids[:num_users]):
            inputs = np.array(h5[f'/train/{user_id}/images']) / 255.0
            mean += np.mean(inputs, axis=(1, 2)).sum(axis=0)
            std += np.std(inputs, axis=(1, 2)).sum(axis=0)
            n += len(inputs)

    return (mean / n).tolist(), (std / n).tolist()


def get_channel_mean_stddevs(
        source: str = 'imagenet',
        hdf5_path: Optional[str] = None,
        data_fraction: Optional[float] = None,
        seed: Optional[int] = None) -> Tuple[List[float], List[float]]:
    """
    Get image channel mean and standard deviations for input transformation.

    :param source:
        A string indicating where the mean and std estimation came from.
        'flair' will estimate the statistics from training data and
        'imagenet' will use the statistics from ImageNet data.
    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param data_fraction:
        Fraction of users for estimating the statistics.
    :param seed:
        Random seed for sampling a fraction of users.
    :return:
        Two list of three numbers representing channel mean and std.
    """
    if source == 'flair':
        assert hdf5_path is not None and data_fraction is not None
        return get_training_channel_mean_stddevs(hdf5_path, data_fraction,
                                                 seed)
    elif source == 'imagenet':
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def get_user_num_images(hdf5_path: str, partition: str) -> Dict[str, int]:
    """
    Get the number of images per user.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :return:
        A dictionary with user IDs as keys and number of images
        as values.
    """
    with h5py.File(hdf5_path, 'r') as h5:
        user_num_images = {}
        for key in tqdm.tqdm(
                h5[f'/{partition}'].keys(),
                desc=f"Creating user_num_images for {partition}"):
            user_num_images[key] = h5[f'/{partition}/{key}/image_ids'].shape[0]
    return user_num_images


def make_central_datasets(hdf5_path: str, partition: str,
                          use_fine_grained_labels: bool) -> Dataset:
    """
    Create central dataset from the flair dataset by concatenating data
    points from a set of users.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :return:
        A central dataset (represented as a ``Dataset``) from the flair
        HDF5 data file. This ``Dataset`` can be used for central evaluation
        with ``CentralEvaluationCallback``.
    """
    num_classes = len(get_label_mapping(hdf5_path, use_fine_grained_labels))
    inputs_all, targets_all = [], []

    with h5py.File(hdf5_path, 'r') as h5:
        for user_id in h5[f'/{partition}'].keys():
            inputs = np.array(h5[f'/{partition}/{user_id}/images'])
            targets_shape = (len(inputs), num_classes)
            targets = get_multi_hot_targets(targets_shape, h5, partition,
                                            user_id, use_fine_grained_labels)
            inputs_all.append(inputs)
            targets_all.append(targets)

    inputs_all = np.vstack(inputs_all)
    targets_all = np.vstack(targets_all)
    data_tensors = [inputs_all, targets_all]
    return Dataset(
        raw_data=data_tensors,
        train_kwargs={"eval": False},
        eval_kwargs={"eval": True})
