# Copyright © 2024 Apple Inc.
import pickle
from typing import Optional

import numpy as np


def load_and_preprocess(pickle_file_path: str,
                        channel_means: Optional[np.ndarray] = None,
                        channel_stddevs: Optional[np.ndarray] = None,
                        exclude_classes=None):
    images, labels = pickle.load(open(pickle_file_path, 'rb'))
    images = images.astype(np.float32)

    # Normalize per-channel.
    if channel_means is None:
        channel_means = images.mean(axis=(0, 1, 2), dtype='float64')
    if channel_stddevs is None:
        channel_stddevs = images.std(axis=(0, 1, 2), dtype='float64')
    images = (images - channel_means) / channel_stddevs

    if exclude_classes is not None:
        for exclude_class in exclude_classes:
            mask = (labels != exclude_class).reshape(-1)
            labels = labels[mask]
            images = images[mask]

    return images, labels, channel_means, channel_stddevs


class FEDCIFAR10:

    def __init__(self):

        (train_images, train_labels, channel_means, channel_stddevs
         ) = load_and_preprocess('../data/cifar10/cifar10_train.p')
        (test_images, test_labels, _,
         _) = load_and_preprocess('../data/cifar10/cifar10_test.p',
                                  channel_means, channel_stddevs)

        test_dict = {
            'users': [],
            'num_samples': [],
            'user_data': {},
            'user_data_label': {}
        }
        train_dict = {
            'users': [],
            'num_samples': [],
            'user_data': {},
            'user_data_label': {}
        }

        for user_id, indices in enumerate(
                np.array_split(list(range(len(train_images))),
                               len(train_images) // 50)):
            train_dict['users'].append(str(user_id))
            train_dict['num_samples'].append(len(indices))
            train_dict['user_data'][str(user_id)] = train_images[indices]
            train_dict['user_data_label'][str(
                user_id)] = train_labels[indices].squeeze()

        test_dict['users'].append('0')
        test_dict['num_samples'].append(len(test_labels))
        test_dict['user_data']['0'] = test_images
        test_dict['user_data_label']['0'] = test_labels.squeeze()

        print(
            f'train_num_users={len(train_dict["users"])} '
            f'train_user_sizes={[len(v) for v in train_dict["user_data"].values()]} '
            f'test_num_users={len(test_dict["users"])} '
            f'test_user_sizes={[len(v) for v in test_dict["user_data"].values()]} '
        )

        print(" Dictionaries ready .. ")
        self.trainset, self.testset = train_dict, test_dict
