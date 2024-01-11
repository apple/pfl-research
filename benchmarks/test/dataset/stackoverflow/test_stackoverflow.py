# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
import json
import os
import pytest

import h5py
from mock import patch
import numpy as np
from pfl.data.sampling import MinimizeReuseUserSampler
from pfl.internal.ops import get_pytorch_major_version, get_tf_major_version
from pfl.internal.ops.selector import _internal_reset_framework_module, set_framework_module

from dataset.stackoverflow.numpy import (get_metadata, get_fraction_of_users,
                                         get_user_weights)


@pytest.fixture
def h5_path(tmp_path):
    h5_path = os.path.join(tmp_path, 'data.hdf5')
    with h5py.File(h5_path, 'w') as h5:
        h5['metadata/vocabulary_size'] = 7
        h5['metadata/unk_symbol'] = 4
        h5['metadata/pad_symbol'] = 0
        h5['metadata/max_sequence_length'] = 10
        user_num_tokens = json.dumps({
            '1': 10,
            '2': 10,
            '3': 10,
        })
        h5['metadata/num_tokens/train'] = user_num_tokens
        h5['metadata/num_tokens/val'] = user_num_tokens
        h5['metadata/num_tokens/test'] = user_num_tokens
        for partition_ix, partition in enumerate(['train', 'val']):
            for user_id in range(1, 4):
                features = np.zeros((3, 2)) + user_id + partition_ix
                targets = np.zeros((3, 2)) + user_id * 10 + partition_ix
                h5.create_dataset(f'/{partition}/{user_id}/inputs',
                                  data=features)
                h5.create_dataset(f'/{partition}/{user_id}/targets',
                                  data=targets)

    yield h5_path


@pytest.fixture
def numpy_so_module():
    from dataset.stackoverflow import numpy

    with patch('dataset.stackoverflow.numpy.get_user_sampler'
               ) as mock_get_sampler:
        mock_get_sampler.side_effect = (
            lambda sample_type, user_ids: MinimizeReuseUserSampler(user_ids))
        yield numpy


@pytest.fixture
def tf2_so_module():
    from dataset.stackoverflow import tensorflow_v2
    from pfl.internal.ops import tensorflow_ops
    _internal_reset_framework_module()
    set_framework_module(tensorflow_ops)
    with patch('dataset.stackoverflow.tensorflow_v2.get_user_sampler'
               ) as mock_get_sampler:
        mock_get_sampler.side_effect = (
            lambda sample_type, user_ids: MinimizeReuseUserSampler(user_ids))
        yield tensorflow_v2


@pytest.fixture
def pytorch_so_module():
    from dataset.stackoverflow import pytorch
    from pfl.internal.ops import pytorch_ops
    _internal_reset_framework_module()
    set_framework_module(pytorch_ops)
    with patch('dataset.stackoverflow.pytorch.get_user_sampler'
               ) as mock_get_sampler:
        mock_get_sampler.side_effect = (
            lambda sample_type, user_ids: MinimizeReuseUserSampler(user_ids))
        yield pytorch


def _check_user_dataset(dataset, ix, partition_ix, expected_num_users):
    np.testing.assert_array_equal(
        dataset.raw_data[0],
        # Users have different data values encoding
        # its id and which partition.
        np.zeros((3, 2)) + (1 + (ix % expected_num_users)) + partition_ix)
    np.testing.assert_array_equal(
        dataset.raw_data[1],
        np.zeros((3, 2)) + (1 + (ix % expected_num_users)) * 10 + partition_ix)


class TestStackOverflowNumpy:

    def test_get_metadata(self, h5_path):
        metadata = get_metadata(h5_path)
        assert metadata == {
            'max_sequence_length': 10,
            'pad_symbol': 0,
            'vocab_size': 7,
            'unk_symbol': 4
        }

    def test_get_fraction_of_users(self, h5_path):
        users = get_fraction_of_users(h5_path,
                                      partition='train',
                                      data_fraction=0.7)
        assert users == ['1', '2']

    def test_get_user_weights(self, h5_path):
        assert get_user_weights(h5_path, 'train') == {
            '1': 10,
            '2': 10,
            '3': 10,
        }


@pytest.mark.parametrize(
    'so_module',
    [
        pytest.param(pytest.lazy_fixture('numpy_so_module'), id='numpy'),
        pytest.param(
            pytest.lazy_fixture('tf2_so_module'),
            marks=[
                pytest.mark.skipif(get_tf_major_version() < 2,
                                   reason='not tf>=2'),
                # This test makes pytest get stuck when running in CI, only run
                # on MacOs for now while rdar://107677363 is not fixed.
                pytest.mark.macos
            ],
            id='tf_v2'),
        pytest.param(pytest.lazy_fixture('pytorch_so_module'),
                     marks=[
                         pytest.mark.skipif(not get_pytorch_major_version(),
                                            reason='PyTorch not installed'),
                     ],
                     id='pytorch'),
    ])
class TestStackOverflowAll:

    @pytest.mark.parametrize('data_fraction', [1.0, 0.7])
    @pytest.mark.parametrize('partition', ['train', 'val'])
    def test_make_federated_dataset(self, so_module, partition, data_fraction,
                                    h5_path):
        fed_data = so_module.make_federated_dataset(
            h5_path,
            partition,
            max_user_sentences=1000,
            data_fraction=data_fraction)

        partition_ix = 0 if partition == 'train' else 1
        expected_num_users = int(3 * data_fraction)
        for ix in range(10):
            user, _ = next(fed_data)
            _check_user_dataset(user, ix, partition_ix, expected_num_users)

    def test_make_central_dataset(self, so_module, h5_path):
        partition = 'train'
        data_fraction = 0.7

        dataset = so_module.make_central_dataset(h5_path, partition,
                                                 data_fraction)
        np.testing.assert_array_equal(
            dataset.raw_data[0],
            np.concatenate([np.ones((3, 2)),
                            np.ones((3, 2)) * 2]))
        np.testing.assert_array_equal(
            dataset.raw_data[1],
            np.concatenate([np.ones((3, 2)) * 10,
                            np.ones((3, 2)) * 20]))

    def test_make_stackoverflow_datasets(self, so_module, h5_path):
        data_fraction = 0.4
        max_user_sentences = 3

        (fed_train, fed_val, central_dataset,
         metadata) = so_module.make_stackoverflow_datasets(
             h5_path, max_user_sentences, data_fraction)
        # 1% of 3 users is rounded down to 0.
        assert len(central_dataset) == 0
        assert metadata == {
            'vocab_size': 7,
            'unk_symbol': 4,
            'pad_symbol': 0,
            'max_sequence_length': 10,
        }
        for ix in range(3):
            user, _ = next(fed_train)
            _check_user_dataset(user,
                                ix=0,
                                partition_ix=0,
                                expected_num_users=1)
        for ix in range(3):
            user, _ = next(fed_val)
            _check_user_dataset(user,
                                ix=ix,
                                partition_ix=1,
                                expected_num_users=3)
