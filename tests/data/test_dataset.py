# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.data.dataset import Dataset, DatasetSplit, TabularDataset
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version
from pfl.internal.ops.selector import get_framework_module as get_ops


@pytest.fixture(scope='module')
def user_id():
    return 'satoshi'


@pytest.fixture(scope='module')
def metadata():
    return {'ThisIsAKey': 'AndAValue'}


@pytest.fixture(scope='module')
def train_kwargs():
    return {'PrivateLearning': 'Good1'}


@pytest.fixture(scope='module')
def eval_kwargs():
    return {'dodML': 'Good2'}


@pytest.fixture(scope='function')
def dataset(request, numpy_ops, train_kwargs, eval_kwargs, metadata, user_id):
    raw_data = (np.zeros((10, 8)), np.arange(10))
    params = {
        'raw_data': raw_data,
        'train_kwargs': train_kwargs,
        'eval_kwargs': eval_kwargs,
        'metadata': metadata,
        'user_id': user_id,
    }
    if hasattr(request, 'param'):
        params.update(**request.param)
    return Dataset(**params)


@pytest.fixture
def tf():
    import tensorflow  # type: ignore
    return tensorflow


@pytest.fixture(scope='function')
def tf_dataset(request, tf, tensorflow_ops, train_kwargs, eval_kwargs,
               metadata, user_id):
    raw_data = (tf.Variable(np.zeros((10, 8))), tf.constant(np.arange(10)))
    return Dataset(raw_data=raw_data,
                   train_kwargs=train_kwargs,
                   eval_kwargs=eval_kwargs,
                   user_id=user_id,
                   metadata=metadata)


@pytest.fixture()
def num_tabular_datapoints():
    return 10


@pytest.fixture()
def num_tabular_features():
    return 2


@pytest.fixture(scope='function')
def tabular_dataset(request, train_kwargs, eval_kwargs, num_tabular_datapoints,
                    num_tabular_features):
    features = np.arange(num_tabular_datapoints *
                         num_tabular_features).reshape(num_tabular_datapoints,
                                                       num_tabular_features)
    labels = np.array([i % 2 for i in range(num_tabular_datapoints)])
    return TabularDataset(features=features,
                          labels=labels,
                          train_kwargs=train_kwargs,
                          eval_kwargs=eval_kwargs)


@pytest.fixture(scope='function')
def tf_tensor_dataset(request, tf, tensorflow_ops, train_kwargs, eval_kwargs,
                      metadata, user_id):
    from pfl.data.tensorflow import TFTensorDataset
    return TFTensorDataset(features=tf.Variable(np.zeros((10, 8))),
                           labels=tf.constant(np.arange(10)),
                           train_kwargs=train_kwargs,
                           eval_kwargs=eval_kwargs,
                           user_id=user_id,
                           metadata=metadata)


@pytest.fixture(scope='function')
def tf_data_dataset(request, tf, tensorflow_ops, train_kwargs, eval_kwargs,
                    metadata, user_id):
    from pfl.data.tensorflow import TFDataDataset
    data = tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices(tf.Variable(np.zeros((10, 8)))),
         tf.data.Dataset.from_tensor_slices(tf.constant(np.arange(10)))))
    with patch('pfl.data.tensorflow.get_ops') as mock_get_ops:
        mock_ops = MagicMock()
        mock_ops.distributed.world_size = 5
        mock_ops.distributed.global_rank = 0
        mock_get_ops.return_value = mock_ops

        dataset = TFDataDataset(raw_data=data,
                                train_kwargs=train_kwargs,
                                eval_kwargs=eval_kwargs,
                                user_id=user_id,
                                metadata=metadata,
                                prefetch=1)
        # Make __len__ available
        for _ in dataset.iter(100):
            pass
        yield dataset


@pytest.fixture(scope='function')
def pytorch_dataset(request, pytorch_ops, train_kwargs, eval_kwargs, metadata,
                    user_id):
    import torch  # type: ignore
    raw_data = (torch.Tensor(np.zeros(
        (10, 8))).cpu(), torch.Tensor(np.arange(10)).cpu())
    return Dataset(raw_data=raw_data,
                   train_kwargs=train_kwargs,
                   eval_kwargs=eval_kwargs,
                   user_id=user_id,
                   metadata=metadata)


@pytest.fixture(scope='function')
def pytorch_tensor_dataset(request, pytorch_ops, train_kwargs, eval_kwargs,
                           metadata, user_id):
    import torch  # type: ignore

    from pfl.data.pytorch import PyTorchTensorDataset
    raw_data = (torch.Tensor(np.zeros((10, 8))), torch.Tensor(np.arange(10)))
    return PyTorchTensorDataset(tensors=raw_data,
                                train_kwargs=train_kwargs,
                                eval_kwargs=eval_kwargs,
                                user_id=user_id,
                                metadata=metadata)


@pytest.fixture(scope='function')
def pytorch_data_dataset(request, pytorch_ops, train_kwargs, eval_kwargs,
                         metadata, user_id):
    import torch  # type: ignore

    from pfl.data.pytorch import PyTorchDataDataset
    data = torch.utils.data.TensorDataset(torch.zeros((10, 8)),
                                          torch.arange(10))

    with patch('pfl.data.pytorch.get_ops') as mock_get_ops:
        mock_ops = MagicMock()
        mock_ops.distributed.world_size = 5
        mock_ops.distributed.global_rank = 0
        mock_get_ops.return_value = mock_ops
        yield PyTorchDataDataset(raw_data=data,
                                 train_kwargs=train_kwargs,
                                 eval_kwargs=eval_kwargs,
                                 user_id=user_id,
                                 metadata=metadata)


@pytest.fixture(scope='function')
def dataset_split(request, dataset):
    return DatasetSplit(dataset, dataset)


def _check_kwargs(d):
    assert d.train_kwargs['PrivateLearning'] == 'Good1'
    assert d.eval_kwargs['dodML'] == 'Good2'
    assert d.metadata['ThisIsAKey'] == 'AndAValue'
    assert d.user_id == 'satoshi'


tf_mark = [pytest.mark.skipif(get_tf_major_version() < 2, reason='not tf>=2')]
pt_mark = [
    pytest.mark.skipif(not get_pytorch_major_version(),
                       reason='PyTorch not installed')
]


@pytest.mark.parametrize('user_dataset', [
    lazy_fixture('dataset'),
    pytest.param(lazy_fixture('tf_dataset'), marks=tf_mark),
    pytest.param(lazy_fixture('tf_tensor_dataset'), marks=tf_mark),
    pytest.param(lazy_fixture('tf_data_dataset'), marks=tf_mark),
    pytest.param(lazy_fixture('pytorch_dataset'), marks=pt_mark),
    pytest.param(lazy_fixture('pytorch_data_dataset'), marks=pt_mark),
])
class TestAllDatasets:

    def test_properties(self, user_dataset, train_kwargs, eval_kwargs,
                        metadata, user_id):
        assert user_dataset.train_kwargs == train_kwargs
        assert user_dataset.eval_kwargs == eval_kwargs
        assert user_dataset.metadata == metadata
        assert user_dataset.user_id == user_id

    def test_len(self, user_dataset):
        assert len(user_dataset) == 10

    def test_iter(self, user_dataset):
        for i, (t1, t2) in enumerate(user_dataset.iter(5)):
            np.testing.assert_array_equal(get_ops().to_numpy(t1),
                                          np.zeros((5, 8)))
            np.testing.assert_array_equal(get_ops().to_numpy(t2),
                                          np.arange(5) + i * 5)

    def test_get_worker_partition(self, user_dataset):
        ops = get_ops()
        with patch.object(ops.distributed,
                          'distribute_range') as mock_distributed_range:
            mock_distributed_range.side_effect = lambda v: range(0, 2)

            distributed_dataset = user_dataset.get_worker_partition()
            for t1, _ in distributed_dataset.iter(100):
                np.testing.assert_array_equal(t1, np.zeros((2, 8)))
            assert len(distributed_dataset) == 2


class TestDataset:

    def test_split_fraction(self, dataset):
        train_dataset, val_dataset = dataset.split(fraction=0.9)
        assert len(train_dataset) == 9
        assert len(val_dataset) == 1
        _check_kwargs(train_dataset)
        _check_kwargs(val_dataset)

    def test_split_fraction_min_size(self, dataset):

        for num_train_samples in range(1, len(dataset) - 1):
            num_val_samples = len(dataset) - num_train_samples
            train_dataset, val_dataset = dataset.split(
                fraction=0.7,
                min_train_size=num_train_samples,
                min_val_size=num_val_samples)
            assert len(train_dataset) == num_train_samples
            assert len(val_dataset) == num_val_samples
            _check_kwargs(train_dataset)
            _check_kwargs(val_dataset)

    def test_split_fraction_min_size_error(self, dataset):
        with pytest.raises(ValueError):
            _ = dataset.split(fraction=0.9, min_train_size=6, min_val_size=6)


class TestTabularDataset:

    def test_properties(self, tabular_dataset, train_kwargs, eval_kwargs):
        assert tabular_dataset.train_kwargs == train_kwargs
        assert tabular_dataset.eval_kwargs == eval_kwargs

    def test_len(self, tabular_dataset, num_tabular_datapoints):
        assert len(tabular_dataset) == num_tabular_datapoints

    def test_features(self, tabular_dataset, num_tabular_datapoints,
                      num_tabular_features):
        np.testing.assert_array_equal(
            tabular_dataset.features,
            np.arange(num_tabular_datapoints * num_tabular_features).reshape(
                num_tabular_datapoints, num_tabular_features))

    def test_labels(self, tabular_dataset, num_tabular_datapoints):
        np.testing.assert_array_equal(
            tabular_dataset.labels,
            np.array([i % 2 for i in range(num_tabular_datapoints)]))

    def test_iter(self, tabular_dataset, num_tabular_features):
        start = 0
        num_datapoints = 5
        for (features, labels) in tabular_dataset.iter(num_datapoints):
            np.testing.assert_array_equal(
                features,
                np.arange(
                    start * num_tabular_features,
                    start * num_tabular_features +
                    (num_datapoints * num_tabular_features)).reshape(
                        num_datapoints, num_tabular_features))
            np.testing.assert_array_equal(
                labels,
                np.array([j % 2
                          for j in range(start, start + num_datapoints)]))
            start += num_datapoints


class TestDatasetSplit:

    def test_properties(self, dataset_split, train_kwargs, eval_kwargs,
                        metadata, user_id):
        assert dataset_split.train_kwargs == train_kwargs
        assert dataset_split.eval_kwargs == eval_kwargs
        assert dataset_split.metadata == metadata
        assert dataset_split.user_id == user_id
        assert dataset_split.raw_data[0][0][0] == 0

    def test_len(self, dataset_split):
        assert len(dataset_split) == 10

    def test_split(self, dataset_split, dataset):
        train_data, val_data = dataset_split.split()
        assert train_data is dataset
        assert val_data is dataset

    def test_split_fail(self, dataset_split):
        with pytest.raises(AssertionError):
            _ = dataset_split.split(fraction=0.5)
        with pytest.raises(AssertionError):
            _ = dataset_split.split(min_train_size=1)
        with pytest.raises(AssertionError):
            _ = dataset_split.split(min_val_size=1)

    @patch('pfl.data.dataset.get_ops')
    def test_get_worker_partition(self, ops, dataset_split):
        mock_ops = MagicMock()
        mock_ops.distributed.distribute_range = lambda v: range(0, 2)
        ops.return_value = mock_ops

        distributed_dataset = dataset_split.get_worker_partition()
        train_data, val_data = distributed_dataset.split()
        assert len(train_data) == 2
        assert len(val_data) == 2
        np.testing.assert_array_equal(train_data.raw_data[1], np.array([0, 1]))
        np.testing.assert_array_equal(val_data.raw_data[1], np.array([0, 1]))
