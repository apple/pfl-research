# Copyright © 2023-2024 Apple Inc.

import itertools
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.data import (
    ArtificialFederatedDataset,
    CrossSiloUserSampler,
    FederatedDataset,
    FederatedDatasetMixture,
    MinimizeReuseUserSampler,
)
from pfl.data.dataset import Dataset, TabularDataset
from pfl.internal.ops.common_ops import get_pytorch_major_version, get_tf_major_version


@pytest.fixture(autouse=True, scope='function')
def patch_seeds():
    seeds = iter(range(3, 20000))
    with patch('pfl.data.federated_dataset.FederatedDatasetBase._get_seed'
               ) as mock_get_seed:
        mock_get_seed.side_effect = lambda: next(seeds)
        yield


class TestFederatedDataset(unittest.TestCase):

    def setUp(self):

        self.data = {
            'Filip': (10, ),
            'Rogier': (11, ),
            'Matt': (12, ),
            'Aine': (13, )
        }
        self.make_default_sampler = lambda: MinimizeReuseUserSampler(
            list(self.data.keys()))

    def _test_dataset_iterator(self, it, data_it, n):

        # Test draw multiple batches.
        for _ in range(10):

            for (user_dataset, _), data in zip(itertools.islice(it, n),
                                               itertools.islice(data_it, n)):
                self.assertEqual(user_dataset.raw_data, data)

    def test_make_dataset_fn(self):

        def make(user_id):
            return Dataset(self.data[user_id])

        it = FederatedDataset(make, self.make_default_sampler())
        data_it = itertools.cycle(self.data.values())

        self._test_dataset_iterator(it, data_it, 1)

    def test_from_slices(self):

        # Test different batch sizes.
        for n in [1, 4, 6]:
            it = FederatedDataset.from_slices(self.data,
                                              self.make_default_sampler())
            data_it = itertools.cycle(self.data.values())
            self._test_dataset_iterator(it, data_it, n)

    def test_from_slices_with_dirichlet_class_distribution(self):

        features = np.arange(40)
        labels = np.repeat(np.arange(4), 10)

        data = (features, labels)
        len_sampler = lambda: 4

        with patch('numpy.random.dirichlet') as mock_dirichlet:
            # uniform.
            mock_dirichlet.side_effect = lambda alpha: np.ones(len(alpha)
                                                               ) / len(alpha)
            with patch('numpy.random.uniform') as mock_uniform:
                cdf_cycler = itertools.cycle([0.2, 0.4, 0.6, 0.8])
                mock_uniform.side_effect = lambda: next(cdf_cycler)

                it = (FederatedDataset.
                      from_slices_with_dirichlet_class_distribution(
                          data,
                          labels,
                          alpha=0.1,
                          user_dataset_len_sampler=len_sampler))
                mock_dirichlet.assert_called_with(alpha=[0.1] * 4)
                # According to mock above, each partition should have one
                # label from each class.
                user_data, seed = next(it)
                assert seed == 3
                user_features, user_labels = user_data.raw_data
                np.testing.assert_array_equal(user_labels, np.arange(4))
                # features should be e.g. [5, 15, 25, 35]
                np.testing.assert_array_equal(
                    user_features[1:] - user_features[:-1],
                    np.ones(3) * 10)

    def test_user_index_sampler(self):
        data_values = list(self.data.values())
        sampler = lambda: next(iter(
            self.data))  # Sampler that always returns the first dataset.

        def make(user_id):
            return Dataset(self.data[user_id])

        it = FederatedDataset(make, sampler)
        data_it = itertools.cycle([data_values[0]])
        self._test_dataset_iterator(it, data_it, 1)


class TestArtificialFederatedDataset(unittest.TestCase):

    def setUp(self):
        self.data = (np.array(['Filip', 'Rogier', 'Matt', 'Aine']), np.ones(4))
        self.sampler = lambda n: list(range(n))
        self.sample_len = lambda: len(self.data[0])

    def test_make_dataset_fn(self):

        def make(indices):
            return Dataset(tuple([vec[indices] for vec in self.data]))

        it = ArtificialFederatedDataset(make, self.sampler, self.sample_len)

        # Expect dataset to contain all samples.
        user_dataset, _ = next(it)
        np.testing.assert_array_equal(self.data, user_dataset.raw_data)

    def test_from_slices(self):

        it = ArtificialFederatedDataset.from_slices(
            self.data, self.sampler, sample_dataset_len=self.sample_len)

        # Expect dataset to contain all samples.
        user_dataset, _ = next(it)
        np.testing.assert_array_equal(self.data, user_dataset.raw_data)

    def test_from_slices_create_dataset_fn(self):

        create_dataset_fn = lambda data: TabularDataset(features=data[0],
                                                        labels=data[1])
        it = ArtificialFederatedDataset.from_slices(
            self.data,
            self.sampler,
            sample_dataset_len=self.sample_len,
            create_dataset_fn=create_dataset_fn)

        # Expect dataset to contain all samples.
        user_dataset, _ = next(it)
        np.testing.assert_array_equal(self.data, user_dataset.raw_data)
        np.testing.assert_array_equal(user_dataset.features, self.data[0])
        np.testing.assert_array_equal(user_dataset.labels, self.data[1])

    def test_sample_dataset_len(self):

        def make(indices):
            return Dataset(indices)

        for length in [1, 10]:
            sample_len = lambda length=length: length  # pylint: disable=cell-var-from-loop
            it = ArtificialFederatedDataset(make, self.sampler, sample_len)

            # Expect dataset to contain the sampled indices.
            user_dataset, _ = next(it)
            np.testing.assert_array_equal(list(range(length)),
                                          user_dataset.raw_data)


@pytest.fixture()
def mixture_weights():
    return [0.5, 0.5]


@pytest.fixture()
def num_datapoints():
    return 4


@pytest.fixture()
def dataset_0(num_datapoints):
    return [
        np.arange(num_datapoints * 3).reshape(num_datapoints, 3),
        np.ones(num_datapoints)
    ]


@pytest.fixture()
def component_dataset_0(dataset_0, num_datapoints):
    return ArtificialFederatedDataset.from_slices(
        data=dataset_0,
        data_sampler=lambda n: list(range(n)),
        sample_dataset_len=lambda: 4)


@pytest.fixture()
def dataset_1():
    data = {0: (10, ), 1: (11, )}
    return data


@pytest.fixture()
def component_dataset_1(dataset_1):
    make_default_sampler = lambda: MinimizeReuseUserSampler(
        list(dataset_1.keys()))

    def make(user_id):
        return Dataset(dataset_1[user_id])

    return FederatedDataset(make, make_default_sampler())


@pytest.fixture()
def federated_dataset_mixture(mixture_weights, component_dataset_0,
                              component_dataset_1):
    return FederatedDatasetMixture(mixture_weights,
                                   [component_dataset_0, component_dataset_1])


class TestFederatedDatasetMixture:

    def test_iterator(self, federated_dataset_mixture, dataset_0, dataset_1):
        with patch('numpy.random.choice') as mock_choice:

            def alternating_values():
                i = 0
                while True:
                    yield i % 2
                    i += 1

            generator = alternating_values()

            def side_effect(*args, **kwargs):
                return next(generator)

            mock_choice.side_effect = side_effect

            for i, (user_dataset, _) in enumerate(
                    itertools.islice(federated_dataset_mixture, 8)):
                if i % 2 == 0:
                    np.testing.assert_equal(user_dataset.raw_data, dataset_0)
                else:
                    i %= 4
                    j = 0 if i == 1 else 1
                    np.testing.assert_equal(user_dataset.raw_data,
                                            dataset_1[j])

    @patch('pfl.data.federated_dataset.get_ops',
           side_effect=lambda: MagicMock(distributed=MagicMock(world_size=3,
                                                               global_rank=1)))
    def test_3_workers(self, mock_get_ops, federated_dataset_mixture,
                       dataset_0, dataset_1):
        """
        Verify correct data when worker is rank 1 out of 3
        """
        self.test_iterator(federated_dataset_mixture, dataset_0, dataset_1)


@pytest.fixture
def user_id_to_weight(request):
    if hasattr(request, 'param') and request.param:
        return {i: i + 1 for i in range(100)}
    else:
        return None


@pytest.fixture
def make_fed_data_numpy(user_id_to_weight):
    raw_data = np.arange(20) * 10
    user_it = itertools.cycle(range(len(raw_data)))
    sampler = lambda: next(user_it)

    def make_dataset_fn(user_id):
        return Dataset((raw_data[user_id], ),
                       train_kwargs={"eval": False},
                       eval_kwargs={"eval": True})

    return lambda: FederatedDataset(
        make_dataset_fn, sampler, user_id_to_weight=user_id_to_weight)


@pytest.fixture
def make_artificial_fed_data_numpy(user_id_to_weight):
    raw_data = np.arange(20) * 10
    sample_dataset_len = lambda: 1
    data_id_it = itertools.cycle(range(len(raw_data)))
    data_sampler = lambda length: list(itertools.islice(data_id_it, length))

    def make_dataset_fn(data_ids):
        return Dataset((raw_data[data_ids], ),
                       train_kwargs={"eval": False},
                       eval_kwargs={"eval": True})

    return lambda: ArtificialFederatedDataset(make_dataset_fn, data_sampler,
                                              sample_dataset_len)


@pytest.fixture
def make_fed_data_pytorch(user_id_to_weight):
    import torch
    from torch.utils.data import TensorDataset

    from pfl.data.pytorch import PyTorchFederatedDataset
    data = torch.Tensor(np.arange(20) * 10).cpu()
    dataset = TensorDataset(data)
    sampler = MinimizeReuseUserSampler(range(len(dataset)))
    return lambda: PyTorchFederatedDataset(
        dataset,
        sampler,
        # Multiple processes for loading data will break consistent results
        # independent of number of workers. This is because we have no control
        # over which process finishes the final few user datasets to return in
        # a central iteration. This may start to work again when
        # rdar://82383168 is done.
        # num_workers=2,
        user_id_to_weight=user_id_to_weight,
        dataset_kwargs={
            "train_kwargs": {
                "eval": False
            },
            "eval_kwargs": {
                "eval": True
            }
        })


@pytest.fixture
def make_fed_data_tf(user_id_to_weight):
    import tensorflow as tf

    from pfl.data.tensorflow import TFFederatedDataset

    raw_data = tf.convert_to_tensor(np.arange(20) * 10)
    user_it = itertools.cycle(range(len(raw_data)))
    sampler = lambda: next(user_it)

    @tf.function
    def pipeline(data):
        data = data.map(lambda i: (raw_data[i], raw_data[i]))
        return data

    return lambda: TFFederatedDataset(pipeline,
                                      sampler,
                                      user_id_to_weight=user_id_to_weight,
                                      dataset_kwargs={
                                          "train_kwargs": {
                                              "eval": False
                                          },
                                          "eval_kwargs": {
                                              "eval": True
                                          }
                                      })


@pytest.mark.parametrize('make_fed_data', [
    pytest.param(lazy_fixture('make_fed_data_numpy'), id='numpy'),
    pytest.param(lazy_fixture('make_artificial_fed_data_numpy'),
                 id='artificial_numpy'),
    pytest.param(lazy_fixture('make_fed_data_pytorch'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ],
                 id='pytorch'),
    pytest.param(lazy_fixture('make_fed_data_tf'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() < 2,
                                        reason='not tf>=2')
                 ],
                 id='tensorflow')
])
class TestFrameworkFederatedDataset:

    def test_dataset_kwargs(self, make_fed_data):
        fed_data = make_fed_data()
        dataset, _ = next(fed_data)
        assert dataset.train_kwargs == {"eval": False}
        assert dataset.eval_kwargs == {"eval": True}

    @pytest.mark.parametrize('user_id_to_weight', (False, True), indirect=True)
    def test_single_worker(self, make_fed_data, user_id_to_weight):
        fed_data = make_fed_data()

        dataset, seed = next(fed_data)
        assert seed == 3
        np.testing.assert_array_equal(dataset.raw_data[0], [0])

        dataset, seed = next(fed_data)
        assert seed == 4
        np.testing.assert_array_equal(dataset.raw_data[0], [10])

        for i, (dataset, seed) in enumerate(fed_data.get_cohort(2)):
            assert seed == i + 5
            np.testing.assert_array_equal(dataset.raw_data[0], [20 + i * 10])

    @patch('pfl.data.federated_dataset.get_ops',
           side_effect=lambda: MagicMock(distributed=MagicMock(world_size=3,
                                                               global_rank=1)))
    def test_3_workers(self, mock_get_ops, make_fed_data):
        """ Verify correct data when worker is rank 1 out of 3 """
        # Need to initialize in here to mock get_distributed_addresses.
        fed_data = make_fed_data()
        seeds = iter(range(3, 100))
        with patch.object(fed_data, '_random_state',
                          MagicMock(randint=lambda a, b, dtype: next(seeds))):
            dataset, seed = next(fed_data)
            assert seed == 4
            np.testing.assert_array_equal(dataset.raw_data[0], [10])

            dataset, seed = next(fed_data)
            assert seed == 7
            np.testing.assert_array_equal(dataset.raw_data[0], [40])

            for i, (dataset, seed) in enumerate(fed_data.get_cohort(10)):
                assert seed == 10 + 3 * i
                np.testing.assert_array_equal(dataset.raw_data[0],
                                              [70 + i * 30])


@pytest.mark.parametrize(
    'make_fed_data',
    [
        pytest.param(lazy_fixture('make_fed_data_numpy'), id='numpy'),
        pytest.param(lazy_fixture('make_fed_data_pytorch'),
                     marks=[
                         pytest.mark.skipif(not get_pytorch_major_version(),
                                            reason='PyTorch not installed')
                     ],
                     id='pytorch'),
        # TODO: Other tests get stuck when this is run with pytest. Need to fix.
        # It is related to using multiprocess in the generator of
        # tf.data.Dataset.from_generator
        #pytest.param(lazy_fixture('make_fed_data_tf'),
        #             marks=[
        #                 pytest.mark.skipif(get_tf_major_version() < 2,
        #                                    reason='not tf>=2'),
        #             ],
        #             id='tensorflow')
    ])
@pytest.mark.parametrize('user_id_to_weight', (True, ), indirect=True)
class TestFrameworkFederatedDatasetSorted:

    @patch('pfl.data.federated_dataset.get_ops',
           side_effect=lambda: MagicMock(distributed=MagicMock(world_size=3,
                                                               global_rank=1)))
    def test_3_workers(self, mock_get_ops, make_fed_data, user_id_to_weight):
        """ Verify correct data when worker is rank 1 out of 3 """
        # Need to initialize in here to mock get_distributed_addresses.
        fed_data = make_fed_data()

        dataset, seed = next(fed_data)
        assert seed == 4
        np.testing.assert_array_equal(dataset.raw_data[0], [10])

        dataset, seed = next(fed_data)
        assert seed == 7
        np.testing.assert_array_equal(dataset.raw_data[0], [40])

        # Sorted like this:
        # [150, 140, 130, 120, 110, 100, 90, 80, 70, 60]
        # rank 1 should pick [140, 110, 80]
        for (expected_data,
             expected_seed), (dataset, seed) in zip([(140, 17), (110, 14),
                                                     (80, 11)],
                                                    fed_data.get_cohort(10)):
            assert seed == expected_seed
            np.testing.assert_array_equal(dataset.raw_data[0], [expected_data])


class TestCrossSiloFederatedDataset:

    @patch('pfl.data.federated_dataset.get_ops',
           side_effect=lambda: MagicMock(distributed=MagicMock(world_size=3,
                                                               global_rank=1)))
    @patch('pfl.data.sampling.get_ops',
           side_effect=lambda: MagicMock(distributed=MagicMock(
               world_size=3, local_size=1, global_rank=1)))
    def test_3_workers(self, mock_get_ops, mock_get_ops_2, mock_ops):
        """ Verify correct data when worker is rank 1 out of 3 """
        # Need to initialize in here to mock get_distributed_addresses.
        raw_data = np.arange(10) * 10
        user_ids = list(range(10))
        silo_to_user_ids = {0: [0, 1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
        user_ids_to_silo = {}
        for k, v in silo_to_user_ids.items():
            for user_id in v:
                user_ids_to_silo[user_id] = k
        sampler = CrossSiloUserSampler(sampling_type='minimize_reuse',
                                       user_ids=user_ids,
                                       silo_to_user_ids=silo_to_user_ids)

        def make_dataset_fn(user_id):
            return Dataset((raw_data[user_id], ),
                           train_kwargs={"eval": False},
                           eval_kwargs={"eval": True})

        fed_data = FederatedDataset(make_dataset_fn, sampler)
        # Expected data is [40, 50, 60] for Silo with rank 1
        for (expected_data,
             expected_seed), (dataset, seed) in zip([(40, 4), (50, 7),
                                                     (60, 10)],
                                                    fed_data.get_cohort(10)):
            assert seed == expected_seed
            np.testing.assert_array_equal(dataset.raw_data[0], [expected_data])


if __name__ == '__main__':
    unittest.main()
