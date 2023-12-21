# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import math

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.internal.ops import (
    all_reduce_metrics,
    all_reduce_metrics_and_stats,
    get_pytorch_major_version,
    get_tf_major_version,
)
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import MappedVectorStatistics


@pytest.mark.parametrize('ops_setup', [
    pytest.param(lazy_fixture('numpy_ops_setup'), id='numpy_ops'),
    pytest.param(lazy_fixture('pytorch_ops_setup'),
                 id='pytorch_ops',
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ]),
    pytest.param(lazy_fixture('tensorflow_ops_setup'),
                 id='tensorflow_ops',
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() != 2,
                                        reason='tf!=2')
                 ]),
])
class TestOps:

    def test_get_shape(self, ops_setup):
        assert ops_setup.ops.get_shape(ops_setup.ops_variables[0]) == (2, 2)

    def test_all_reduce_metrics_single_device(self, ops_setup):
        metrics = Metrics([
            (StringMetricName('fake_loss'), Weighted.from_unweighted(1)),
            (StringMetricName('fake_accuracy'), Weighted.from_unweighted(3))
        ])

        reduced_metrics = all_reduce_metrics(metrics)

        for key, val in metrics:
            assert val == reduced_metrics[key]

    def test_all_reduce_metrics_and_stats_single_device(
            self, ops_setup, check_equal_stats):
        metrics = Metrics([
            (StringMetricName('fake_loss'), Weighted.from_unweighted(1)),
            (StringMetricName('fake_accuracy'), Weighted.from_unweighted(3))
        ])

        stats = MappedVectorStatistics({'a': np.ones(4)})
        (reduced_stats,
         reduced_metrics) = all_reduce_metrics_and_stats(stats, metrics)

        for key, val in metrics:
            assert val == reduced_metrics[key]

        check_equal_stats(stats, reduced_stats)

    @pytest.mark.parametrize('noise_fn_name',
                             ['add_gaussian_noise', 'add_laplacian_noise'])
    def test_add_noise(self, noise_fn_name, ops_setup, numpy_vars,
                       check_equal_tensors):
        with ops_setup.deterministic_noise():
            noised_tensors = getattr(ops_setup.ops,
                                     noise_fn_name)(ops_setup.ops_variables,
                                                    1.0, None)

        check_equal_tensors([v + 1 for v in numpy_vars], noised_tensors,
                            ops_setup)

    @pytest.mark.parametrize('noise_fn_name',
                             ['add_gaussian_noise', 'add_laplacian_noise'])
    def test_add_gaussian_noise_seeded(self, noise_fn_name, ops_setup):
        noise_fn = getattr(ops_setup.ops, noise_fn_name)
        noised_tensors1 = noise_fn(ops_setup.ops_variables, 1.0, 1)
        noised_tensors2 = noise_fn(ops_setup.ops_variables, 1.0, 1)
        noised_tensors3 = noise_fn(ops_setup.ops_variables, 1.0, 2)

        for t1, t2 in zip(noised_tensors1, noised_tensors2):
            np.testing.assert_array_equal(ops_setup.ops.to_numpy(t1),
                                          ops_setup.ops.to_numpy(t2))

        for t1, t3 in zip(noised_tensors1, noised_tensors3):
            assert np.any(
                np.not_equal(ops_setup.ops.to_numpy(t1),
                             ops_setup.ops.to_numpy(t3)))

    def test_flatten(self, ops_setup, numpy_vars):
        expected_vector = np.array([0, 1, 2, 3, 0])
        vector, _, _ = ops_setup.ops.flatten(ops_setup.ops_variables)
        numpy_vector = ops_setup.ops.to_numpy(vector)
        np.testing.assert_array_equal(numpy_vector, expected_vector)

    def test_flatten_reshape(self, ops_setup, numpy_vars):
        vector, shapes, dtypes = ops_setup.ops.flatten(ops_setup.ops_variables)
        assert ops_setup.ops.get_shape(vector) == (5, )
        new_weights = ops_setup.ops.reshape(vector, shapes, dtypes)
        new_weights_numpy = [ops_setup.ops.to_numpy(w) for w in new_weights]

        for original_weight, new_weight in zip(numpy_vars, new_weights_numpy):
            np.testing.assert_array_equal(original_weight, new_weight)

    def test_flatten_reshape_element(self, ops_setup, numpy_vars):
        # Flatten and reshape for a vector with a single element must work.
        weight = ops_setup.ops_variables[1]
        vector, shapes, dtypes = ops_setup.ops.flatten([weight])

        assert ops_setup.ops.get_shape(vector) == (1, )
        new_weight, = ops_setup.ops.reshape(vector, shapes, dtypes)
        np.testing.assert_array_equal(weight,
                                      ops_setup.ops.to_numpy(new_weight))

    def test_norm(self, ops_setup):
        l1_norm = ops_setup.ops.global_norm(ops_setup.ops_variables, 1.0)
        assert ops_setup.ops.to_numpy(l1_norm) == 6.0
        l2_norm = ops_setup.ops.global_norm(ops_setup.ops_variables, 2.0)
        assert ops_setup.ops.to_numpy(l2_norm) == pytest.approx(math.sqrt(14))
        inf_norm = ops_setup.ops.global_norm(ops_setup.ops_variables, np.inf)
        assert ops_setup.ops.to_numpy(inf_norm) == 3.0

    def test_to_from_numpy(self, ops_setup):
        array = np.arange(10, dtype=np.float32)
        tensor = ops_setup.ops.to_tensor(array)
        new_array = ops_setup.ops.to_tensor(tensor)
        np.testing.assert_array_equal(array, new_array)

    def test_clone(self, ops_setup):
        array = np.arange(10, dtype=np.float32)
        tensor = ops_setup.ops.to_tensor(array)
        cloned_tensor = ops_setup.ops.clone(tensor)
        assert tensor is not cloned_tensor
        np.testing.assert_array_equal(array, cloned_tensor)

    def test_one_hot(self, ops_setup):
        array = np.arange(3, dtype=np.float32)
        tensor = ops_setup.ops.to_tensor(array)
        one_hot_tensor = ops_setup.ops.one_hot(tensor, depth=5)
        expected_one_hot = np.asarray([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ])
        np.testing.assert_array_equal(ops_setup.ops.to_numpy(one_hot_tensor),
                                      expected_one_hot)


@pytest.mark.parametrize('ops_setup', [
    pytest.param(lazy_fixture('numpy_ops_setup')),
    pytest.param(lazy_fixture('pytorch_ops_setup'),
                 marks=[
                     pytest.mark.skipif(not get_pytorch_major_version(),
                                        reason='PyTorch not installed')
                 ]),
    pytest.param(lazy_fixture('tensorflow_ops_setup'),
                 marks=[
                     pytest.mark.skipif(get_tf_major_version() != 2,
                                        reason='tf!=2')
                 ]),
])
class TestEMAOps:

    def test_clone_variable(self, ops_setup):
        cloned_variable = ops_setup.ops.clone_variable(ops_setup.variable,
                                                       name="clone")
        # cloned variable should be equal to variable
        np.testing.assert_array_equal(
            ops_setup.variable_to_numpy_func(ops_setup.variable),
            ops_setup.variable_to_numpy_func(cloned_variable))
        assert ops_setup.variable is not cloned_variable

    def test_assign_variable(self, ops_setup):
        # reference should NOT be equal to variable before assignment
        assert not np.array_equal(
            ops_setup.variable_to_numpy_func(ops_setup.reference),
            ops_setup.variable_to_numpy_func(ops_setup.variable))
        ops_setup.ops.assign_variable(ops_setup.reference, ops_setup.variable)
        # reference should be equal to variable after assignment
        np.testing.assert_array_equal(
            ops_setup.variable_to_numpy_func(ops_setup.reference),
            ops_setup.variable_to_numpy_func(ops_setup.variable))

    @pytest.mark.parametrize('decay', [0.5, 0.8, 0.9, 0.99, 0.999, 0.9999])
    @pytest.mark.parametrize('steps', [1, 5, 10])
    def test_exponential_moving_average_update(self, ops_setup, decay, steps):
        values = list(range(1, steps + 1))
        variable = ops_setup.variable
        ema_variable = ops_setup.ops.clone_variable(variable, name="ema")
        expected_ema_value = ops_setup.variable_to_numpy_func(
            ops_setup.variable)

        for value in values:
            value = np.ones_like(expected_ema_value) * value
            expected_ema_value = decay * expected_ema_value + (1 -
                                                               decay) * value
            ops_setup.ops.assign_variable(
                variable, ops_setup.numpy_to_tensor_func(value))
            ops_setup.ops.exponential_moving_average_update([variable],
                                                            [ema_variable],
                                                            decay)
            ema_value = ops_setup.variable_to_numpy_func(ema_variable)
            np.testing.assert_array_almost_equal(ema_value, expected_ema_value)
