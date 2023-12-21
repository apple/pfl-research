# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Test adaptive_clipping.py.
'''

import math
import typing

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.common_types import Population
from pfl.context import CentralContext, UserContext
from pfl.hyperparam import AlgorithmHyperParams, ModelHyperParams, get_param_value
from pfl.internal.ops import get_pytorch_major_version, get_tf_major_version
from pfl.metrics import MetricName, Metrics, get_overall_value
from pfl.privacy.adaptive_clipping import AdaptiveClippingGaussianMechanism
from pfl.privacy.gaussian_mechanism import GaussianMechanism
from pfl.privacy.privacy_accountant import PLDPrivacyAccountant
from pfl.stats import ElementWeightedMappedVectorStatistics, MappedVectorStatistics

_clipping_indicator_name = "adaptive_clipping/clipping_indicator"


class TestAdaptiveClipping:
    # These fixtures sets the internal framework module.
    @pytest.mark.parametrize('ops_module', [
        pytest.param(lazy_fixture('tensorflow_ops'),
                     marks=[
                         pytest.mark.skipif(get_tf_major_version() < 2,
                                            reason='not tf>=2')
                     ]),
        pytest.param(lazy_fixture('tensorflow_ops'),
                     marks=[
                         pytest.mark.skipif(get_tf_major_version() < 2,
                                            reason='not tf>=2')
                     ]),
        pytest.param(lazy_fixture('pytorch_ops'),
                     marks=[
                         pytest.mark.skipif(not get_pytorch_major_version(),
                                            reason='PyTorch not installed')
                     ]),
    ])
    def test_adaptive_clipping_simulation(self, ops_module):
        cohort_size = 10
        clipping_bound = 1.0
        adaptive_clipping_norm_quantile = 0.5
        expected_clipping_indicator_scale = 0.75
        make_gaussian_mechanism = lambda clipping_bound: GaussianMechanism(
            clipping_bound=clipping_bound, relative_noise_stddev=0.6)
        adaptive_clipping_mechanism = AdaptiveClippingGaussianMechanism(
            make_gaussian_mechanism=make_gaussian_mechanism,
            initial_clipping_bound=clipping_bound,
            clipping_indicator_noise_stddev=cohort_size * 0.1,
            adaptive_clipping_norm_quantile=adaptive_clipping_norm_quantile)

        user_context = UserContext(num_datapoints=1, seed=None)
        # test postprocess_one_user
        for vector, not_clipped in [(np.ones(5, dtype=np.float32), -1),
                                    (np.zeros(5, dtype=np.float32), 1)]:
            stats = MappedVectorStatistics({"a": ops_module.to_tensor(vector)})
            stats, _ = adaptive_clipping_mechanism.postprocess_one_user(
                stats=stats, user_context=user_context)
            stats = typing.cast(MappedVectorStatistics, stats)
            assert _clipping_indicator_name in stats
            np.testing.assert_allclose(
                ops_module.to_numpy(stats[_clipping_indicator_name]),
                np.array([expected_clipping_indicator_scale * not_clipped]))

        # test postprocess_server
        central_context = CentralContext(
            current_central_iteration=0,
            do_evaluation=True,
            population=Population.TRAIN,
            cohort_size=cohort_size,
            algorithm_params=AlgorithmHyperParams(),
            model_train_params=ModelHyperParams(),
            model_eval_params=ModelHyperParams(),
            seed=5)

        stats = MappedVectorStatistics(
            {
                "a":
                ops_module.to_tensor(np.zeros(5, dtype=np.float32)),
                _clipping_indicator_name:
                ops_module.to_tensor(np.zeros(1, dtype=np.float32))
            },
            weight=float(cohort_size))
        metrics = Metrics()
        stats, post_metrics = adaptive_clipping_mechanism.postprocess_server(
            stats=stats,
            central_context=central_context,
            aggregate_metrics=Metrics())
        metrics |= post_metrics
        stats = typing.cast(MappedVectorStatistics, stats)

        expected_noise_stddev = 1.25 * 0.6  # sqrt(1 + 0.75 ** 2)
        expected_noise = ops_module.add_gaussian_noise(
            [
                ops_module.to_tensor(np.zeros(5, dtype=np.float32)),
                ops_module.to_tensor(np.zeros(1, dtype=np.float32))
            ],
            stddev=expected_noise_stddev,
            seed=5)
        np.testing.assert_array_equal(ops_module.to_numpy(stats["a"]),
                                      ops_module.to_numpy(expected_noise[0]))
        assert _clipping_indicator_name not in stats
        expected_norm_quantile = (expected_noise[1] / cohort_size /
                                  expected_clipping_indicator_scale + 1) / 2
        expected_clipping_bound = clipping_bound * math.exp(
            -0.2 * (expected_norm_quantile - 0.5))
        np.testing.assert_allclose(
            get_param_value(
                adaptive_clipping_mechanism.mutable_clipping_bound),
            expected_clipping_bound)
        assert get_overall_value(metrics[MetricName(
            "norm quantile", Population.TRAIN)]) == expected_norm_quantile  # pytype: disable=wrong-arg-count # pylint: disable=line-too-long

    @pytest.mark.parametrize(
        ('adaptive_clipping_norm_quantile',
         'log_space_clipping_bound_step_size', 'expected_clipping_bound'),
        [(0.5, 0.1, 3), (0.2, 0.2, 2.825293600752746)])
    def test_adaptive_clipping(self, adaptive_clipping_norm_quantile,
                               log_space_clipping_bound_step_size,
                               expected_clipping_bound):
        max_cohort_size = 1000
        population_size = 100000000
        clipping_bound = 3.
        norm_quantile_noise_stddev = 0.05
        accountant = PLDPrivacyAccountant(
            num_compositions=100,
            sampling_probability=max_cohort_size / population_size,
            mechanism='gaussian',
            epsilon=2,
            delta=1e-6)
        make_gaussian_mechanism = (
            lambda clipping_bound: GaussianMechanism.from_privacy_accountant(
                accountant=accountant, clipping_bound=clipping_bound))
        adaptive_clipping_mechanism = AdaptiveClippingGaussianMechanism(
            make_gaussian_mechanism=make_gaussian_mechanism,
            initial_clipping_bound=clipping_bound,
            clipping_indicator_noise_stddev=norm_quantile_noise_stddev * 2 *
            max_cohort_size,
            adaptive_clipping_norm_quantile=adaptive_clipping_norm_quantile,
            log_space_step_size=log_space_clipping_bound_step_size)
        current_iteration = 50
        central_context = CentralContext(
            current_central_iteration=current_iteration,
            do_evaluation=True,
            population=Population.TRAIN,
            cohort_size=max_cohort_size,
            algorithm_params=AlgorithmHyperParams(),
            model_train_params=ModelHyperParams(),
            model_eval_params=ModelHyperParams(),
            seed=5)
        # r = cohort_sigma / clipping_indicator_noise_stddev
        # = 0.3282815 / 100 =  0.003282815 # assumes using PLD accountant
        # clipping_indicator_scale = r / sqrt(1-r**2) * clipping_bound
        # = 0.3282815 / (sqrt(1-0.3282815**2) * 3 = 0.00328283689 * 3
        # = 0.009848498722081937
        expected_clipping_indicator_scale = 0.009848498722081937
        # test postprocess_server
        norm_quantile = 0.5
        scaled_clipping_indicator = (norm_quantile * 2 -
                                     1) * expected_clipping_indicator_scale

        data = {
            "vector": np.arange(10),
            _clipping_indicator_name: np.asarray([scaled_clipping_indicator])
        }
        weight = {
            "vector": np.ones(10) * max_cohort_size,
            _clipping_indicator_name: np.asarray([max_cohort_size])
        }
        statistics = ElementWeightedMappedVectorStatistics(data, weight)

        # test that clipping indicator is popped
        popped_statistics, metrics = adaptive_clipping_mechanism.\
            postprocess_server_live(
            stats=statistics,
            central_context=central_context,
            aggregate_metrics=Metrics())
        popped_statistics = typing.cast(ElementWeightedMappedVectorStatistics,
                                        popped_statistics)
        assert _clipping_indicator_name not in statistics
        np.testing.assert_array_equal(statistics["vector"],
                                      popped_statistics["vector"])
        np.testing.assert_array_equal(statistics.weights["vector"],
                                      popped_statistics.weights["vector"])

        # test that clipping bound is updated correctly
        assert get_param_value(
            adaptive_clipping_mechanism.mutable_clipping_bound
        ) == expected_clipping_bound
        assert get_overall_value(metrics[MetricName(
            "norm quantile", Population.TRAIN)]) == norm_quantile  # pytype: disable=wrong-arg-count # pylint: disable=line-too-long
