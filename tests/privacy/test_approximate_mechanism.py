# Copyright © 2023-2024 Apple Inc.
'''
Test approximate_mechanism.py through the implementations of it.
'''

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from pytest_lazyfixture import lazy_fixture

from pfl.metrics import Metrics
from pfl.privacy import GaussianMechanism, LaplaceMechanism, PrivUnitMechanism
from pfl.privacy.approximate_mechanism import SquaredErrorLocalPrivacyMechanism
from pfl.stats import MappedVectorStatistics


class TestApproximateMechanism:

    def compare_approximation(self,
                              mechanism: SquaredErrorLocalPrivacyMechanism,
                              num_samples, num_dimensions, l2_norm,
                              norm_bound):
        """
        Compare the moments of a central approximation of a DP mechanism to the
        original.
        """
        # Note that this is a different use of "num_samples".
        # But for statistical power, we use the same.
        cohort_size = num_samples

        vector = l2_norm * np.ones(num_dimensions) / math.sqrt(num_dimensions)
        statistics = MappedVectorStatistics({'': vector})

        sensitivity_scaling = mechanism.sensitivity_scaling(num_dimensions)

        sensitivity_tolerance = ((1 / math.sqrt(num_samples))
                                 if l2_norm < norm_bound else 1e-5)

        # Applying local DP.
        local_sensitivity_mean_statistics = np.zeros(num_dimensions)
        local_sensitivity_variance_statistics = np.zeros(num_dimensions)
        local_noise_mean_statistics = np.zeros(num_dimensions)
        local_noise_variance_statistics = np.zeros(num_dimensions)

        for _ in range(cohort_size):
            clipped, _ = mechanism.constrain_sensitivity(statistics)
            assert isinstance(clipped, MappedVectorStatistics)
            local_sensitivity_mean_statistics += clipped['']
            local_sensitivity_variance_statistics += np.square(clipped[''])

            noised, _ = mechanism.add_noise(clipped, cohort_size=1)
            assert isinstance(noised, MappedVectorStatistics)

            noise_diff = noised[''] - (sensitivity_scaling * clipped[''])

            local_noise_mean_statistics += noise_diff
            local_noise_variance_statistics += np.square(noise_diff)

        local_sensitivity_mean = (local_sensitivity_mean_statistics /
                                  cohort_size)
        local_sensitivity_variance = (
            local_sensitivity_variance_statistics / cohort_size -
            np.square(local_sensitivity_mean))
        local_noise_mean = local_noise_mean_statistics / cohort_size
        assert np.isclose(local_noise_mean,
                          0,
                          atol=5 *
                          (norm_bound / math.sqrt(num_samples))).all()
        local_noise_variance = (local_noise_variance_statistics / cohort_size)

        theoretical_add_noise_squared_error = (
            mechanism.add_noise_squared_error(num_dimensions, cohort_size=1))
        assert np.sum(local_noise_variance) == pytest.approx(
            theoretical_add_noise_squared_error,
            abs=200 / math.sqrt(num_samples))

        cohort_noise_variance = local_noise_variance * cohort_size

        # Approximate central mechanism.
        approximate_mechanism = mechanism.approximate_as_central_mechanism()

        aggregate = MappedVectorStatistics({'': np.zeros(num_dimensions)})
        sensitivity_variance_statistics = np.zeros(num_dimensions)
        user_context = MagicMock(seed=None)
        for _ in range(cohort_size):
            constrained, _ = approximate_mechanism.postprocess_one_user(
                stats=statistics, user_context=user_context)
            assert isinstance(constrained, MappedVectorStatistics)

            aggregate += constrained
            sensitivity_variance_statistics += np.square(constrained[''])

        scaled_aggregate = aggregate[''] * sensitivity_scaling
        theoretical_sensitivity_squared_error = (
            mechanism.sensitivity_squared_error(num_dimensions, l2_norm))
        sensitivity_mean = aggregate[''] / cohort_size
        sensitivity_variance = (
            (sensitivity_variance_statistics / cohort_size) -
            np.square(sensitivity_mean))
        assert (np.sum(sensitivity_variance) == pytest.approx(
            theoretical_sensitivity_squared_error,
            abs=10 / math.sqrt(num_samples)))

        assert np.isclose(sensitivity_variance,
                          local_sensitivity_variance,
                          atol=sensitivity_tolerance /
                          math.sqrt(num_dimensions)).all()

        assert (np.sum(aggregate['']) / cohort_size == pytest.approx(
            np.sum(local_sensitivity_mean),
            abs=10 * math.sqrt(num_dimensions) * sensitivity_tolerance))

        assert np.isclose(aggregate[''] / cohort_size,
                          local_sensitivity_mean,
                          atol=sensitivity_tolerance).all()

        central_noise_mean_statistics = np.zeros(num_dimensions)
        central_noise_variance_statistics = np.zeros(num_dimensions)
        central_context = MagicMock(cohort_size=cohort_size,
                                    effective_cohort_size=cohort_size,
                                    seed=None)
        for _ in range(num_samples):
            output, _ = approximate_mechanism.postprocess_server(
                stats=aggregate,
                central_context=central_context,
                aggregate_metrics=Metrics())
            assert isinstance(output, MappedVectorStatistics)
            central_noise_mean_statistics += output['']
            central_noise_variance_statistics += np.square(output[''])

        central_noise_mean = (central_noise_mean_statistics / num_samples)
        central_noise_variance = (
            central_noise_variance_statistics / num_samples -
            np.square(central_noise_mean))

        assert np.isclose(central_noise_mean, scaled_aggregate, atol=10).all()

        # The noise on the aggregate must have the same variance as the noise
        # on the cohort would have had.
        assert np.isclose(np.sum(central_noise_variance) / (num_samples - 1),
                          np.sum(cohort_noise_variance) / num_samples,
                          atol=30 * math.sqrt(num_dimensions) /
                          math.sqrt(num_samples))

        assert np.isclose(central_noise_variance / (num_samples - 1),
                          cohort_noise_variance / num_samples,
                          atol=30 * math.sqrt(num_dimensions) /
                          math.sqrt(num_samples)).all()

    @pytest.mark.parametrize('framework', [lazy_fixture('numpy_ops')])
    @pytest.mark.parametrize('num_samples,num_dimensions', [(10, 20),
                                                            (10, 1000),
                                                            (100, 20),
                                                            (1000, 20)])
    @pytest.mark.parametrize('l2_norm, norm_bound', [(4, .5), (2., 5.)])
    def test_gaussian_mechanism_approximation(self, framework, num_samples,
                                              num_dimensions, l2_norm,
                                              norm_bound):
        np.random.seed(123)

        self.compare_approximation(GaussianMechanism(norm_bound, .1),
                                   num_samples=num_samples,
                                   num_dimensions=num_dimensions,
                                   l2_norm=l2_norm,
                                   norm_bound=norm_bound)

    @pytest.mark.parametrize('num_samples,num_dimensions', [(10, 20),
                                                            (10, 1000),
                                                            (100, 20),
                                                            (1000, 20)])
    @pytest.mark.parametrize('l2_norm, norm_bound', [(4, .5), (2., 5.)])
    def test_laplace_mechanism_approximation(self, num_samples, num_dimensions,
                                             l2_norm, norm_bound):
        np.random.seed(123)

        self.compare_approximation(LaplaceMechanism(norm_bound, 6.),
                                   num_samples=num_samples,
                                   num_dimensions=num_dimensions,
                                   l2_norm=l2_norm,
                                   norm_bound=norm_bound)

    # Run >1000 samples only over lunch.
    @pytest.mark.parametrize('framework', [lazy_fixture('numpy_ops')])
    @pytest.mark.parametrize('num_samples,num_dimensions', [(10, 20),
                                                            (10, 1000),
                                                            (100, 20),
                                                            (1000, 20)])
    @pytest.mark.parametrize('l2_norm, norm_bound', [(2., 4.), (4., 4.),
                                                     (6., 4.)])
    def test_priv_unit_mechanism_approximation(self, framework, num_samples,
                                               num_dimensions, l2_norm,
                                               norm_bound):
        np.random.seed(321)

        self.compare_approximation(PrivUnitMechanism(norm_bound, 6.),
                                   num_samples=num_samples,
                                   num_dimensions=num_dimensions,
                                   l2_norm=l2_norm,
                                   norm_bound=norm_bound)
