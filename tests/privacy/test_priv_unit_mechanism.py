# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Test priv_unit_mechanism.py.
'''

import math
from typing import cast

import numpy as np
import pytest

from pfl.internal.ops import numpy_ops
from pfl.internal.ops.selector import _internal_reset_framework_module
from pfl.internal.priv_unit_functions import compute_optimal_cap_offset
from pfl.internal.random.draw_from_cap import draw_random_unit_vector
from pfl.privacy.gaussian_mechanism import GaussianMechanism
from pfl.privacy.priv_unit_mechanism import PrivUnitMechanism
from pfl.privacy.privacy_mechanism import SplitPrivacyMechanism
from pfl.stats import MappedVectorStatistics


def sensitivity_theoretical_squared_error(norm_fraction):
    if norm_fraction >= 1:
        return 0

    non_flip_probability = (1 + norm_fraction) / 2

    non_flip_error = (1 - norm_fraction)**2
    flip_error = (-1 - norm_fraction)**2

    return (non_flip_probability * non_flip_error +
            (1 - non_flip_probability) * flip_error)


class TestPrivUnitMechanism:

    @pytest.mark.parametrize('num_dimensions', [20, 100, 5000])
    @pytest.mark.parametrize('magnitude', [0., .6, 2., 5.])
    @pytest.mark.parametrize('test_vector_type',
                             ['one-hot', 'oscillating', 'random'])
    def test_l2_error(self, num_dimensions, magnitude, test_vector_type,
                      fix_global_random_seeds):

        # Make sure that no particular framework is selected from a previous
        # test.
        _internal_reset_framework_module()

        num_samples = 1000
        clipping_bound = 2.

        if test_vector_type == 'one-hot':
            # 1-hot vector.
            vector = np.zeros(shape=(num_dimensions, ))
            vector[0] = magnitude

            reference_vector = np.zeros(shape=(num_dimensions, ))
            reference_vector[0] = (magnitude if magnitude < clipping_bound else
                                   clipping_bound)

        elif test_vector_type == 'oscillating':
            vector = np.zeros(shape=(num_dimensions, ))
            for i in range(num_dimensions):
                vector[i] = (((-1)**i) * magnitude / math.sqrt(num_dimensions))
            assert np.linalg.norm(vector) == pytest.approx(magnitude)
            reference_vector = (vector if magnitude < clipping_bound else
                                (clipping_bound / magnitude) * vector)

        else:
            assert test_vector_type == 'random'
            # Random vector.
            vector = magnitude * draw_random_unit_vector(num_dimensions)
            reference_vector = (vector if magnitude < clipping_bound else
                                (clipping_bound / magnitude) * vector)

        def statistics_for_mechanism(mechanism: SplitPrivacyMechanism, vector,
                                     reference_vector):
            mean_statistics = np.zeros_like(vector)
            variance_statistics = np.zeros_like(vector)
            vector_as_statistics = MappedVectorStatistics({'': vector})

            for _ in range(num_samples):
                # Note: for vectors with l2 norm < 1, we need to call
                # constrain_sensitivity every time, since this will be
                # nondeterministic.
                (privatized_statistics,
                 _) = mechanism.privatize(vector_as_statistics)
                privatized_vector = numpy_ops.flatten(
                    privatized_statistics.get_weights()[1])[0]
                mean_statistics += privatized_vector
                variance_statistics += np.square(privatized_vector)

            mean = mean_statistics / num_samples
            variance = (variance_statistics -
                        np.square(reference_vector)) / num_samples

            return mean, variance

        statistics = [
            statistics_for_mechanism(mechanism, vector, reference_vector)
            for mechanism in [
                GaussianMechanism.construct_single_iteration(
                    clipping_bound, 8., 1e-5),
                PrivUnitMechanism(clipping_bound, 8.)
            ]
        ]

        ((gaussian_mean, gaussian_variance), (priv_unit_mean,
                                              priv_unit_variance)) = statistics

        tolerance = (10 * math.sqrt(num_dimensions / num_samples))
        assert np.isclose(gaussian_mean, reference_vector,
                          atol=tolerance).all()

        assert np.isclose(priv_unit_mean, reference_vector,
                          atol=tolerance).all()

        # PrivUnit should be a bit better than the Gaussian mechanism.
        assert sum(priv_unit_variance) < sum(gaussian_variance)
        # The variance is less by a factor of 2, very roughly.
        assert ((sum(priv_unit_variance) /
                 sum(gaussian_variance)) == pytest.approx(.5, abs=.2))

    def test_constrain_sensitivity(self):
        # Note that constrain_sensitivity scales to 1.
        np.random.seed(123)
        num_samples = 1000

        for num_dimensions in [16, 256, 8192]:

            for l2_norm, l2_norm_bound in [(2., 4.), (2., .5), (0., 2.),
                                           (5., 5.)]:
                epsilon = 4.  # Unused

                theoretical_squared_error = (
                    sensitivity_theoretical_squared_error(l2_norm /
                                                          l2_norm_bound))
                # Unit vector
                unit_vector = np.ones(num_dimensions) / math.sqrt(
                    num_dimensions)

                mechanism = PrivUnitMechanism(l2_norm_bound, epsilon)

                example_vector = l2_norm * unit_vector
                statistics = MappedVectorStatistics({'': example_vector})

                mean_statistics = np.zeros_like(example_vector)
                variance_statistics = np.zeros_like(example_vector)
                for _ in range(num_samples):
                    result_statistics, _ = mechanism.constrain_sensitivity(
                        statistics)
                    result = cast(MappedVectorStatistics,
                                  result_statistics)['']

                    mean_statistics += result
                    variance_statistics += np.square(result)

                theoretical_mean = example_vector / max(l2_norm, l2_norm_bound)

                mean = mean_statistics / num_samples
                variance = variance_statistics / num_samples - np.square(
                    theoretical_mean)

                assert np.isclose(
                    mean,
                    theoretical_mean,
                    atol=1. *
                    math.sqrt(theoretical_squared_error / num_samples)).all()

                assert np.isclose(np.sum(variance),
                                  theoretical_squared_error,
                                  rtol=.01)

    def test_theoretical_error(self):
        epsilon = 2.
        for num_dimensions in [16, 256, 1e5]:
            for l2_norm, l2_norm_bound in [(2., 4.), (2., .5), (0., 2.),
                                           (5., 5.)]:
                print(num_dimensions, l2_norm, l2_norm_bound)

                mechanism = PrivUnitMechanism(l2_norm_bound, epsilon)

                scaling, _ = compute_optimal_cap_offset(
                    epsilon, num_dimensions)

                unit_sensitivity_error = (
                    sensitivity_theoretical_squared_error(l2_norm /
                                                          l2_norm_bound))
                sensitivity_error = (l2_norm_bound**2) * unit_sensitivity_error

                add_noise_error = (l2_norm_bound**2) * ((scaling**2).value - 1)

                overall_error = (add_noise_error + sensitivity_error)

                assert (mechanism.sensitivity_scaling(num_dimensions) ==
                        pytest.approx(l2_norm_bound))

                assert (mechanism.sensitivity_squared_error(
                    num_dimensions,
                    l2_norm) == pytest.approx(unit_sensitivity_error))

                assert (mechanism.add_noise_squared_error(
                    num_dimensions,
                    cohort_size=1) == pytest.approx(add_noise_error))

                assert (mechanism.get_squared_error(
                    num_dimensions, l2_norm,
                    cohort_size=1) == pytest.approx(overall_error))
