# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import math

import numpy as np
import pytest
import scipy.special

from pfl.internal.distribution import LogFloat
from pfl.internal.priv_unit_functions import (
    _compute_cap_offset_epsilon,
    _compute_pole_probability,
    _compute_variance,
    _privatize_manual,
    compute_optimal_cap_offset,
)


# Regular PrivUnit variance calculation.
# This comes from Kunal/Vitaly and is only lightly edited, so this should be
# an independent implementation.
def find_best_gamma(dim, eps):
    gamma = (math.exp(eps) - 1) / (math.exp(eps) + 1) * np.sqrt(math.pi /
                                                                (2 * dim - 2))
    g_2 = min(math.exp(eps) / (6 * math.sqrt(dim)), 0.999)
    while eps < 1 / 2 * math.log(
            dim * 36) - (dim - 1) / 2 * math.log(1 - g_2**2) + math.log(g_2):
        g_2 = g_2 / 1.01
    if g_2 > math.sqrt(2 / dim):
        gamma = max(g_2, gamma)
    return gamma


def privUnit_sq_norm(dim, eps, theta):
    alpha = (dim - 1.0) / 2.0
    eps_one = (1 - theta) * eps
    gamma = find_best_gamma(dim, eps_one)
    tau = (1 + gamma) / 2
    eps_zero = theta * eps
    p = math.exp(eps_zero) / (1 + math.exp(eps_zero))
    m = (dim - 2) / (dim - 1) * (1 - gamma**2)**alpha / (2 * np.sqrt(
        math.pi *
        (dim - 3) / 2)) * (p / (1 - scipy.special.betainc(alpha, alpha, tau)) -
                           (1 - p) / scipy.special.betainc(alpha, alpha, tau))
    return (1 / (m * m))


def get_gamma(cap_size_epsilon, num_dimensions):
    # Find a lower-bound on gamma with (16a).
    gamma_lower_bound = ((math.exp(cap_size_epsilon) - 1) /
                         (math.exp(cap_size_epsilon) + 1) *
                         math.sqrt(math.pi / (2 * (num_dimensions - 1))))

    # Try whether we can do better with (16b).
    # This is often possible, particularly for high num_dimensions.
    # This is a good starting point...
    gamma_2 = math.exp(cap_size_epsilon) / 6 * math.sqrt(num_dimensions)
    # ... unless it's in impossible territory:
    if gamma_2 > .999:
        gamma_2 = .999
    # Reduce gamma_2 until the implied epsilon is within budget.
    while cap_size_epsilon < (
            .5 * math.log(num_dimensions) + math.log(6) -
        (num_dimensions - 1) / 2 * math.log(1 - gamma_2**2) +
            math.log(gamma_2)):
        gamma_2 /= 1.01

    if gamma_2 >= math.sqrt(2 / num_dimensions):
        # gamma_2 is valid.
        if gamma_2 > gamma_lower_bound:
            return gamma_2
    return gamma_lower_bound


class TestPrivUnit:

    def test_get_gamma(self):
        results = []
        for cap_size_epsilon in [.1, .2, .5, 1., 2., 4., 7.]:
            for num_dimensions in [10, 100, 1000, 1e4, 1e5, 1e6, 1e7, 1e8]:
                gamma = get_gamma(cap_size_epsilon, num_dimensions)
                cap_size_epsilon_back = _compute_cap_offset_epsilon(
                    gamma, num_dimensions)
                # Aside from numerical issues, the reconstructed gamma could be
                # slightly lower, and cap_size_epsilon_back therefore as well.
                assert cap_size_epsilon * 1.001 > cap_size_epsilon_back
                assert cap_size_epsilon == pytest.approx(cap_size_epsilon_back,
                                                         rel=.1)

                results.append((cap_size_epsilon, cap_size_epsilon_back))
                print(cap_size_epsilon, cap_size_epsilon_back)
        for cap_size_epsilon, cap_size_epsilon_back in results:
            assert cap_size_epsilon * 1.001 > cap_size_epsilon_back
            assert cap_size_epsilon == pytest.approx(cap_size_epsilon_back,
                                                     rel=.1)

    @pytest.mark.parametrize('num_dimensions', [10000, 1000, 100, 10])
    @pytest.mark.parametrize(
        'num_samples',
        [
            1, 10, 100, 1000
            # Only run the higher numbers of samples over lunch.
            # , 10000, 100000,
        ])
    def test_mean_expectation(self, num_dimensions, num_samples):
        np.random.seed(77)

        epsilon = 8.
        gamma = .01

        # Make a unit vector.
        vector = np.zeros(num_dimensions)
        vector[0] = 1.
        # vector = math.sqrt(1 / num_dimensions) * np.ones(num_dimensions)

        assert np.linalg.norm(vector) == pytest.approx(1)

        mean_statistics = np.zeros(vector.shape)
        variance_statistics = np.zeros(vector.shape)

        for _ in range(num_samples):
            privatized_vector = _privatize_manual(epsilon, gamma, vector)
            mean_statistics += privatized_vector
            variance_statistics += np.square(privatized_vector)

        theoretical_variance = _compute_variance(epsilon, gamma,
                                                 num_dimensions)

        mean = mean_statistics / num_samples
        mean_tolerance = (
            5 * math.sqrt(theoretical_variance / num_samples / num_dimensions))
        assert np.allclose(mean, vector, atol=mean_tolerance)

        # print(f'At {num_dimensions} dimensions and '
        #       f'{num_samples} samples:')
        # print(f'Max deviation: {max(np.abs(mean - vector))}; '
        #       f'tolerance {mean_tolerance}')
        # print(f'Relative: '
        #       f'{max(np.abs(mean - vector))/mean_tolerance}')

        # Check the variance against the theoretical variance.
        # Since all samples are on the unit spere, the variance is
        # surprisingly exact (relative to the exact mean).

        variance = sum((variance_statistics / num_samples) - np.square(vector))

        assert variance == pytest.approx(
            _compute_variance(epsilon, gamma, num_dimensions))

    @pytest.mark.parametrize(
        'gamma,num_dimensions',
        [
            (0.1, 10),
            (0.1, 100),
            (0.1, 1000),
            # For higher dimensionality, larger gammas are impossible.
            (0.01, 10),
            (0.01, 100),
            (0.01, 1000),
            (0.01, 10000),
            (0.01, 100000),
            (0.001, 10),
            (0.001, 100),
            (0.001, 1000),
            (0.001, 10000),
            (0.001, 100000),
            (1e-5, 10),
            (1e-5, 100),
            (1e-5, 1000),
            (1e-5, 10000),
            (1e-5, 100000),
        ])
    def test_variance(self, gamma, num_dimensions):
        epsilon = 8.

        (_, pole_epsilon,
         _) = _compute_pole_probability(epsilon, gamma, num_dimensions)
        theta = pole_epsilon / epsilon
        reference_variance = privUnit_sq_norm(num_dimensions + 1, epsilon,
                                              theta)
        assert (_compute_variance(epsilon, gamma,
                                  num_dimensions) == pytest.approx(
                                      reference_variance,
                                      rel=2 / math.sqrt(num_dimensions)))

    def test_optimal_gamma(self):
        compute_optimal_cap_offset(128, 1000)
        # Examples computed entirely separately with a line search
        for epsilon, num_dimensions, near_optimal_variance in [
            (6., 100, 26.89),
            (1., 1000, 6476.4),
            (2., 1000, 1773.23),
            (4., 1000, 614.73),
            (7., 1000, 200.04),
            (10., 1000, 107.54),
            (2., 500000, 887193.5),
            (5., 500000, 191004.5),
            (9., 500000, 63884.62),
        ]:
            optimal_scaling, _optimal_gamma = compute_optimal_cap_offset(
                epsilon, num_dimensions)

            optimal_squared_error = (optimal_scaling**2).value
            assert (optimal_squared_error == pytest.approx(
                near_optimal_variance, rel=.1))
            # The near-optimal squared error should be less optimal, but
            # slightly approximate.
            assert (optimal_squared_error < near_optimal_variance * 1.01)

        # It is hard to compute this for higher epsilons, but clearly the
        # optimal variance should reduce to close to 1 as epsilon increases.

        # for epsilon = 10
        last_optimal_scaling = LogFloat.from_value(107.54)**.5
        num_dimensions = 1000
        for epsilon in list(range(11, 63)) + [2**n for n in range(6, 15)]:
            optimal_scaling, _ = compute_optimal_cap_offset(
                epsilon, num_dimensions)
            relative_tolerance = LogFloat.from_value(1.0001)
            assert (LogFloat.from_value(1) < optimal_scaling <
                    last_optimal_scaling * relative_tolerance)
            last_optimal_scaling = optimal_scaling
