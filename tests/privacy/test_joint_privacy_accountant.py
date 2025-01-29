# Copyright © 2023-2024 Apple Inc.
'''
Test privacy accountants for DP in privacy_accountant.py.
'''

from unittest.mock import patch

import numpy as np
import pytest
from dp_accounting import dp_event
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.rdp import rdp_privacy_accountant
from prv_accountant import LaplaceMechanism, PoissonSubsampledGaussianMechanism, PRVAccountant

from pfl.privacy import JointPLDPrivacyAccountant


@pytest.fixture()
def num_compositions():
    return [2, 10, 1e2, 1e4]


@pytest.fixture()
def sampling_probability():
    return [1e-8, 1e-5, 1e-2, 1e-1, 0.5]


@pytest.fixture()
def epsilon():
    return [0.5, 2, 5, 20]


@pytest.fixture()
def delta():
    return [1e-12, 1e-6, 1e-2]


@pytest.fixture()
def sigma():
    return []


def get_expected_delta_pld(noise_parameters, sampling_probability,
                           num_compositions, mechanisms, epsilon):

    plds = []
    for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
        if mechanism == 'gaussian':
            pld = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=noise_parameter,
                sensitivity=1,
                sampling_prob=sampling_probability,
            ).self_compose(num_compositions)
        elif mechanism == 'laplace':
            pld = privacy_loss_distribution.from_laplace_mechanism(
                parameter=noise_parameter,
                sensitivity=1,
                sampling_prob=sampling_probability,
            ).self_compose(num_compositions)
        else:
            raise ValueError(f'Mechanism {mechanism} is not a valid value.')

        plds.append(pld)

    composed_pld = None
    for pld in plds:
        if composed_pld:
            composed_pld = composed_pld.compose(pld)
        else:
            composed_pld = pld

    expected_delta = composed_pld.get_delta_for_epsilon(epsilon)
    return expected_delta, plds


class TestPrivacyAccountants:

    @pytest.mark.parametrize(
        'accountant_class, fn_expected_delta, max_bound',
        [(JointPLDPrivacyAccountant, get_expected_delta_pld, 15)])
    @pytest.mark.parametrize(
        'num_compositions, sampling_probability, epsilon, delta, noise_parameters, noise_scale, mechanisms, budget_proportions',  # pylint: disable=line-too-long
        [(1000, 0.01, 2, None, [0.76, 1], 1.0, ['gaussian', 'gaussian'], None),
         (100, 0.1, None, 1e-8, [0.5, 0.5], 0.5, ['gaussian', 'laplace'], None),
         (100, 0.1, 2, 1e-8, None, 0.8, ['gaussian', 'gaussian'], [0.25, 0.75])])
    def test(self, num_compositions, sampling_probability, epsilon, delta,
             noise_parameters, noise_scale, mechanisms, budget_proportions,
             accountant_class, fn_expected_delta, max_bound):
        # these are patches for hyperparameters for the binary search for the
        # noise parameter - these settings speed up the binary search for tests
        with patch(
                'pfl.privacy.joint_privacy_accountant.MIN_BOUND_NOISE_PARAMETER',
                new=2), patch(
                    'pfl.privacy.joint_privacy_accountant.MAX_BOUND_NOISE_PARAMETER',
                    new=max_bound), patch(
                        'pfl.privacy.joint_privacy_accountant.MIN_BOUND_EPSILON',
                        new=2.5), patch(
                            'pfl.privacy.joint_privacy_accountant.MAX_BOUND_EPSILON',
                            new=2.6):
            with patch('pfl.privacy.joint_privacy_accountant.RTOL_NOISE_PARAMETER',
                       new=0.1), patch(
                            'pfl.privacy.joint_privacy_accountant.RTOL_EPSILON',
                            new=0.1):
                accountant = accountant_class(
                    num_compositions=num_compositions,
                    sampling_probability=sampling_probability,
                    mechanisms=mechanisms,
                    epsilon=epsilon,
                    delta=delta,
                    budget_proportions=budget_proportions,
                    noise_parameters=noise_parameters,
                    noise_scale=noise_scale)
                noise_parameters = ([cohort_noise_parameter / noise_scale
                                     for cohort_noise_parameter in accountant.cohort_noise_parameters])

                expected_delta, plds = fn_expected_delta(noise_parameters,
                                                         sampling_probability,
                                                         num_compositions, mechanisms,
                                                         accountant.epsilon)

                np.testing.assert_almost_equal(accountant.delta,
                                               expected_delta)

                if budget_proportions:
                    for pld, p in zip(plds, budget_proportions):
                        np.testing.assert_almost_equal(pld.get_epsilon_for_delta(delta), accountant.large_epsilon * p, decimal=2)

    # @pytest.mark.xfail(raises=(ValueError, AssertionError), strict=True)
    # @pytest.mark.parametrize('accountant_class', [(PLDPrivacyAccountant),
    #                                               (PRVPrivacyAccountant),
    #                                               (RDPPrivacyAccountant)])
    # @pytest.mark.parametrize(
    #     'num_compositions, sampling_probability, epsilon, delta, noise_parameter, noise_scale, mechanism',  # pylint: disable=line-too-long
    #     [(100, 0.1, 2, None, None, 1.0, 'gaussian'),
    #      (100, 0.1, None, None, None, 1.0, 'laplace'),
    #      (100, 0.1, 2, 1e-8, None, 1.2, 'gaussian'),
    #      (100, 0.1, 2, 1e-8, None, 1.0, 'bernoulli'),
    #      (100, 0.1, 2, 1e-8, 10, 1.0, 'gaussian')])
    # def test_fail(self, num_compositions, sampling_probability, epsilon, delta,
    #               noise_parameter, noise_scale, mechanism, accountant_class):
    #     accountant_class(
    #         num_compositions=num_compositions,
    #         sampling_probability=sampling_probability,
    #         mechanism=mechanism,
    #         epsilon=epsilon,
    #         delta=delta,
    #         noise_parameter=noise_parameter,
    #         noise_scale=noise_scale,
    #     )

