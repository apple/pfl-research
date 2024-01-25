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

from pfl.privacy import PLDPrivacyAccountant, PRVPrivacyAccountant, RDPPrivacyAccountant


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


def get_expected_delta_pld(noise_parameter, sampling_probability,
                           num_compositions, mechanism, epsilon):
    if mechanism == 'gaussian':
        composed_pld = privacy_loss_distribution.from_gaussian_mechanism(
            standard_deviation=noise_parameter,
            sensitivity=1,
            sampling_prob=sampling_probability,
        ).self_compose(num_compositions)
    elif mechanism == 'laplace':
        composed_pld = privacy_loss_distribution.from_laplace_mechanism(
            parameter=noise_parameter,
            sensitivity=1,
            sampling_prob=sampling_probability,
        ).self_compose(num_compositions)
    else:
        raise ValueError(f'Mechanism {mechanism} is not a valid value.')

    expected_delta = composed_pld.get_delta_for_epsilon(epsilon)

    return expected_delta


def get_expected_delta_prv(noise_parameter, sampling_probability,
                           num_compositions, mechanism, epsilon):
    if mechanism == 'gaussian':
        prv = PoissonSubsampledGaussianMechanism(
            sampling_probability=sampling_probability,
            noise_multiplier=noise_parameter)

    elif mechanism == 'laplace':
        prv = LaplaceMechanism(mu=noise_parameter)

    else:
        raise ValueError(
            f'Mechanism {mechanism} is not supported for PRV accountant')

    acc_prv = PRVAccountant(prvs=prv,
                            max_self_compositions=int(num_compositions),
                            eps_error=0.07,
                            delta_error=1e-10)

    _, expected_delta, _ = acc_prv.compute_delta(epsilon, num_compositions)

    return expected_delta


def get_expected_delta_rdp(noise_parameter, sampling_probability,
                           num_compositions, mechanism, epsilon):
    if mechanism == 'gaussian':
        event = dp_event.PoissonSampledDpEvent(
            sampling_probability, dp_event.GaussianDpEvent(noise_parameter))

    elif mechanism == 'laplace':
        event = dp_event.LaplaceDpEvent(noise_parameter)

    else:
        raise ValueError(
            f'Mechanism {mechanism} is not supported for PRV accountant')

    rdp_accountant = rdp_privacy_accountant.RdpAccountant().compose(
        event, int(num_compositions))

    expected_delta = rdp_accountant.get_delta(epsilon)

    return expected_delta


class TestPrivacyAccountants:

    @pytest.mark.parametrize(
        'accountant_class, fn_expected_delta, max_bound',
        [(PLDPrivacyAccountant, get_expected_delta_pld, 0.75),
         (PRVPrivacyAccountant, get_expected_delta_prv, 0.75),
         (RDPPrivacyAccountant, get_expected_delta_rdp, 1.0)])
    @pytest.mark.parametrize(
        'num_compositions, sampling_probability, epsilon, delta, noise_parameter, noise_scale, mechanism',  # pylint: disable=line-too-long
        [(1000, 0.01, 2, None, 0.76, 1.0, 'gaussian'),
         (100, 0.1, None, 1e-8, 0.5, 0.5, 'gaussian'),
         (1000, 0.001, 2, 1e-8, None, 0.8, 'gaussian'),
         (100, 0.1, None, 1e-8, 0.5, 1.0, 'laplace')])
    def test(self, num_compositions, sampling_probability, epsilon, delta,
             noise_parameter, noise_scale, mechanism, accountant_class,
             fn_expected_delta, max_bound):
        # these are patches for hyperparameters for the binary search for the
        # noise parameter - these settings speed up the binary search for tests
        with patch(
                'pfl.privacy.privacy_accountant.MIN_BOUND_NOISE_PARAMETER',
                new=0.65), patch(
                    'pfl.privacy.privacy_accountant.MAX_BOUND_NOISE_PARAMETER',
                    new=max_bound):
            with patch(
                    'pfl.privacy.privacy_accountant.RTOL_NOISE_PARAMETER',
                    new=0.1):
                accountant = accountant_class(
                    num_compositions=num_compositions,
                    sampling_probability=sampling_probability,
                    mechanism=mechanism,
                    epsilon=epsilon,
                    delta=delta,
                    noise_parameter=noise_parameter,
                    noise_scale=noise_scale)
                cohort_noise_parameter = (accountant.cohort_noise_parameter /
                                          noise_scale)

                expected_delta = fn_expected_delta(cohort_noise_parameter,
                                                   sampling_probability,
                                                   num_compositions, mechanism,
                                                   accountant.epsilon)

                np.testing.assert_almost_equal(accountant.delta,
                                               expected_delta)

    @pytest.mark.xfail(raises=(ValueError, AssertionError), strict=True)
    @pytest.mark.parametrize('accountant_class', [(PLDPrivacyAccountant),
                                                  (PRVPrivacyAccountant),
                                                  (RDPPrivacyAccountant)])
    @pytest.mark.parametrize(
        'num_compositions, sampling_probability, epsilon, delta, noise_parameter, noise_scale, mechanism',  # pylint: disable=line-too-long
        [(100, 0.1, 2, None, None, 1.0, 'gaussian'),
         (100, 0.1, None, None, None, 1.0, 'laplace'),
         (100, 0.1, 2, 1e-8, None, 1.2, 'gaussian'),
         (100, 0.1, 2, 1e-8, None, 1.0, 'bernoulli'),
         (100, 0.1, 2, 1e-8, 10, 1.0, 'gaussian')])
    def test_fail(self, num_compositions, sampling_probability, epsilon, delta,
                  noise_parameter, noise_scale, mechanism, accountant_class):
        accountant_class(
            num_compositions=num_compositions,
            sampling_probability=sampling_probability,
            mechanism=mechanism,
            epsilon=epsilon,
            delta=delta,
            noise_parameter=noise_parameter,
            noise_scale=noise_scale,
        )
