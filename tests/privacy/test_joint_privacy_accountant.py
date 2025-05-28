# Copyright © 2023-2024 Apple Inc.
'''
Test joint privacy accountants for DP in joint_privacy_accountant.py.
'''

from unittest.mock import patch

import numpy as np
import pytest
from dp_accounting import dp_event
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.rdp import rdp_privacy_accountant
from prv_accountant import LaplaceMechanism, PoissonSubsampledGaussianMechanism, PRVAccountant

from pfl.privacy import JointPrivacyAccountant, PLDPrivacyAccountant, PRVPrivacyAccountant, RDPPrivacyAccountant


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
                           num_compositions, mechanisms, epsilon, accountants,
                           mechanism_epsilons):

    if not isinstance(sampling_probability, list):
        sampling_probability = [sampling_probability] * len(mechanisms)
    if not isinstance(num_compositions, list):
        num_compositions = [num_compositions] * len(mechanisms)
    plds = []
    for mechanism, noise_parameter, n, p in zip(mechanisms, noise_parameters,
                                                num_compositions,
                                                sampling_probability):
        if mechanism == 'gaussian':
            pld = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=noise_parameter,
                sensitivity=1,
                sampling_prob=p,
            ).self_compose(n)
        elif mechanism == 'laplace':
            pld = privacy_loss_distribution.from_laplace_mechanism(
                parameter=noise_parameter,
                sensitivity=1,
                sampling_prob=p,
            ).self_compose(n)
        else:
            raise ValueError(f'Mechanism {mechanism} is not a valid value.')

        plds.append(pld)

    composed_pld = None
    for pld in plds:
        composed_pld = composed_pld.compose(pld) if composed_pld else pld

    expected_delta = composed_pld.get_delta_for_epsilon(epsilon)
    return expected_delta, plds


def get_expected_delta_prv(noise_parameters, sampling_probability,
                           num_compositions, mechanisms, epsilon, accountants,
                           mechanism_epsilons):

    if not isinstance(sampling_probability, list):
        sampling_probability = [sampling_probability] * len(mechanisms)
    if not isinstance(num_compositions, list):
        num_compositions = [num_compositions] * len(mechanisms)

    prvs = []
    for mechanism, noise_parameter, p in zip(mechanisms, noise_parameters,
                                             sampling_probability):
        if mechanism == 'gaussian':
            prv = PoissonSubsampledGaussianMechanism(
                sampling_probability=p, noise_multiplier=noise_parameter)

        elif mechanism == 'laplace':
            prv = LaplaceMechanism(mu=noise_parameter)

        else:
            raise ValueError(
                f'Mechanism {mechanism} is not supported for PRV accountant')

        prvs.append(prv)

    acc_prv = PRVAccountant(prvs=prvs,
                            max_self_compositions=num_compositions,
                            eps_error=0.07,
                            delta_error=1e-10)

    _, expected_delta, _ = acc_prv.compute_delta(epsilon, num_compositions)

    individual_prvs = [
        PRVAccountant(prvs=prv,
                      max_self_compositions=n,
                      eps_error=0.07,
                      delta_error=1e-10)
        for prv, n in zip(prvs, num_compositions)
    ]
    return expected_delta, individual_prvs


def get_expected_delta_rdp(noise_parameters, sampling_probability,
                           num_compositions, mechanisms, epsilon, accountants,
                           mechanism_epsilons):

    if not isinstance(sampling_probability, list):
        sampling_probability = [sampling_probability] * len(mechanisms)
    if not isinstance(num_compositions, list):
        num_compositions = [num_compositions] * len(mechanisms)

    rdps = []
    full_rdp_accountant = rdp_privacy_accountant.RdpAccountant()
    for mechanism, noise_parameter, n, p in zip(mechanisms, noise_parameters,
                                                num_compositions,
                                                sampling_probability):
        if mechanism == 'gaussian':
            event = dp_event.PoissonSampledDpEvent(
                p, dp_event.GaussianDpEvent(noise_parameter))

        elif mechanism == 'laplace':
            event = dp_event.LaplaceDpEvent(noise_parameter)

        else:
            raise ValueError(
                f'Mechanism {mechanism} is not supported for PRV accountant')

        rdp_accountant = rdp_privacy_accountant.RdpAccountant().compose(
            event, int(n))

        rdps.append(rdp_accountant)

        full_rdp_accountant = full_rdp_accountant.compose(event, int(n))

    expected_delta = full_rdp_accountant.get_delta(epsilon)
    return expected_delta, rdps


def get_expected_delta_multiple_accountants(noise_parameters,
                                            sampling_probability,
                                            num_compositions, mechanisms,
                                            epsilon, accountants,
                                            mechanism_epsilons):

    if not isinstance(sampling_probability, list):
        sampling_probability = [sampling_probability] * len(mechanisms)
    if not isinstance(num_compositions, list):
        num_compositions = [num_compositions] * len(mechanisms)

    expected_delta_mapping = {
        PLDPrivacyAccountant: get_expected_delta_pld,
        PRVPrivacyAccountant: get_expected_delta_prv,
        RDPPrivacyAccountant: get_expected_delta_rdp
    }

    delta_sum = 0
    for accountant_cls, sigma, p, n, mechanism, mechanism_epsilon in zip(
            accountants, noise_parameters, sampling_probability,
            num_compositions, mechanisms, mechanism_epsilons):
        expected_delta_fn = expected_delta_mapping[accountant_cls]
        delta_sum += expected_delta_fn([sigma], p, n, [mechanism],
                                       mechanism_epsilon, None, None)[0]

    return delta_sum, None


class TestPrivacyAccountants:

    @pytest.mark.parametrize(
        'accountants, fn_expected_delta, max_bound',
        [(PLDPrivacyAccountant, get_expected_delta_pld, 5),
         (PRVPrivacyAccountant, get_expected_delta_prv, 5),
         (RDPPrivacyAccountant, get_expected_delta_rdp, 5),
         ([PLDPrivacyAccountant, PRVPrivacyAccountant
           ], get_expected_delta_multiple_accountants, 5)])
    @pytest.mark.parametrize(
        'num_compositions, sampling_probability, epsilon, delta, mechanism_epsilons, mechanism_deltas, noise_parameters, noise_scale, mechanisms, budget_proportions',
        [(1000, 0.01, 2, None, None, None, [0.76, 1], 1.0,
          ['gaussian', 'gaussian'], None),
         ([10, 100], [0.1, 0.1], None, 1e-8, None, None, [1, 1.5], 0.5,
          ['laplace', 'gaussian'], None),
         ([1000, 100], [0.01, 0.01], None, 1e-8, [0.5, 0.25], [2e-8, 2e-8],
          None, 0.5, ['gaussian', 'gaussian'], None),
         ([1000, 100], [0.01, 0.1], 2, 1e-8, None, None, None, 0.8,
          ['gaussian', 'gaussian'], [0.25, 0.75])])
    def test(self, num_compositions, sampling_probability, epsilon, delta,
             mechanism_epsilons, mechanism_deltas, noise_parameters,
             noise_scale, mechanisms, budget_proportions, accountants,
             fn_expected_delta, max_bound):

        if isinstance(accountants, list) and mechanism_epsilons is not None:
            # In the different accountants case with mechanism epsilons and deltas
            # We need delta to be None to compute epsilon and delta by summing
            delta = None
        # these are patches for hyperparameters for the binary search for the
        # noise parameter - these settings speed up the binary search for tests
        with patch(
                'pfl.privacy.joint_privacy_accountant.MIN_BOUND_NOISE_PARAMETER',
                new=2
        ), patch(
                'pfl.privacy.joint_privacy_accountant.MAX_BOUND_NOISE_PARAMETER',
                new=max_bound
        ), patch('pfl.privacy.joint_privacy_accountant.MIN_BOUND_EPSILON',
                 new=2.5), patch(
                     'pfl.privacy.joint_privacy_accountant.MAX_BOUND_EPSILON',
                     new=2.6):
            with patch(
                    'pfl.privacy.joint_privacy_accountant.RTOL_NOISE_PARAMETER',
                    new=0.1), patch(
                        'pfl.privacy.joint_privacy_accountant.RTOL_EPSILON',
                        new=0.1):
                joint_accountant = JointPrivacyAccountant(
                    mechanisms=mechanisms,
                    accountants=accountants,
                    num_compositions=num_compositions,
                    sampling_probability=sampling_probability,
                    total_epsilon=epsilon,
                    total_delta=delta,
                    mechanism_epsilons=mechanism_epsilons,
                    mechanism_deltas=mechanism_deltas,
                    budget_proportions=budget_proportions,
                    noise_parameters=noise_parameters,
                    noise_scale=noise_scale)
                noise_parameters = ([
                    cohort_noise_parameter / noise_scale
                    for cohort_noise_parameter in
                    joint_accountant.cohort_noise_parameters
                ])

                expected_delta, mechanism_accountants = fn_expected_delta(
                    noise_parameters, sampling_probability, num_compositions,
                    mechanisms, joint_accountant.total_epsilon, accountants,
                    joint_accountant.mechanism_epsilons)

                np.testing.assert_almost_equal(
                    joint_accountant.total_delta,
                    expected_delta,
                    err_msg=f'{joint_accountant.total_delta} {expected_delta}')

                if budget_proportions:
                    if accountants is PLDPrivacyAccountant:
                        for acc, p in zip(mechanism_accountants,
                                          budget_proportions):
                            np.testing.assert_almost_equal(
                                acc.get_epsilon_for_delta(delta * p),
                                joint_accountant.naive_epsilon * p,
                                decimal=2)
                    elif accountants is PRVPrivacyAccountant:
                        for acc, p, n in zip(mechanism_accountants,
                                             budget_proportions,
                                             num_compositions):
                            np.testing.assert_almost_equal(
                                acc.compute_epsilon(delta * p, [n])[1],
                                joint_accountant.naive_epsilon * p,
                                decimal=2)
                    elif accountants is RDPPrivacyAccountant:
                        for acc, p in zip(mechanism_accountants,
                                          budget_proportions):
                            np.testing.assert_almost_equal(
                                acc.get_epsilon(delta * p),
                                joint_accountant.naive_epsilon * p,
                                decimal=2)

    @pytest.mark.xfail(raises=(ValueError, AssertionError), strict=True)
    @pytest.mark.parametrize(
        'accountants, num_compositions, sampling_probability, epsilon, delta, mechanism_epsilons, mechanism_deltas, noise_parameters, noise_scale, mechanisms, budget_proportions',
        [
            (PLDPrivacyAccountant, 100, 0.1, 2, None, None, None, None, 1.0,
             ['gaussian', 'gaussian'], [0.5, 0.5]),
            (PRVPrivacyAccountant, 100, 0.1, None, None, None, None, None, 1.0,
             ['gaussian', 'gaussian'], [0.5, 0.5]),
            (RDPPrivacyAccountant, 100, 0.1, 2, 1e-8, None, None, None, 1.2,
             ['gaussian', 'gaussian'], [0.5, 0.5]),
            (PLDPrivacyAccountant, 100, 0.1, 2, 1e-8, None, None, None, 1.0,
             ['bernoulli', 'gaussian'], [0.5, 0.5]),
            (PLDPrivacyAccountant, 100, 0.1, 2, 1e-8, None, None, [10, 10],
             1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
            (PLDPrivacyAccountant, 100, 0.1, 2, 1e-8, None, None, None, 0.8,
             ['gaussian', 'gaussian'], [0.1, 0.75]),
            (PLDPrivacyAccountant, 100, 0.1, 2, 1e-8, None, None, None, 0.8,
             ['gaussian', 'gaussian'], [1.1, 0.75]),
            (PLDPrivacyAccountant, 100, 0.1, 2, None, [1, 1], None, None, 1.0,
             ['gaussian', 'gaussian'], [0.5, 0.5]),
            (PLDPrivacyAccountant, 100, 0.1, None, None, [1, 1], [1e-8, 1e-8],
             None, 1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
            (PLDPrivacyAccountant, 100, 0.1, 2, 1e-8, [1, 1], [1e-8, 1e-8],
             None, 1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
            (PLDPrivacyAccountant, 100, 0.1, 2, None, [1, 1], [1e-8, 1e-8],
             [10, 10], 1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
            ([PLDPrivacyAccountant, PRVPrivacyAccountant], 100, 0.1, 2, None,
             None, None, None, 1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
            ([PLDPrivacyAccountant, PRVPrivacyAccountant], 100, 0.1, None,
             None, None, None, None, 1.0, ['gaussian', 'gaussian'], [0.5, 0.5
                                                                     ]),
            ([PLDPrivacyAccountant, PRVPrivacyAccountant], 100, 0.1, 2, 1e-8,
             None, None, [10, 10], 1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
            ([PLDPrivacyAccountant, PRVPrivacyAccountant], 100, 0.1, 2, 1e-8, [
                1, 2
            ], [1e-8, 1e-8], None, 1.0, ['gaussian', 'gaussian'], [0.5, 0.5]),
        ])
    def test_fail(self, num_compositions, sampling_probability, epsilon, delta,
                  mechanism_epsilons, mechanism_deltas, noise_parameters,
                  noise_scale, mechanisms, budget_proportions, accountants):
        JointPrivacyAccountant(
            mechanisms=mechanisms,
            accountants=accountants,
            num_compositions=num_compositions,
            sampling_probability=sampling_probability,
            total_epsilon=epsilon,
            total_delta=delta,
            mechanism_epsilons=mechanism_epsilons,
            mechanism_deltas=mechanism_deltas,
            budget_proportions=budget_proportions,
            noise_parameters=noise_parameters,
            noise_scale=noise_scale,
        )
