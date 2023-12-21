# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Privacy accountants for differential privacy.
'''

import math
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar

from dp_accounting import dp_event
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.rdp import rdp_privacy_accountant
from prv_accountant import LaplaceMechanism, PoissonSubsampledGaussianMechanism, PRVAccountant


@dataclass
class PrivacyAccountant:
    """
    Tracks the privacy loss over multiple composition steps.
    Either two or three of the variables epsilon, delta and noise_parameter
    must be defined. If two are defined, the remaining variable can be
    computed. If all three are defined a check will be performed to make sure
    a valid set of variable values has been provided.
    :param num_compositions:
        Maximum number of compositions to be performed with mechanism.
    :param sampling_probability:
       Maximum probability of sampling each entity being privatized. E.g. if
       the unit of privacy is one device, this is the probability of each
       device participating.
    :param mechanism:
        The noise mechanism used. E.g. Gaussian, Laplace.
    :param epsilon:
        The privacy loss random variable. It controls how much the output of
        the mechanism can vary between two neighboring databases.
    :param delta:
        The probability that all privacy will be lost.
    :param noise_parameter:
        A parameter for DP noise. For the Gaussian mechanism, the noise
        parameter is the standard deviation of the noise. For the Laplace
        mechanism, the noise parameter is the scale of the noise.
    :param noise_scale:
        A value \\in [0, 1] multiplied with the standard deviation of the noise
        to be added for privatization. Typically used to experiment with lower
        sampling probabilities when it is not possible or desirable to increase
        the population size of the units being privatized, e.g. user devices.
    """
    num_compositions: int
    sampling_probability: float
    mechanism: str
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    noise_parameter: Optional[float] = None
    noise_scale: float = 1.0

    def __post_init__(self):
        assert [
            self.epsilon, self.delta, self.noise_parameter
        ].count(None) <= 1, (
            f'At least two of epsilon ({self.epsilon}),'
            f'delta ({self.delta}) and noise parameter ({self.noise_parameter})'
            'must be defined for a privacy accountant')
        if self.noise_scale <= 0 or self.noise_scale > 1.0:
            raise ValueError("noise_scale must be in range (0,1]")
        assert (
            self.sampling_probability >= 0
            and self.sampling_probability <= 1.0), (
                f'Sampling probability {self.sampling_probability} is invalid.'
                'Must be in range [0, 1]')
        assert self.num_compositions > 0 and isinstance(
            self.num_compositions,
            int), ('Number of compositions must be a positive integer')
        if self.noise_parameter is not None:
            assert self.noise_parameter > 0, (
                'Noise parameter must be a positive real value.')
        if self.epsilon is not None:
            assert self.epsilon >= 0, (
                'Epsilon must be a non-negative real value')
        if self.delta is not None:
            assert self.delta > 0 and self.delta < 1, (
                'Delta should be a positive real value in range (0, 1)')

        self.mechanism = self.mechanism.lower()

    @property
    def cohort_noise_parameter(self):
        """
        Noise parameter to be used on a cohort of users.
        Noise scale is considered.
        """
        return self.noise_parameter * self.noise_scale


PrivacyAccountantKind = TypeVar('PrivacyAccountantKind',
                                bound=PrivacyAccountant)


@dataclass
class PLDPrivacyAccountant(PrivacyAccountant):
    """
    Privacy Loss Distribution (PLD) privacy accountant, from dp-accounting
    package.
    Code: https://github.com/google/differential-privacy/blob/main/python/dp_accounting/pld/pld_privacy_accountant.py # pylint: disable=line-too-long
    The PLD algorithm is based on: “Tight on budget?: Tight bounds for r-fold
    approximate differential privacy.”, Meiser and Mohammadi, in CCS, pages
    247–264, 2018, https://eprint.iacr.org/2017/1034.pdf
    The Connect-the-Docts algorithm is based on: “Connect the Dots: Tighter
    Discrete Approximations of Privacy Loss Distributions”, Doroshenko et al.,
    PoPETs 2022, https://arxiv.org/pdf/2207.04380.pdf
    This class supports Gaussian and Laplacian mechanisms.
    :param value_discretization_interval:
        The length of the dicretization interval for the privacy loss
        distribution. Rounding will occur to integer multiples of
        value_discretization_interval. Smaller values yield more accurate
        estimates of the privacy loss, while incurring higher compute and
        memory. Hence, larger values decrease computation time. Note that the
        accountant algorithm maintains similar error bounds as the value of
        value_discretization_interval is changed.
    :param use_connect_dots:
        boolean indicating whether or not to use Connect-the-Dots algorithm by
        Doroshenko et al., which gives tighter discrete approximations of PLDs.
    :param pessimistic_estimate:
        boolean indicating whether rounding used in PLD algorithm results in
        epsilon-hockey stick divergence computation yielding upper estimate to
        real value.
    :param log_mass_truncation_bound:
        The natural log of probability mass that may be discarded from noise
        distribution.
        Larger values will increase the error.
    """
    value_discretization_interval: float = 1e-4
    use_connect_dots: bool = True
    pessimistic_estimate: bool = True
    log_mass_truncation_bound: float = -50

    def __post_init__(self):
        super().__post_init__()

        assert self.mechanism in [
            'gaussian', 'laplace'
        ], ('Only gaussian and laplace mechanisms are supported.')

        # Epsilon, delta, noise parameter all defined. Nothing to do.
        if [self.epsilon, self.delta, self.noise_parameter].count(None) == 0:
            assert math.isclose(
                self.get_composed_accountant(
                    self.mechanism, self.noise_parameter,
                    self.pessimistic_estimate, self.sampling_probability,
                    self.use_connect_dots, self.value_discretization_interval,
                    self.num_compositions).get_delta_for_epsilon(self.epsilon),
                self.delta,
                rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameter for '
                    'PLDPrivacyAccountant')

        else:
            # Only two of epsilon, delta, noise parameter defined.
            # Compute remaining variable
            if self.noise_parameter:

                composed_pld = self.get_composed_accountant(
                    self.mechanism, self.noise_parameter,
                    self.pessimistic_estimate, self.sampling_probability,
                    self.use_connect_dots, self.value_discretization_interval,
                    self.num_compositions)

                if self.epsilon:
                    self.delta = composed_pld.get_delta_for_epsilon(
                        self.epsilon)
                else:
                    self.epsilon = composed_pld.get_epsilon_for_delta(
                        self.delta)

            else:
                # Do binary search over noise_parameter
                min_bound, max_bound = 0, 100
                confidence_threshold = 1e-8
                rtol = 0.0001

                func = lambda noise_parameter: self.get_composed_accountant(
                    self.mechanism, noise_parameter, self.pessimistic_estimate,
                    self.sampling_probability, self.use_connect_dots, self.
                    value_discretization_interval, self.num_compositions
                ).get_delta_for_epsilon(self.epsilon)
                func_monotonically_increasing = False
                try:
                    self.noise_parameter = binary_search_function(
                        func=func,
                        func_monotonically_increasing=
                        func_monotonically_increasing,
                        target_value=self.delta,
                        min_bound=min_bound,
                        max_bound=max_bound,
                        rtol=rtol,
                        confidence_threshold=confidence_threshold)
                except Exception as e:
                    raise ValueError(
                        'Error occurred during binary search for '
                        'noise_parameter using PLD privacy accountant: '
                        f'{e}') from e

    @staticmethod
    def get_composed_accountant(mechanism, noise_parameter,
                                pessimistic_estimate, sampling_probability,
                                use_connect_dots,
                                value_discretization_interval,
                                num_compositions):

        if mechanism == 'gaussian':
            pld = privacy_loss_distribution.from_gaussian_mechanism(
                standard_deviation=noise_parameter,
                sensitivity=1,
                pessimistic_estimate=pessimistic_estimate,
                sampling_prob=sampling_probability,
                use_connect_dots=use_connect_dots,
                value_discretization_interval=value_discretization_interval)
        elif mechanism == 'laplace':
            pld = privacy_loss_distribution.from_laplace_mechanism(
                parameter=noise_parameter,
                sensitivity=1,
                pessimistic_estimate=pessimistic_estimate,
                sampling_prob=sampling_probability,
                use_connect_dots=use_connect_dots,
                value_discretization_interval=value_discretization_interval)
        else:
            raise ValueError(f'mechanism {mechanism} is not supported.')

        composed_pld = pld.self_compose(num_compositions)
        return composed_pld


@dataclass
class PRVPrivacyAccountant(PrivacyAccountant):
    """
    Privacy Random Variable (PRV) accountant, for heterogeneous composition,
    using prv-accountant package.
    prv-accountant package: https://pypi.org/project/prv-accountant/
    Based on: “Numerical Composition of Differential Privacy”, Gopi et al.,
    2021, https://arxiv.org/pdf/2106.02848.pdf
    The PRV accountant methods compute_delta() and compute_epsilon() return
    a lower bound, an estimated value, and an upper bound for the delta and
    epsilon respectively. The estimated value is used for all further
    computations.
    :param eps_error:
        Maximum permitted error in epsilon. Typically around 0.1.
    :param delta_error:
        Maximum error allowed in delta. Typically around delta * 1e-3
    """
    eps_error: Optional[float] = 0.07
    delta_error: Optional[float] = 1e-10

    def __post_init__(self):
        super().__post_init__()

        # epsilon, delta, noise_parameter all defined
        if [self.epsilon, self.delta, self.noise_parameter].count(None) == 0:
            assert math.isclose(
                self.get_composed_accountant(
                    self.mechanism, self.noise_parameter,
                    self.sampling_probability, self.num_compositions,
                    self.eps_error,
                    self.delta_error).compute_delta(self.epsilon,
                                                    self.num_compositions)[1],
                self.delta,
                rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameter'
                    'for PRVPrivacyAccountant')

        else:
            if self.noise_parameter:

                prv_acc = self.get_composed_accountant(
                    self.mechanism, self.noise_parameter,
                    self.sampling_probability, self.num_compositions,
                    self.eps_error, self.delta_error)

                if self.epsilon:
                    # prv_acc.compute_delta() returns lower bound on delta,
                    # estimate of delta, and upper bound on delta.
                    # Estimate of delta is used.
                    (_, delta_estim,
                     _) = prv_acc.compute_delta(self.epsilon,
                                                self.num_compositions)
                    self.delta = delta_estim
                else:
                    # prv_acc.compute_epsilon() returns lower bound on epsilon,
                    # estimate of epsilon, and upper bound on epsion.
                    # Estimate of epsilon is used.
                    (_, epsilon_estim,
                     _) = prv_acc.compute_epsilon(self.delta,
                                                  self.num_compositions)
                    self.epsilon = epsilon_estim

            else:

                # Do binary search over noise_parameter
                min_bound, max_bound = 0, 100
                confidence_threshold = 1e-8
                rtol = 0.0001

                func = lambda noise_parameter: self.get_composed_accountant(
                    self.mechanism, noise_parameter, self.sampling_probability,
                    self.num_compositions, self.eps_error, self.delta_error
                ).compute_delta(self.epsilon, self.num_compositions)[1]
                func_monotonically_increasing = False
                try:
                    self.noise_parameter = binary_search_function(
                        func=func,
                        func_monotonically_increasing=
                        func_monotonically_increasing,
                        target_value=self.delta,
                        min_bound=min_bound,
                        max_bound=max_bound,
                        rtol=rtol,
                        confidence_threshold=confidence_threshold)
                except Exception as e:
                    raise ValueError(
                        'Error occurred during binary search for '
                        f'noise_parameter using PRV privacy accountant: {e}'
                    ) from e

    @staticmethod
    def get_composed_accountant(mechanism, noise_parameter,
                                sampling_probability, num_compositions,
                                eps_error, delta_error):
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
                                eps_error=eps_error,
                                delta_error=delta_error)

        return acc_prv


@dataclass
class RDPPrivacyAccountant(PrivacyAccountant):
    """
    Privacy accountant using Renyi differential privacy (RDP) from
    dp-accounting package.
    Implementation in dp-accounting: https://github.com/google/differential-privacy/blob/main/python/dp_accounting/rdp/rdp_privacy_accountant.py # pylint: disable=line-too-long
    The default neighbouring relation for the RDP account is "add or remove
    one". The default RDP orders used are:
    ([1 + x / 10. for x in range(1, 100)] + list(range(11, 64)) +
    [128, 256, 512, 1024]).
    """

    def __post_init__(self):
        super().__post_init__()

        # epsilon, delta, noise_parameter all defined
        if [self.epsilon, self.delta, self.noise_parameter].count(None) == 0:
            assert math.isclose(
                self.get_composed_accountant(
                    self.mechanism, self.noise_parameter,
                    self.sampling_probability,
                    self.num_compositions).get_delta(self.epsilon),
                self.delta,
                rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameter '
                    'for RDPPrivacyAccountant')

        else:
            if self.noise_parameter:

                rdp_accountant = self.get_composed_accountant(
                    self.mechanism, self.noise_parameter,
                    self.sampling_probability, self.num_compositions)

                if self.epsilon:
                    self.delta = rdp_accountant.get_delta(self.epsilon)

                else:
                    self.epsilon = rdp_accountant.get_epsilon(self.delta)

            else:
                # Do binary search over noise_parameter
                min_bound, max_bound = 0, 100
                confidence_threshold = 1e-8
                rtol = 0.0001

                func = lambda noise_parameter: self.get_composed_accountant(
                    self.mechanism, noise_parameter, self.sampling_probability,
                    self.num_compositions).get_delta(self.epsilon)
                func_monotonically_increasing = False
                try:
                    self.noise_parameter = binary_search_function(
                        func=func,
                        func_monotonically_increasing=
                        func_monotonically_increasing,
                        target_value=self.delta,
                        min_bound=min_bound,
                        max_bound=max_bound,
                        rtol=rtol,
                        confidence_threshold=confidence_threshold)
                except Exception as e:
                    raise ValueError(
                        'Error occurred during binary search for noise_'
                        '_parameter using RDP privacy accountant: {e}') from e

    @staticmethod
    def get_composed_accountant(mechanism, noise_parameter,
                                sampling_probability, num_compositions):
        if mechanism == 'gaussian':
            event = dp_event.PoissonSampledDpEvent(
                sampling_probability,
                dp_event.GaussianDpEvent(noise_parameter))

        elif mechanism == 'laplace':
            event = dp_event.LaplaceDpEvent(noise_parameter)
            pass

        else:
            raise ValueError(
                f'Mechanism {mechanism} is not supported for Renyi accountant')

        rdp_accountant = rdp_privacy_accountant.RdpAccountant().compose(
            event, int(num_compositions))

        return rdp_accountant


def binary_search_function(func: Callable,
                           func_monotonically_increasing: bool,
                           target_value: float,
                           min_bound: float,
                           max_bound: float,
                           func_guess: Callable = lambda min_bound, max_bound:
                           (min_bound + max_bound) / 2,
                           rtol: float = 0.1,
                           confidence_threshold: float = 1.1):
    """
    Function to perform a binary search to find the value of some input to a
    function such that some target output value is reached.
    :param func:
        Function on which binary search will be performed.
    :param func_monotonically_increasing:
        Boolean indicating function's output increases monotonically with input
        variable being varied during binary search.
    :param target_value:
        Target output value of function for which corresponding input value
        is being searched.
    :param min_bound:
        Minimum value of input parameter for search.
    :param max_bound:
        Maximum value of input parameter for search.
    :param func_guess:
        Function mapping from min_bound, max_bound to a value for the input
        parameter being searched for the next iteration of the binary search.
        Typically, the mapping is to the midpoint, (min_bound + max_bound) / 2
    :param rtol:
        Relative tolerance in error allowed between target value and output of
        function during binary search in order to converge to solution.
    :param confidence_threshold:
        Minimum difference in min_bound and max_bound to consider during binary
        search. Binary search will end if the difference between min_bound and
        max_bound is less than this value.
    """
    assert min_bound < max_bound, (
        f'Invalid search space for binary search with min_bound = {min_bound},'
        'max_bound = {max_bound}')
    assert rtol < 1, f'Invalid value for rtol: {rtol}'

    # TODO include feature to double max bound or half min bound if guess
    # max_bound - min_bound > confidence_threshold but has not converged.
    val = target_value * 3
    guess = func_guess(min_bound, max_bound)
    while abs(val - target_value) / target_value > rtol and (
            max_bound - min_bound > confidence_threshold):
        guess = func_guess(min_bound, max_bound)
        val = func(guess)

        if val == target_value:
            break
        if val < target_value:
            if func_monotonically_increasing:
                min_bound = guess
            else:
                max_bound = guess
        else:
            if func_monotonically_increasing:
                max_bound = guess
            else:
                min_bound = guess

    val = func(guess)
    assert abs(val - target_value) / target_value <= rtol, (
        'Binary search did not converge')

    return guess
