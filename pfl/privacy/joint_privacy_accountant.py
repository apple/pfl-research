# Copyright © 2023-2024 Apple Inc.
'''
Joint privacy accountants for differential privacy with multiple mechanisms.
'''

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from dp_accounting import dp_event
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.rdp import rdp_privacy_accountant
from prv_accountant import LaplaceMechanism, PoissonSubsampledGaussianMechanism, PRVAccountant

from .privacy_accountant import (binary_search_function, MAX_BOUND_NOISE_PARAMETER, MIN_BOUND_NOISE_PARAMETER,
                                 RTOL_NOISE_PARAMETER, CONFIDENCE_THRESHOLD_NOISE_PARAMETER,
                                 PLDPrivacyAccountant, PRVPrivacyAccountant, RDPPrivacyAccountant)

MIN_BOUND_EPSILON = 0
MAX_BOUND_EPSILON = 30
RTOL_EPSILON = 0.001
CONFIDENCE_THRESHOLD_EPSILON = 1e-8

PFL_ACCOUNTANT_TYPE = PLDPrivacyAccountant | PRVPrivacyAccountant | RDPPrivacyAccountant


@dataclass
class JointPrivacyAccountant:
    """
    Tracks the privacy loss over multiple mechanisms simultaneously.

    This class can be used in a number of different ways.

    1. An overall budget (total_epsilon, total_delta) is known, and we wish to
    find the corresponding noise parameters of each of the mechanisms so that
    the composition of all mechanisms adheres to the overall budget.
    The budget_proportions parameter specifies what fraction of the total budget
    each mechanism is allocated. For budget_proportions = [p_1, p_2, ...] the
    noise parameters are then computed as follows. When the accountant class
    is the same for each mechanism we find naive_epsilon > total_epsilon and
    noise parameters [sigma_1, sigma_2, ...] such that the following two
    constraints hold:
        A. For each i, mechanism_i with noise parameter sigma_i is
        (naive_epsilon * p_i, total_delta * p_i) DP after all composition steps,
        B. The composition of all mechanisms over all steps is
        (total_epsilon, total_delta) DP.
    When the accountant classes to be used for each mechanism are different we
    can only use basic composition and the noise parameter for each mechanism is computed
    separately so that mechanism_i is (total_epsilon * p_i, total_delta * p_i) DP

    2. The individual mechanism privacy budgets are known, i.e. mechanism_epsilons and
    mechanism_deltas are given. In this case we can compute the required noise parameter
    of each mechanism. The overall budget is then computed as described in 3.

    3. The mechanism noise parameters are known, either because they have been computed
    in 2. or because they have been given directly. In this case we compute the overall
    privacy budget (total_epsilon, total_delta) by efficient composition if possible
    (i.e. when using the same privacy accountant class) and by basic composition if not
    possible.

    :param mechanisms:
        The list of noise mechanisms to be used, each can be either Gaussian or Laplace.
    :param accountants:
        An accountant class, or list of accountant classes, specifying the type of
        accountant being used for each of the mechanisms. Must be one of
        PLDPrivacyAccountant, PRVPrivacyAccountant or RDPPrivacyAccountant.
    :param num_compositions:
        A positive integer, or list of positive integers, specifying the maximum
        number of compositions to be performed with each mechanism.
    :param sampling_probability:
        A probability, or list of probabilities, specifying the sampling rate
        of each mechanism.
    :param total_epsilon:
        The total epsilon allowed for the composition of all the mechanisms.
    :param total_delta:
        The total delta allowed for the composition of all the mechanisms.
    :param mechanism_epsilons:
        The list of individual epsilons allowed for each mechanism.
    :param mechanism_deltas:
        The list of individual deltas allowed for each mechanism.
    :param budget_proportions:
        List specifying the proportion of the overall (total_epsilon, total_delta) privacy budget
        each mechanism is allocated. Defaults to [1 / len(mechanisms)] * len(mechanisms) if not set,
        i.e. even split of the overall budget.
    :param noise_parameters:
        The parameters for DP noise for each mechanism. For the Gaussian
        mechanism, the noise parameter is the standard deviation of the noise.
        For the Laplace mechanism, the noise parameter is the scale of the noise.
    :param noise_scale:
        A value \\in [0, 1] multiplied with the standard deviation of the noise
        to be added for privatization. Typically used to experiment with lower
        sampling probabilities when it is not possible or desirable to increase
        the population size of the units being privatized, e.g. user devices.
    :param pld_accountant_kwargs:
        Optional key word arguments that are fed into the PLDPrivacyAccountant
        constructor to change the default parameters, see privacy_accountant.py
    :param prv_accountant_kwargs:
        Optional key word arguments that are fed into the PRVPrivacyAccountant
        constructor to change the default parameters, see privacy_accountant.py
    """
    mechanisms: List[str]
    accountants: Type[PFL_ACCOUNTANT_TYPE] | List[Type[PFL_ACCOUNTANT_TYPE]]
    num_compositions: int | List[int]
    sampling_probability: float | List[float]
    total_epsilon: Optional[float] = None
    total_delta: Optional[float] = None
    mechanism_epsilons: Optional[List[float]] = None
    mechanism_deltas: Optional[List[float]] = None
    budget_proportions: Optional[List[float]] = None
    noise_parameters: Optional[List[float]] = None
    noise_scale: float = 1.0
    pld_accountant_kwargs: Optional[Dict] = None
    prv_accountant_kwargs: Optional[Dict] = None

    def __post_init__(self):
        self.check_valid_parameter_settings()

        # Set initial min and max bounds for the inner binary search
        self.min_bounds = [MIN_BOUND_NOISE_PARAMETER] * len(self.mechanisms)
        self.max_bounds = [MAX_BOUND_NOISE_PARAMETER] * len(self.mechanisms)

        if self.mechanism_epsilons is not None:
            # Individual mechanism privacy budgets are given and noise parameters
            # can be computed separately
            self.noise_parameters = self.compute_for_each_mechanism(
                mechanism_epsilons=self.mechanism_epsilons,
                mechanism_deltas=self.mechanism_deltas)

        if len(set(self.accountants)) == 1:
            # All accountant classes are the same, hence we can efficiently compose
            accountant_cls = self.accountants[0]
            accountant_method = self.get_accountant_method(accountant_cls)
            accountant_kwargs = self.get_accountant_kwargs(accountant_cls)

            # Total epsilon, total delta, and noise parameters all defined. Check if
            # composition of mechanisms with given noise parameters satisfies
            # (total_epsilon, total_delta)-DP.
            if [self.total_epsilon, self.total_delta, self.noise_parameters].count(None) == 0:
                assert math.isclose(
                    accountant_method(
                         self.mechanisms, self.noise_parameters,
                         self.num_compositions, self.sampling_probability,
                         epsilon=self.total_epsilon, **accountant_kwargs),
                    self.total_delta,
                    rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameters')
            else:
                if self.noise_parameters is not None:
                    if self.total_epsilon:
                        self.total_delta = accountant_method(
                            self.mechanisms, self.noise_parameters,
                            self.num_compositions, self.sampling_probability,
                            epsilon=self.total_epsilon, **accountant_kwargs)
                    else:
                        self.total_epsilon = accountant_method(
                            self.mechanisms, self.noise_parameters,
                            self.num_compositions, self.sampling_probability,
                            delta=self.total_delta, **accountant_kwargs)

                else:
                    # Do binary search over naive_epsilon. Within each iteration of the binary search
                    # we run an inner binary search to compute the noise parameter for each mechanism
                    # that enforce condition 1 from above.
                    def compute_delta(naive_epsilon, delta_method=accountant_method):
                        self.noise_parameters = self.compute_for_each_mechanism(
                            mechanism_epsilons=[naive_epsilon * p for p in self.budget_proportions],
                            mechanism_deltas=[self.total_delta * p for p in self.budget_proportions],
                        )

                        delta = delta_method(
                            self.mechanisms,
                            self.noise_parameters,
                            self.num_compositions,
                            self.sampling_probability,
                            epsilon=self.total_epsilon,
                        )

                        if delta < self.total_delta:
                            # large_epsilon was too small, i.e. noise was too large.
                            # We can decrease our starting max bound for the next noise parameter search
                            self.max_bounds = self.noise_parameters
                        else:
                            # large_epsilon was too large, i.e. noise was too small.
                            # We can increase our starting min bound for the next noise parameter search
                            self.min_bounds = self.noise_parameters

                        return delta

                    try:
                        self.naive_epsilon = binary_search_function(
                            func=compute_delta,
                            func_monotonically_increasing=True,
                            target_value=self.total_delta,
                            min_bound=max(MIN_BOUND_EPSILON, self.total_epsilon),
                            max_bound=min(MAX_BOUND_EPSILON, self.total_epsilon / min(*self.budget_proportions)),
                            rtol=RTOL_EPSILON,
                            confidence_threshold=
                            CONFIDENCE_THRESHOLD_EPSILON)
                    except Exception as e:
                        raise ValueError(
                            'Error occurred during binary search for '
                            'large_epsilon: '
                            f'{e}') from e

        else:
            # Accountant classes are different, we must use basic composition
            if self.mechanism_epsilons is not None:
                self.total_epsilon = sum(self.mechanism_epsilons)
                self.total_delta = sum(self.mechanism_deltas)
            else:
                if [self.total_epsilon, self.total_delta, self.noise_parameters].count(None) == 0:
                    # Check vals are correct
                    pass

                else:
                    if self.noise_parameters is not None:
                        if self.total_epsilon:
                            mechanism_epsilons = [self.total_epsilon * p for p in self.budget_proportions]
                            deltas = self.compute_for_each_mechanism(
                                mechanism_epsilons=mechanism_epsilons,
                                noise_parameters=self.noise_parameters
                            )
                            self.total_delta = sum(deltas)

                        else:
                            assert self.total_delta is not None
                            mechanism_deltas = [self.total_delta * p for p in self.budget_proportions]
                            epsilons = self.compute_for_each_mechanism(
                                mechanism_deltas=mechanism_deltas,
                                noise_parameters=self.noise_parameters
                            )
                            self.total_epsilon = sum(epsilons)

                    else:
                        mechanism_epsilons = [b * self.total_epsilon for b in self.budget_proportions]
                        mechanism_deltas = [b * self.total_delta for b in self.budget_proportions]

                        self.noise_parameters = self.compute_for_each_mechanism(
                            mechanism_epsilons=mechanism_epsilons,
                            mechanism_deltas=mechanism_deltas
                        )

    def compute_for_each_mechanism(self, mechanism_epsilons=None, mechanism_deltas=None, noise_parameters=None):
        """
        Compute for each mechanism independently using the corresponding accountant.
        """

        if noise_parameters is None:
            computed_val = 'noise_parameter'
            noise_parameters = [None] * len(mechanism_epsilons)
        elif mechanism_deltas is None:
            computed_val = 'delta'
            mechanism_deltas = [None] * len(mechanism_epsilons)
        else:
            computed_val = 'epsilon'
            mechanism_epsilons = [None] * len(mechanism_deltas)

        computed_vals = []
        for mechanism, accountant_cls, n, p, e, d, s, min_bound, max_bound in zip(self.mechanisms,
                                                                                  self.accountants,
                                                                                  self.num_compositions,
                                                                                  self.sampling_probability,
                                                                                  mechanism_epsilons,
                                                                                  mechanism_deltas,
                                                                                  noise_parameters,
                                                                                  self.min_bounds,
                                                                                  self.max_bounds):

            accountant = accountant_cls(mechanism=mechanism,
                                        epsilon=e,
                                        delta=d,
                                        noise_parameter=s,
                                        num_compositions=n,
                                        sampling_probability=p,
                                        min_bound_noise_parameter=min_bound,
                                        max_bound_noise_parameter=max_bound,
                                        **self.get_accountant_kwargs(accountant_cls))
            computed_vals.append(getattr(accountant, computed_val))

        return computed_vals

    def get_accountant_kwargs(self, accountant_cls):
        get_kwargs_accountant_map = {
            PLDPrivacyAccountant: self.pld_accountant_kwargs,
            PRVPrivacyAccountant: self.prv_accountant_kwargs,
            RDPPrivacyAccountant: {},
        }
        return get_kwargs_accountant_map[accountant_cls]

    def get_accountant_method(self, accountant_cls):
        get_method_accountant_map = {
            PLDPrivacyAccountant: self.composed_pld_accountant,
            PRVPrivacyAccountant: self.composed_prv_accountant,
            RDPPrivacyAccountant: self.composed_rdp_accountant,
        }
        return get_method_accountant_map[accountant_cls]

    @staticmethod
    def composed_pld_accountant(mechanisms, noise_parameters, num_compositions, sampling_probability,
                                epsilon=None,
                                delta=None,
                                **kwargs
                                ):
        assert [epsilon, delta].count(None) <= 1, (
            'At least one of epsilon and delta must be specified')

        composed_pld = None
        for mechanism, noise_parameter, n, p in zip(mechanisms, noise_parameters, num_compositions, sampling_probability):
            if mechanism == 'gaussian':
                pld = privacy_loss_distribution.from_gaussian_mechanism(
                    standard_deviation=noise_parameter,
                    sensitivity=1,
                    sampling_prob=p,
                    **kwargs)
            elif mechanism == 'laplace':
                pld = privacy_loss_distribution.from_laplace_mechanism(
                    parameter=noise_parameter,
                    sensitivity=1,
                    sampling_prob=p,
                    **kwargs)

            else:
                raise ValueError(f'mechanism {mechanism} is not supported.')

            if composed_pld is None:
                composed_pld = pld.self_compose(n)
            else:
                composed_pld = composed_pld.compose(pld.self_compose(n))

        if delta is None:
            return composed_pld.get_delta_for_epsilon(epsilon)
        else:
            return composed_pld.get_epsilon_for_delta(delta)

    @staticmethod
    def composed_prv_accountant(mechanisms, noise_parameters, num_compositions, sampling_probability,
                                epsilon=None, delta=None,
                                **kwargs):

        assert [epsilon, delta].count(None) <= 1, (
            'At least one of epsilon and delta must be specified')

        prvs = []
        for mechanism, noise_parameter, _, p in zip(mechanisms, noise_parameters, num_compositions, sampling_probability):
            if mechanism == 'gaussian':
                prv = PoissonSubsampledGaussianMechanism(
                    sampling_probability=p,
                    noise_multiplier=noise_parameter)

            elif mechanism == 'laplace':
                prv = LaplaceMechanism(mu=noise_parameter)

            else:
                raise ValueError(
                    f'Mechanism {mechanism} is not supported for PRV accountant')

            prvs.append(prv)

        acc_prv = PRVAccountant(prvs=prvs,
                                max_self_compositions=num_compositions,
                                **kwargs)

        if delta is None:
            return acc_prv.compute_delta(epsilon, num_compositions)[1]
        else:
            return acc_prv.compute_epsilon(delta, num_compositions)[1]

    @staticmethod
    def composed_rdp_accountant(mechanisms, noise_parameters, num_compositions, sampling_probability,
                                epsilon=None, delta=None):

        assert [epsilon, delta].count(None) <= 1, (
            'At least one of epsilon and delta must be specified')

        rdp_accountant = rdp_privacy_accountant.RdpAccountant()
        for mechanism, noise_parameter, n, p in zip(mechanisms, noise_parameters, num_compositions, sampling_probability):
            if mechanism == 'gaussian':
                event = dp_event.PoissonSampledDpEvent(
                    p,
                    dp_event.GaussianDpEvent(noise_parameter))

            elif mechanism == 'laplace':
                event = dp_event.LaplaceDpEvent(noise_parameter)

            else:
                raise ValueError(
                    f'Mechanism {mechanism} is not supported for Renyi accountant')

            rdp_accountant = rdp_accountant.compose(
                event, n)

        if delta is None:
            return rdp_accountant.get_delta(epsilon)
        else:
            return rdp_accountant.get_epsilon(delta)

    @property
    def cohort_noise_parameters(self):
        """
        Noise parameters to be used on a cohort of users.
        Noise scale is considered.
        """
        return [
            noise_parameter * self.noise_scale
            for noise_parameter in self.noise_parameters
        ]

    def check_valid_parameter_settings(self):
        """
        Check that all input parameter settings are valid
        """

        # Check mechanisms
        self.mechanisms = [mechanism.lower() for mechanism in self.mechanisms]
        for mechanism in self.mechanisms:
            assert mechanism in ['gaussian', 'laplace'], (
                'Only gaussian and laplace mechanisms are supported.')

        # Check accountants
        if isinstance(self.accountants, list):
            assert len(self.accountants) == len(self.mechanisms), (
                'Accountants and mechanism names must have the same length'
            )
        else:
            self.accountants = [self.accountants] * len(self.mechanisms)

        assert all(isinstance(acc, PFL_ACCOUNTANT_TYPE) for acc in self.accountants), (
            f'Accountants must be one of the accountant classes {PFL_ACCOUNTANT_TYPE}'
        )

        # Check num_compositions
        if isinstance(self.num_compositions, list):
            assert len(self.num_compositions) == len(self.mechanisms), (
                'Num compostitions and mechanism names must have the same length'
            )
        else:
            self.num_compositions = [self.num_compositions] * len(self.mechanisms)

        assert all(0 < n and isinstance(n, int) for n in self.num_compositions), (
                'Number of compositions must be a positive integer')

        # Check sampling_probability
        if isinstance(self.sampling_probability, list):
            assert len(self.sampling_probability) == len(self.mechanisms), (
                'Sampling probabilites and mechanism names must have the same length'
            )
        else:
            self.sampling_probability = [self.sampling_probability] * len(self.mechanisms)

        assert all(0 <= p <= 1.0 for p in self.sampling_probability), (
            f'Sampling probabilities {self.sampling_probability} are invalid.'
            'Must be in range [0, 1]')

        # Check total_epsilon
        if self.total_epsilon is not None:
            assert self.total_epsilon >= 0, (
                'Epsilon must be a non-negative real value')

        # Check total_delta
        if self.total_delta is not None:
            assert 0 < self.total_delta < 1, (
                'Delta should be a positive real value in range (0, 1)')

        # Check mechanism_epsilons and mechanism_deltas
        if (self.mechanism_epsilons is not None) or (self.mechanism_deltas is not None):
            assert (self.mechanism_epsilons is not None) and (self.mechanism_deltas is not None), (
                'If one of mechanism_epsilons and mechanism_deltas is set then both must be.'
            )

            assert all(eps >= 0 for eps in self.mechanism_epsilons), (
                'All mechanism epsilons must be non-negative real values'
            )

            assert all(0 < delta < 1 for delta in self.mechanism_deltas), (
                'All mechanism deltas must be positive real values in (0, 1)'
            )

            assert len(self.mechanism_epsilons) == len(self.mechanisms), (
                'Mechanism epsilons and mechanism names must have the same length'
            )

            assert len(self.mechanism_deltas) == len(self.mechanisms), (
                'Mechanism deltas and mechanism names must have the same length'
            )

        # Check budget_proportions
        if self.budget_proportions is not None:
            assert len(self.mechanisms) == len(self.budget_proportions), (
                'Mechansim names and budget proportions must have the same length'
            )

            assert all(0 < p < 1.0 for p in self.budget_proportions), (
                    f'Budget proportions {self.budget_proportions} are invalid.'
                    'Must be in range (0, 1)')

            assert math.isclose(sum(self.budget_proportions), 1, rel_tol=1e-3), (
                'Privacy budget proportions must sum to 1.'
            )
        else:
            self.budget_proportions = [1 / len(self.mechanisms)] * len(self.mechanisms)

        # Check noise_parameters
        if self.noise_parameters is not None:
            assert all(0 < s for s in self.noise_parameters), (
                    'All noise parameters must be positive real values.')

        # Check noise_scale
        if self.noise_scale <= 0 or self.noise_scale > 1.0:
            raise ValueError("noise_scale must be in range (0,1]")

        assert [
            self.total_epsilon, self.total_delta, self.noise_parameters
        ].count(None) <= 1, (
            f'At least two of total epsilon ({self.total_epsilon}),'
            f'total delta ({self.total_delta}) and noise parameters ({self.noise_parameters})'
            'must be defined for a joint privacy accountant')

#
#
#
# @dataclass
# class JointPLDPrivacyAccountant(JointPrivacyAccountant):
#     """
#     Uses Privacy Loss Distribution (PLD) privacy accountant, from dp-accounting
#     package for each mechanism.
#
#     :param value_discretization_interval:
#         The length of the dicretization interval for the privacy loss
#         distribution. Rounding will occur to integer multiples of
#         value_discretization_interval. Smaller values yield more accurate
#         estimates of the privacy loss, while incurring higher compute and
#         memory. Hence, larger values decrease computation time. Note that the
#         accountant algorithm maintains similar error bounds as the value of
#         value_discretization_interval is changed.
#     :param use_connect_dots:
#         boolean indicating whether or not to use Connect-the-Dots algorithm by
#         Doroshenko et al., which gives tighter discrete approximations of PLDs.
#     :param pessimistic_estimate:
#         boolean indicating whether rounding used in PLD algorithm results in
#         epsilon-hockey stick divergence computation yielding upper estimate to
#         real value.
#     :param log_mass_truncation_bound:
#         The natural log of probability mass that may be discarded from noise
#         distribution.
#         Larger values will increase the error.
#     """
#     value_discretization_interval: float = 1e-4
#     use_connect_dots: bool = True
#     pessimistic_estimate: bool = True
#     log_mass_truncation_bound: float = -50
#
#     def __post_init__(self):
#         super().__post_init__()
#
#         # Epsilon, delta, noise parameter all defined. Nothing to do.
#         if [self.epsilon, self.delta, self.noise_parameters].count(None) == 0:
#             assert math.isclose(
#                 self.get_composed_accountant(
#                     self.mechanisms, self.noise_parameters,
#                     self.pessimistic_estimate, self.sampling_probability,
#                     self.use_connect_dots, self.value_discretization_interval,
#                     self.num_compositions).get_delta_for_epsilon(self.epsilon),
#                 self.delta,
#                 rel_tol=1e-3), (
#                     'Invalid settings of epsilon, delta, noise_parameters for '
#                     'JointPLDPrivacyAccountant')
#
#         else:
#             # Only two of epsilon, delta, noise parameters defined.
#             # Compute remaining variable
#             if self.noise_parameters:
#                 composed_pld = self.get_composed_accountant(
#                         self.mechanisms, self.noise_parameters,
#                         self.pessimistic_estimate, self.sampling_probability,
#                         self.use_connect_dots, self.value_discretization_interval,
#                         self.num_compositions)
#
#                 if self.epsilon:
#                     self.delta = composed_pld.get_delta_for_epsilon(
#                         self.epsilon)
#                 else:
#                     self.epsilon = composed_pld.get_epsilon_for_delta(
#                         self.delta)
#
#             else:
#                 # Do binary search over large_epsilon. Within each iteration of the binary search
#                 # we run an inner binary search to compute the noise parameter for each mechanism
#                 # that enforce condition 1 from above.
#                 def compute_delta(large_epsilon):
#                     delta = self.get_composed_accountant(
#                         self.mechanisms, self.compute_noise_paramters(large_epsilon), self.pessimistic_estimate,
#                         self.sampling_probability, self.use_connect_dots, self.
#                         value_discretization_interval, self.num_compositions,
#                     ).get_delta_for_epsilon(self.epsilon)
#
#                     if delta < self.delta:
#                         # large_epsilon was too small, i.e. noise was too large.
#                         # We can decrease our starting max bound for the next noise parameter search
#                         self.max_bounds = self.noise_parameters
#                     else:
#                         # large_epsilon was too large, i.e. noise was too small.
#                         # We can increase our starting min bound for the next noise parameter search
#                         self.min_bounds = self.noise_parameters
#
#                     return delta
#
#                 try:
#                     self.large_epsilon = binary_search_function(
#                         func=compute_delta,
#                         func_monotonically_increasing=True,
#                         target_value=self.delta,
#                         min_bound=max(MIN_BOUND_EPSILON, self.epsilon),
#                         max_bound=min(MAX_BOUND_EPSILON, self.epsilon / min(*self.budget_proportions)),
#                         rtol=RTOL_EPSILON,
#                         confidence_threshold=
#                         CONFIDENCE_THRESHOLD_EPSILON)
#                 except Exception as e:
#                     raise ValueError(
#                         'Error occurred during binary search for '
#                         'large_epsilon using PLD privacy accountant: '
#                         f'{e}') from e
#
#     @staticmethod
#     def get_composed_accountant(mechanisms, noise_parameters,
#                                 pessimistic_estimate, sampling_probability,
#                                 use_connect_dots,
#                                 value_discretization_interval,
#                                 num_compositions):
#         composed_pld = None
#         for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
#             if mechanism == 'gaussian':
#                 pld = privacy_loss_distribution.from_gaussian_mechanism(
#                     standard_deviation=noise_parameter,
#                     sensitivity=1,
#                     pessimistic_estimate=pessimistic_estimate,
#                     sampling_prob=sampling_probability,
#                     use_connect_dots=use_connect_dots,
#                     value_discretization_interval=value_discretization_interval)
#             elif mechanism == 'laplace':
#                 pld = privacy_loss_distribution.from_laplace_mechanism(
#                     parameter=noise_parameter,
#                     sensitivity=1,
#                     pessimistic_estimate=pessimistic_estimate,
#                     sampling_prob=sampling_probability,
#                     use_connect_dots=use_connect_dots,
#                     value_discretization_interval=value_discretization_interval)
#
#             else:
#                 raise ValueError(f'mechanism {mechanism} is not supported.')
#
#             if composed_pld is None:
#                 composed_pld = pld.self_compose(num_compositions)
#             else:
#                 composed_pld = composed_pld.compose(pld.self_compose(num_compositions))
#
#         return composed_pld
#
#     def compute_noise_paramters(self, large_epsilon):
#         noise_parameters = []
#
#         for mechanism, p, min_bound, max_bound in zip(self.mechanisms, self.budget_proportions, self.min_bounds, self.max_bounds):
#             mechanism_epsilon = large_epsilon * p
#             func = lambda noise_param: self.get_composed_accountant(
#                 [mechanism], [noise_param], self.pessimistic_estimate,
#                 self.sampling_probability, self.use_connect_dots, self.
#                 value_discretization_interval, self.num_compositions,
#             ).get_delta_for_epsilon(mechanism_epsilon)
#             try:
#                 noise_parameter = binary_search_function(
#                     func=func,
#                     func_monotonically_increasing=False,
#                     target_value=self.delta,
#                     min_bound=min_bound,
#                     max_bound=max_bound,
#                     rtol=RTOL_NOISE_PARAMETER,
#                     confidence_threshold=
#                     CONFIDENCE_THRESHOLD_NOISE_PARAMETER)
#             except Exception as e:
#                 raise ValueError(
#                     'Error occurred during binary search for '
#                     'noise_parameter using PLD privacy accountant: '
#                     f'{e}') from e
#
#             noise_parameters.append(noise_parameter)
#
#         self.noise_parameters = noise_parameters
#         return noise_parameters
#
#
# @dataclass
# class JointPRVPrivacyAccountant(JointPrivacyAccountant):
#     """
#     Privacy Random Variable (PRV) accountant, for heterogeneous composition,
#     using prv-accountant package.
#     prv-accountant package: https://pypi.org/project/prv-accountant/
#     Based on: “Numerical Composition of Differential Privacy”, Gopi et al.,
#     2021, https://arxiv.org/pdf/2106.02848.pdf
#     The PRV accountant methods compute_delta() and compute_epsilon() return
#     a lower bound, an estimated value, and an upper bound for the delta and
#     epsilon respectively. The estimated value is used for all further
#     computations.
#
#     :param eps_error:
#         Maximum permitted error in epsilon. Typically around 0.1.
#     :param delta_error:
#         Maximum error allowed in delta. Typically around delta * 1e-3
#     """
#     eps_error: Optional[float] = 0.07
#     delta_error: Optional[float] = 1e-10
#
#     def __post_init__(self):
#         super().__post_init__()
#
#         # epsilon, delta, noise_parameter all defined
#         if [self.epsilon, self.delta, self.noise_parameters].count(None) == 0:
#             assert math.isclose(
#                 self.get_composed_accountant(
#                     self.mechanisms, self.noise_parameters,
#                     self.sampling_probability, self.num_compositions,
#                     self.eps_error,
#                     self.delta_error).compute_delta(self.epsilon,
#                                                     [self.num_compositions] * len(self.mechanisms))[1],
#                 self.delta,
#                 rel_tol=1e-3), (
#                     'Invalid settings of epsilon, delta, noise_parameter'
#                     'for PRVPrivacyAccountant')
#
#         else:
#             if self.noise_parameters:
#                 prv_acc = self.get_composed_accountant(
#                     self.mechanisms, self.noise_parameters,
#                     self.sampling_probability, self.num_compositions,
#                     self.eps_error, self.delta_error)
#
#                 if self.epsilon:
#                     # prv_acc.compute_delta() returns lower bound on delta,
#                     # estimate of delta, and upper bound on delta.
#                     # Estimate of delta is used.
#                     (_, delta_estim,
#                      _) = prv_acc.compute_delta(self.epsilon,
#                                                 [self.num_compositions] * len(self.mechanisms))
#                     self.delta = delta_estim
#                 else:
#                     # prv_acc.compute_epsilon() returns lower bound on epsilon,
#                     # estimate of epsilon, and upper bound on epsion.
#                     # Estimate of epsilon is used.
#                     (_, epsilon_estim,
#                      _) = prv_acc.compute_epsilon(self.delta,
#                                                   [self.num_compositions] * len(self.mechanisms))
#                     self.epsilon = epsilon_estim
#
#             else:
#                 # Do binary search over large_epsilon. Within each iteration of the binary search
#                 # we run an inner binary search to compute the noise parameter for each mechanism
#                 # that enforce condition 1 from above.
#                 def compute_delta(large_epsilon):
#                     delta = self.get_composed_accountant(
#                         self.mechanisms, self.compute_noise_paramters(large_epsilon),
#                         self.sampling_probability, self.num_compositions,
#                         self.eps_error, self.delta_error,
#                     ).compute_delta(self.epsilon, [self.num_compositions] * len(self.mechanisms))[1]
#
#                     if delta < self.delta:
#                         # large_epsilon was too small, i.e. noise was too large.
#                         # We can decrease our starting max bound for the next noise parameter search
#                         self.max_bounds = self.noise_parameters
#                     else:
#                         # large_epsilon was too large, i.e. noise was too small.
#                         # We can increase our starting min bound for the next noise parameter search
#                         self.min_bounds = self.noise_parameters
#
#                     return delta
#
#                 try:
#                     self.large_epsilon = binary_search_function(
#                         func=compute_delta,
#                         func_monotonically_increasing=True,
#                         target_value=self.delta,
#                         min_bound=max(MIN_BOUND_EPSILON, self.epsilon),
#                         max_bound=min(MAX_BOUND_EPSILON, self.epsilon / min(*self.budget_proportions)),
#                         rtol=RTOL_EPSILON,
#                         confidence_threshold=
#                         CONFIDENCE_THRESHOLD_EPSILON)
#                 except Exception as e:
#                     raise ValueError(
#                         'Error occurred during binary search for '
#                         'large_epsilon using PRV privacy accountant: '
#                         f'{e}') from e
#
#     @staticmethod
#     def get_composed_accountant(mechanisms, noise_parameters,
#                                 sampling_probability, num_compositions,
#                                 eps_error, delta_error):
#
#         prvs = []
#         for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
#             if mechanism == 'gaussian':
#                 prv = PoissonSubsampledGaussianMechanism(
#                     sampling_probability=sampling_probability,
#                     noise_multiplier=noise_parameter)
#
#             elif mechanism == 'laplace':
#                 prv = LaplaceMechanism(mu=noise_parameter)
#
#             else:
#                 raise ValueError(
#                     f'Mechanism {mechanism} is not supported for PRV accountant')
#
#             prvs.append(prv)
#
#         acc_prv = PRVAccountant(prvs=prvs,
#                                 max_self_compositions=[int(num_compositions)] * len(prvs),
#                                 eps_error=eps_error,
#                                 delta_error=delta_error)
#
#         return acc_prv
#
#     def compute_noise_paramters(self, large_epsilon):
#         noise_parameters = []
#
#         for mechanism, p, min_bound, max_bound in zip(self.mechanisms, self.budget_proportions, self.min_bounds, self.max_bounds):
#             mechanism_epsilon = large_epsilon * p
#             func = lambda noise_param: self.get_composed_accountant(
#                 [mechanism], [noise_param],
#                     self.sampling_probability, self.num_compositions,
#                     self.eps_error, self.delta_error,
#             ).compute_delta(mechanism_epsilon, [self.num_compositions])[1]
#             try:
#                 noise_parameter = binary_search_function(
#                     func=func,
#                     func_monotonically_increasing=False,
#                     target_value=self.delta,
#                     min_bound=min_bound,
#                     max_bound=max_bound,
#                     rtol=RTOL_NOISE_PARAMETER,
#                     confidence_threshold=
#                     CONFIDENCE_THRESHOLD_NOISE_PARAMETER)
#             except Exception as e:
#                 raise ValueError(
#                     'Error occurred during binary search for '
#                     'noise_parameter using PRV privacy accountant: '
#                     f'{e}') from e
#
#             noise_parameters.append(noise_parameter)
#
#         self.noise_parameters = noise_parameters
#         return noise_parameters
#
#
# @dataclass
# class JointRDPPrivacyAccountant(JointPrivacyAccountant):
#     """
#     Privacy accountant using Renyi differential privacy (RDP) from
#     dp-accounting package.
#     Implementation in dp-accounting: https://github.com/google/differential-privacy/blob/main/python/dp_accounting/rdp/rdp_privacy_accountant.py # pylint: disable=line-too-long
#     The default neighbouring relation for the RDP account is "add or remove
#     one". The default RDP orders used are:
#     ([1 + x / 10. for x in range(1, 100)] + list(range(11, 64)) +
#     [128, 256, 512, 1024]).
#     """
#
#     def __post_init__(self):
#         super().__post_init__()
#
#         # epsilon, delta, noise_parameters all defined
#         if [self.epsilon, self.delta, self.noise_parameters].count(None) == 0:
#             assert math.isclose(
#                 self.get_composed_accountant(
#                     self.mechanisms, self.noise_parameters,
#                     self.sampling_probability,
#                     self.num_compositions).get_delta(self.epsilon),
#                 self.delta,
#                 rel_tol=1e-3), (
#                     'Invalid settings of epsilon, delta, noise_parameter '
#                     'for RDPPrivacyAccountant')
#
#         else:
#             if self.noise_parameters:
#
#                 rdp_accountant = self.get_composed_accountant(
#                     self.mechanisms, self.noise_parameters,
#                     self.sampling_probability, self.num_compositions)
#
#                 if self.epsilon:
#                     self.delta = rdp_accountant.get_delta(self.epsilon)
#
#                 else:
#                     self.epsilon = rdp_accountant.get_epsilon(self.delta)
#
#             else:
#                 # Do binary search over large_epsilon. Within each iteration of the binary search
#                 # we run an inner binary search to compute the noise parameter for each mechanism
#                 # that enforce condition 1 from above.
#                 def compute_delta(large_epsilon):
#                     delta = self.get_composed_accountant(
#                         self.mechanisms, self.compute_noise_paramters(large_epsilon),
#                         self.sampling_probability, self.num_compositions
#                     ).get_delta(self.epsilon)
#
#                     if delta < self.delta:
#                         # large_epsilon was too small, i.e. noise was too large.
#                         # We can decrease our starting max bound for the next noise parameter search
#                         self.max_bounds = self.noise_parameters
#                     else:
#                         # large_epsilon was too large, i.e. noise was too small.
#                         # We can increase our starting min bound for the next noise parameter search
#                         self.min_bounds = self.noise_parameters
#
#                     return delta
#
#                 try:
#                     self.large_epsilon = binary_search_function(
#                         func=compute_delta,
#                         func_monotonically_increasing=True,
#                         target_value=self.delta,
#                         min_bound=max(MIN_BOUND_EPSILON, self.epsilon),
#                         max_bound=min(MAX_BOUND_EPSILON, self.epsilon / min(*self.budget_proportions)),
#                         rtol=RTOL_EPSILON,
#                         confidence_threshold=
#                         CONFIDENCE_THRESHOLD_EPSILON)
#                 except Exception as e:
#                     raise ValueError(
#                         'Error occurred during binary search for '
#                         'large_epsilon using RDP privacy accountant: '
#                         f'{e}') from e
#
#     @staticmethod
#     def get_composed_accountant(mechanisms, noise_parameters,
#                                 sampling_probability, num_compositions):
#
#         rdp_accountant = rdp_privacy_accountant.RdpAccountant()
#         for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
#             if mechanism == 'gaussian':
#                 event = dp_event.PoissonSampledDpEvent(
#                     sampling_probability,
#                     dp_event.GaussianDpEvent(noise_parameter))
#
#             elif mechanism == 'laplace':
#                 event = dp_event.LaplaceDpEvent(noise_parameter)
#                 pass
#
#             else:
#                 raise ValueError(
#                     f'Mechanism {mechanism} is not supported for Renyi accountant')
#
#             rdp_accountant = rdp_accountant.compose(
#                 event, int(num_compositions))
#
#         return rdp_accountant
#
#     def compute_noise_paramters(self, large_epsilon):
#         noise_parameters = []
#
#         for mechanism, p, min_bound, max_bound in zip(self.mechanisms, self.budget_proportions, self.min_bounds, self.max_bounds):
#             mechanism_epsilon = large_epsilon * p
#             func = lambda noise_param: self.get_composed_accountant(
#                 [mechanism], [noise_param],
#                 self.sampling_probability, self.num_compositions
#             ).get_delta(mechanism_epsilon)
#             try:
#                 noise_parameter = binary_search_function(
#                     func=func,
#                     func_monotonically_increasing=False,
#                     target_value=self.delta,
#                     min_bound=min_bound,
#                     max_bound=max_bound,
#                     rtol=RTOL_NOISE_PARAMETER,
#                     confidence_threshold=
#                     CONFIDENCE_THRESHOLD_NOISE_PARAMETER)
#             except Exception as e:
#                 raise ValueError(
#                     'Error occurred during binary search for '
#                     'noise_parameter using RDP privacy accountant: '
#                     f'{e}') from e
#
#             noise_parameters.append(noise_parameter)
#
#         self.noise_parameters = noise_parameters
#         return noise_parameters
