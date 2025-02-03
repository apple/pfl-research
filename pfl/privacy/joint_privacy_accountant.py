# Copyright © 2023-2024 Apple Inc.
'''
Joint privacy accountants for differential privacy with multiple mechanisms.
'''

import math
from dataclasses import dataclass
from typing import List, Optional

from dp_accounting import dp_event
from dp_accounting.pld import privacy_loss_distribution
from dp_accounting.rdp import rdp_privacy_accountant
from prv_accountant import LaplaceMechanism, PoissonSubsampledGaussianMechanism, PRVAccountant

from .privacy_accountant import binary_search_function

MIN_BOUND_NOISE_PARAMETER = 0
MAX_BOUND_NOISE_PARAMETER = 100
RTOL_NOISE_PARAMETER = 0.001
CONFIDENCE_THRESHOLD_NOISE_PARAMETER = 1e-8

MIN_BOUND_EPSILON = 0
MAX_BOUND_EPSILON = 30
RTOL_EPSILON = 0.001
CONFIDENCE_THRESHOLD_EPSILON = 1e-8


@dataclass
class JointPrivacyAccountant:
    """
    Tracks the privacy loss over multiple composition steps with multiple
    mechanisms simultaneously. Either two or three of the variables epsilon,
    delta and noise_parameters must be defined.
    If all three are defined a check will be performed to make sure a valid set of
    variable values has been provided.
    If two are defined, the remaining variable can be computed. In the case that
    epsilon and delta are defined then the budget_proportions parameter
    must be provided, specifying what fraction of the total budget each
    mechanism is allocated. For budget_proportions = [p_1, p_2, ...]
    the noise parameters are then computed as follows. We find large_epsilon
    and noise parameters [sigma_1, sigma_2, ...] such that the following two
    constraints hold:
    1. For each i, mechanism_i with noise parameter sigma_i is (large_epsilon * p_i, delta)
    DP after all composition steps,
    2. The composition of all mechanisms over all steps is (epsilon, delta) DP.

    :param num_compositions:
        Maximum number of compositions to be performed with each mechanism.
    :param sampling_probability:
       Maximum probability of sampling each entity being privatized. E.g. if
       the unit of privacy is one device, this is the probability of each
       device participating.
    :param mechanisms:
        The list of noise mechanisms to be used, each can be either Gaussian or Laplace.
    :param epsilon:
        The privacy loss random variable. The total epsilon allowed for
        the composition of all the mechanisms.
    :param delta:
        The probability that all privacy will be lost. The total delta allowed for
        the composition of all the mechanisms.
    :param budget_proportions:
        List specifying the proportion of the total (epsilon, delta) privacy budget
        each mechanism is allocated.
    :param noise_parameters:
        The parameters for DP noise for each mechanism. For the Gaussian
        mechanism, the noise parameter is the standard deviation of the noise.
        For the Laplace mechanism, the noise parameter is the scale of the noise.
    :param noise_scale:
        A value \\in [0, 1] multiplied with the standard deviation of the noise
        to be added for privatization. Typically used to experiment with lower
        sampling probabilities when it is not possible or desirable to increase
        the population size of the units being privatized, e.g. user devices.
    """
    num_compositions: int
    sampling_probability: float
    mechanisms: List[str]
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    budget_proportions: Optional[List[float]] = None
    noise_parameters: Optional[List[float]] = None
    noise_scale: float = 1.0

    def __post_init__(self):
        assert [
            self.epsilon, self.delta, self.noise_parameters
        ].count(None) <= 1, (
            f'At least two of epsilon ({self.epsilon}),'
            f'delta ({self.delta}) and noise parameters ({self.noise_parameters})'
            'must be defined for a joint privacy accountant')
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
        if self.noise_parameters is not None:
            for noise_parameter in self.noise_parameters:
                assert noise_parameter > 0, (
                    'All noise parameters must be positive real values')
        if self.epsilon is not None:
            assert self.epsilon >= 0, (
                'Epsilon must be a non-negative real value')
        if self.delta is not None:
            assert self.delta > 0 and self.delta < 1, (
                'Delta should be a positive real value in range (0, 1)')

        self.mechanisms = [mechanism.lower() for mechanism in self.mechanisms]

        for mechanism in self.mechanisms:
            assert mechanism in [
                'gaussian', 'laplace'
            ], ('Only gaussian and laplace mechanisms are supported')

        if self.budget_proportions:
            assert len(self.mechanisms) == len(self.budget_proportions), (
                'Mechansim names and budget proportions must have the same length'
            )

            assert math.isclose(
                sum(self.budget_proportions), 1,
                rel_tol=1e-3), ('Privacy budget proportions must sum to 1')

            for p in self.budget_proportions:
                assert (p > 0) and (p < 1), (
                    'Privacy budget proportions must be in range (0, 1)')

        self.min_bounds = [MIN_BOUND_NOISE_PARAMETER] * len(self.mechanisms)
        self.max_bounds = [MAX_BOUND_NOISE_PARAMETER] * len(self.mechanisms)

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


@dataclass
class JointPLDPrivacyAccountant(JointPrivacyAccountant):
    """
    Uses Privacy Loss Distribution (PLD) privacy accountant, from dp-accounting
    package for each mechanism.

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

        # Epsilon, delta, noise parameter all defined. Nothing to do.
        if [self.epsilon, self.delta, self.noise_parameters].count(None) == 0:
            assert math.isclose(
                self.get_composed_accountant(
                    self.mechanisms, self.noise_parameters,
                    self.pessimistic_estimate, self.sampling_probability,
                    self.use_connect_dots, self.value_discretization_interval,
                    self.num_compositions).get_delta_for_epsilon(self.epsilon),
                self.delta,
                rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameters for '
                    'JointPLDPrivacyAccountant')

        else:
            # Only two of epsilon, delta, noise parameters defined.
            # Compute remaining variable
            if self.noise_parameters:
                composed_pld = self.get_composed_accountant(
                    self.mechanisms, self.noise_parameters,
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
                # Do binary search over large_epsilon. Within each iteration of the binary search
                # we run an inner binary search to compute the noise parameter for each mechanism
                # that enforce condition 1 above.
                def compute_delta(large_epsilon):
                    delta = self.get_composed_accountant(
                        self.mechanisms,
                        self.compute_noise_paramters(large_epsilon),
                        self.pessimistic_estimate,
                        self.sampling_probability,
                        self.use_connect_dots,
                        self.value_discretization_interval,
                        self.num_compositions,
                    ).get_delta_for_epsilon(self.epsilon)

                    if delta < self.delta:
                        # large_epsilon was too small, i.e. noise was too large.
                        # We can decrease our starting max bound for the next noise parameter search
                        self.max_bounds = self.noise_parameters
                    else:
                        # large_epsilon was too large, i.e. noise was too small.
                        # We can increase our starting min bound for the next noise parameter search
                        self.min_bounds = self.noise_parameters

                    return delta

                try:
                    self.large_epsilon = binary_search_function(
                        func=compute_delta,
                        func_monotonically_increasing=True,
                        target_value=self.delta,
                        min_bound=max(MIN_BOUND_EPSILON, self.epsilon),
                        max_bound=min(
                            MAX_BOUND_EPSILON,
                            self.epsilon / min(*self.budget_proportions)),
                        rtol=RTOL_EPSILON,
                        confidence_threshold=CONFIDENCE_THRESHOLD_EPSILON)
                except Exception as e:
                    raise ValueError(
                        'Error occurred during binary search for '
                        'large_epsilon using PLD privacy accountant: '
                        f'{e}') from e

    @staticmethod
    def get_composed_accountant(mechanisms, noise_parameters,
                                pessimistic_estimate, sampling_probability,
                                use_connect_dots,
                                value_discretization_interval,
                                num_compositions):
        composed_pld = None
        for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
            if mechanism == 'gaussian':
                pld = privacy_loss_distribution.from_gaussian_mechanism(
                    standard_deviation=noise_parameter,
                    sensitivity=1,
                    pessimistic_estimate=pessimistic_estimate,
                    sampling_prob=sampling_probability,
                    use_connect_dots=use_connect_dots,
                    value_discretization_interval=value_discretization_interval
                )
            elif mechanism == 'laplace':
                pld = privacy_loss_distribution.from_laplace_mechanism(
                    parameter=noise_parameter,
                    sensitivity=1,
                    pessimistic_estimate=pessimistic_estimate,
                    sampling_prob=sampling_probability,
                    use_connect_dots=use_connect_dots,
                    value_discretization_interval=value_discretization_interval
                )

            else:
                raise ValueError(f'mechanism {mechanism} is not supported.')

            if composed_pld is None:
                composed_pld = pld.self_compose(num_compositions)
            else:
                composed_pld = composed_pld.compose(
                    pld.self_compose(num_compositions))

        return composed_pld

    def compute_noise_paramters(self, large_epsilon):
        """
        Compute noise parameter for each mechanism such that when self composed
        it is (large_epsilon * p, delta) DP, where p is the budget proportion
        of the mechanism
        """
        noise_parameters = []

        for mechanism, p, min_bound, max_bound in zip(self.mechanisms,
                                                      self.budget_proportions,
                                                      self.min_bounds,
                                                      self.max_bounds):
            mechanism_epsilon = large_epsilon * p
            func = lambda noise_param: self.get_composed_accountant(
                [mechanism],
                [noise_param],
                self.pessimistic_estimate,
                self.sampling_probability,
                self.use_connect_dots,
                self.value_discretization_interval,
                self.num_compositions,
            ).get_delta_for_epsilon(mechanism_epsilon)
            try:
                noise_parameter = binary_search_function(
                    func=func,
                    func_monotonically_increasing=False,
                    target_value=self.delta,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    rtol=RTOL_NOISE_PARAMETER,
                    confidence_threshold=CONFIDENCE_THRESHOLD_NOISE_PARAMETER)
            except Exception as e:
                raise ValueError(
                    'Error occurred during binary search for '
                    'noise_parameter using PLD privacy accountant: '
                    f'{e}') from e

            noise_parameters.append(noise_parameter)

        self.noise_parameters = noise_parameters
        return noise_parameters


@dataclass
class JointPRVPrivacyAccountant(JointPrivacyAccountant):
    """
    For each mechanism uses the Privacy Random Variable (PRV) accountant,
    for heterogeneous composition, using prv-accountant package.
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
        if [self.epsilon, self.delta, self.noise_parameters].count(None) == 0:
            assert math.isclose(
                self.get_composed_accountant(
                    self.mechanisms, self.noise_parameters,
                    self.sampling_probability, self.num_compositions,
                    self.eps_error, self.delta_error).compute_delta(
                        self.epsilon,
                        [self.num_compositions] * len(self.mechanisms))[1],
                self.delta,
                rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameter'
                    'for PRVPrivacyAccountant')

        else:
            if self.noise_parameters:
                prv_acc = self.get_composed_accountant(
                    self.mechanisms, self.noise_parameters,
                    self.sampling_probability, self.num_compositions,
                    self.eps_error, self.delta_error)

                if self.epsilon:
                    # prv_acc.compute_delta() returns lower bound on delta,
                    # estimate of delta, and upper bound on delta.
                    # Estimate of delta is used.
                    (_, delta_estim, _) = prv_acc.compute_delta(
                        self.epsilon,
                        [self.num_compositions] * len(self.mechanisms))
                    self.delta = delta_estim
                else:
                    # prv_acc.compute_epsilon() returns lower bound on epsilon,
                    # estimate of epsilon, and upper bound on epsion.
                    # Estimate of epsilon is used.
                    (_, epsilon_estim, _) = prv_acc.compute_epsilon(
                        self.delta,
                        [self.num_compositions] * len(self.mechanisms))
                    self.epsilon = epsilon_estim

            else:
                # Do binary search over large_epsilon. Within each iteration of the binary search
                # we run an inner binary search to compute the noise parameter for each mechanism
                # that enforce condition 1 from above.
                def compute_delta(large_epsilon):
                    delta = self.get_composed_accountant(
                        self.mechanisms,
                        self.compute_noise_paramters(large_epsilon),
                        self.sampling_probability,
                        self.num_compositions,
                        self.eps_error,
                        self.delta_error,
                    ).compute_delta(self.epsilon, [self.num_compositions] *
                                    len(self.mechanisms))[1]

                    if delta < self.delta:
                        # large_epsilon was too small, i.e. noise was too large.
                        # We can decrease our starting max bound for the next noise parameter search
                        self.max_bounds = self.noise_parameters
                    else:
                        # large_epsilon was too large, i.e. noise was too small.
                        # We can increase our starting min bound for the next noise parameter search
                        self.min_bounds = self.noise_parameters

                    return delta

                try:
                    self.large_epsilon = binary_search_function(
                        func=compute_delta,
                        func_monotonically_increasing=True,
                        target_value=self.delta,
                        min_bound=max(MIN_BOUND_EPSILON, self.epsilon),
                        max_bound=min(
                            MAX_BOUND_EPSILON,
                            self.epsilon / min(*self.budget_proportions)),
                        rtol=RTOL_EPSILON,
                        confidence_threshold=CONFIDENCE_THRESHOLD_EPSILON)
                except Exception as e:
                    raise ValueError(
                        'Error occurred during binary search for '
                        'large_epsilon using PRV privacy accountant: '
                        f'{e}') from e

    @staticmethod
    def get_composed_accountant(mechanisms, noise_parameters,
                                sampling_probability, num_compositions,
                                eps_error, delta_error):

        prvs = []
        for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
            if mechanism == 'gaussian':
                prv = PoissonSubsampledGaussianMechanism(
                    sampling_probability=sampling_probability,
                    noise_multiplier=noise_parameter)

            elif mechanism == 'laplace':
                prv = LaplaceMechanism(mu=noise_parameter)

            else:
                raise ValueError(
                    f'Mechanism {mechanism} is not supported for PRV accountant'
                )

            prvs.append(prv)

        acc_prv = PRVAccountant(prvs=prvs,
                                max_self_compositions=[int(num_compositions)] *
                                len(prvs),
                                eps_error=eps_error,
                                delta_error=delta_error)

        return acc_prv

    def compute_noise_paramters(self, large_epsilon):
        noise_parameters = []

        for mechanism, p, min_bound, max_bound in zip(self.mechanisms,
                                                      self.budget_proportions,
                                                      self.min_bounds,
                                                      self.max_bounds):
            mechanism_epsilon = large_epsilon * p
            func = lambda noise_param: self.get_composed_accountant(
                [mechanism],
                [noise_param],
                self.sampling_probability,
                self.num_compositions,
                self.eps_error,
                self.delta_error,
            ).compute_delta(mechanism_epsilon, [self.num_compositions])[1]
            try:
                noise_parameter = binary_search_function(
                    func=func,
                    func_monotonically_increasing=False,
                    target_value=self.delta,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    rtol=RTOL_NOISE_PARAMETER,
                    confidence_threshold=CONFIDENCE_THRESHOLD_NOISE_PARAMETER)
            except Exception as e:
                raise ValueError(
                    'Error occurred during binary search for '
                    'noise_parameter using PRV privacy accountant: '
                    f'{e}') from e

            noise_parameters.append(noise_parameter)

        self.noise_parameters = noise_parameters
        return noise_parameters


@dataclass
class JointRDPPrivacyAccountant(JointPrivacyAccountant):
    """
    For each mechanism uses the Privacy accountant using Renyi differential
    privacy (RDP) from dp-accounting package.
    Implementation in dp-accounting: https://github.com/google/differential-privacy/blob/main/python/dp_accounting/rdp/rdp_privacy_accountant.py # pylint: disable=line-too-long
    The default neighbouring relation for the RDP account is "add or remove
    one". The default RDP orders used are:
    ([1 + x / 10. for x in range(1, 100)] + list(range(11, 64)) +
    [128, 256, 512, 1024]).
    """

    def __post_init__(self):
        super().__post_init__()

        # epsilon, delta, noise_parameters all defined
        if [self.epsilon, self.delta, self.noise_parameters].count(None) == 0:
            assert math.isclose(
                self.get_composed_accountant(
                    self.mechanisms, self.noise_parameters,
                    self.sampling_probability,
                    self.num_compositions).get_delta(self.epsilon),
                self.delta,
                rel_tol=1e-3), (
                    'Invalid settings of epsilon, delta, noise_parameter '
                    'for RDPPrivacyAccountant')

        else:
            if self.noise_parameters:

                rdp_accountant = self.get_composed_accountant(
                    self.mechanisms, self.noise_parameters,
                    self.sampling_probability, self.num_compositions)

                if self.epsilon:
                    self.delta = rdp_accountant.get_delta(self.epsilon)

                else:
                    self.epsilon = rdp_accountant.get_epsilon(self.delta)

            else:
                # Do binary search over large_epsilon. Within each iteration of the binary search
                # we run an inner binary search to compute the noise parameter for each mechanism
                # that enforce condition 1 from above.
                def compute_delta(large_epsilon):
                    delta = self.get_composed_accountant(
                        self.mechanisms,
                        self.compute_noise_paramters(large_epsilon),
                        self.sampling_probability,
                        self.num_compositions).get_delta(self.epsilon)

                    if delta < self.delta:
                        # large_epsilon was too small, i.e. noise was too large.
                        # We can decrease our starting max bound for the next noise parameter search
                        self.max_bounds = self.noise_parameters
                    else:
                        # large_epsilon was too large, i.e. noise was too small.
                        # We can increase our starting min bound for the next noise parameter search
                        self.min_bounds = self.noise_parameters

                    return delta

                try:
                    self.large_epsilon = binary_search_function(
                        func=compute_delta,
                        func_monotonically_increasing=True,
                        target_value=self.delta,
                        min_bound=max(MIN_BOUND_EPSILON, self.epsilon),
                        max_bound=min(
                            MAX_BOUND_EPSILON,
                            self.epsilon / min(*self.budget_proportions)),
                        rtol=RTOL_EPSILON,
                        confidence_threshold=CONFIDENCE_THRESHOLD_EPSILON)
                except Exception as e:
                    raise ValueError(
                        'Error occurred during binary search for '
                        'large_epsilon using RDP privacy accountant: '
                        f'{e}') from e

    @staticmethod
    def get_composed_accountant(mechanisms, noise_parameters,
                                sampling_probability, num_compositions):

        rdp_accountant = rdp_privacy_accountant.RdpAccountant()
        for mechanism, noise_parameter in zip(mechanisms, noise_parameters):
            if mechanism == 'gaussian':
                event = dp_event.PoissonSampledDpEvent(
                    sampling_probability,
                    dp_event.GaussianDpEvent(noise_parameter))

            elif mechanism == 'laplace':
                event = dp_event.LaplaceDpEvent(noise_parameter)
                pass

            else:
                raise ValueError(
                    f'Mechanism {mechanism} is not supported for Renyi accountant'
                )

            rdp_accountant = rdp_accountant.compose(event,
                                                    int(num_compositions))

        return rdp_accountant

    def compute_noise_paramters(self, large_epsilon):
        noise_parameters = []

        for mechanism, p, min_bound, max_bound in zip(self.mechanisms,
                                                      self.budget_proportions,
                                                      self.min_bounds,
                                                      self.max_bounds):
            mechanism_epsilon = large_epsilon * p
            func = lambda noise_param: self.get_composed_accountant(
                [mechanism], [noise_param], self.sampling_probability, self.
                num_compositions).get_delta(mechanism_epsilon)
            try:
                noise_parameter = binary_search_function(
                    func=func,
                    func_monotonically_increasing=False,
                    target_value=self.delta,
                    min_bound=min_bound,
                    max_bound=max_bound,
                    rtol=RTOL_NOISE_PARAMETER,
                    confidence_threshold=CONFIDENCE_THRESHOLD_NOISE_PARAMETER)
            except Exception as e:
                raise ValueError(
                    'Error occurred during binary search for '
                    'noise_parameter using RDP privacy accountant: '
                    f'{e}') from e

            noise_parameters.append(noise_parameter)

        self.noise_parameters = noise_parameters
        return noise_parameters
