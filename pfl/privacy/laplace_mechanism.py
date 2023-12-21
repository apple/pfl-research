# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
The Laplace mechanism for differential privacy.
'''

from typing import Optional, Tuple

from pfl.hyperparam import HyperParamClsOrFloat, get_param_value
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import TrainingStatistics

from .approximate_mechanism import SquaredErrorLocalPrivacyMechanism
from .privacy_mechanism import CentrallyApplicablePrivacyMechanism, NormClipping


class LaplaceMechanism(CentrallyApplicablePrivacyMechanism, NormClipping,
                       SquaredErrorLocalPrivacyMechanism):
    """
    Apply the Laplace mechanism for differential privacy according to Section
    3.3 in:
    https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf

    The l1 norm is computed over all the arrays and clipped by a bound
    to be able to use Definition 3.1. Thereafter, Laplacian noise,
    parameterized according to Definition 3.3, is added to all arrays.

    :param epsilon:
        The ε parameter of differential privacy.
        This gives an upper bound on the amount of privacy loss.
    """

    def __init__(self, clipping_bound: HyperParamClsOrFloat, epsilon: float):
        NormClipping.__init__(self, 1., clipping_bound)
        self._epsilon = epsilon

    def sensitivity_scaling(self, num_dimensions):
        return 1

    def sensitivity_squared_error(self, num_dimensions: int, l2_norm: float):
        return 0.

    def add_noise_squared_error(self, num_dimensions: int, cohort_size: int):
        noise_scale = get_param_value(self._clipping_bound) / self._epsilon
        variance = 2 * noise_scale**2
        return num_dimensions * variance

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:

        # Compute the noise scale (`b` in Definition 3.3) that parameterizes
        # the Laplace distribution in order to guarantee epsilon privacy
        noise_scale = get_param_value(self._clipping_bound) / self._epsilon

        data_with_noise = statistics.apply(get_ops().add_laplacian_noise,
                                           scale=noise_scale,
                                           seed=seed)

        metrics = Metrics([
            (name_formatting_fn('Laplace DP noise scale'),
             Weighted.from_unweighted(noise_scale)),
        ])

        return data_with_noise, metrics
