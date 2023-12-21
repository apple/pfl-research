# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
The PrivUnit_2 mechanism for differential privacy.
'''

from typing import Optional, Tuple

import numpy as np

from pfl.internal import priv_unit_functions
from pfl.internal.ops.numpy_ops import NumpySeedScope
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.internal.random.draw_from_cap import draw_random_unit_vector
from pfl.metrics import Metrics, StringMetricName, Weighted
from pfl.stats import TrainingStatistics

from .approximate_mechanism import SquaredErrorLocalPrivacyMechanism
from .privacy_snr import SNRMetric


class PrivUnitMechanism(SquaredErrorLocalPrivacyMechanism):
    """
    Apply the PrivUnit_2 mechanism (with the optimal parameters) to a Numpy
    array.

    Implements Algorithm1 from
    Bhowmick et al. (2018) "Protection Against Reconstruction and
    Its Applications in Private Federated Learning".
    arxiv:1812.00984

    This is the optimal algorithm for mean estimation
    under local differential privacy (Asi et al. (2022), arXiv:2205.02466).

    The "constrain_sensitivity" operation considers three cases:

    1. If the vector has l2 norm 0, a new, random vector is drawn by
       generating each coordinate using a unit normal distribution and
       normalizing to a unit l2 norm.
    2. If the vector has a greater magnitude than the clipping_bound,
       the vector is scaled down as in standard norm clipping but to
       a unit l2 norm, not one equal to the clipping bound.
    3. Otherwise, the vector is normalized to have a unit
       l2 norm and is flipped with some probability so that the expectation
       is correct.

    Thus, the resulting clipped vector has l2 norm of 1 and is equal in
    expectation to the input vector (up to scaling).

    :param clipping_bound:
        The norm bound for clipping.
    """

    def __init__(self, clipping_bound: float, epsilon: float):
        self._clipping_bound = clipping_bound
        self._epsilon = epsilon

    def constrain_sensitivity(
            self,
            statistics: TrainingStatistics,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:
        """
        Turn the statistics into a unit vector, by clipping if it has too large
        a magnitude, and by normalising and probabilistically flipping
        otherwise.
        """
        _metadata, weights = statistics.get_weights()
        norm = get_ops().global_norm(weights, order=2)
        with NumpySeedScope(seed):
            if norm == 0:
                clipped_count = 0
                # Draw a random unit vector.
                random_vector = draw_random_unit_vector(
                    statistics.num_parameters)
                metadata, old_weights = statistics.get_weights()
                _, *reshape_context = get_ops().flatten(old_weights)
                random_weights = get_ops().reshape(random_vector,
                                                   *reshape_context)
                normalized_statistics = statistics.from_weights(
                    metadata, random_weights)
                clipped_count = 0
            else:
                normalisation_factor = 1 / norm
                if norm >= self._clipping_bound:
                    clipped_count = 1
                else:
                    # Maybe flip, but don't clip.
                    clipped_count = 0
                    flip_probability = 0.5 * (1 - norm / self._clipping_bound)
                    assert 0 <= flip_probability <= 0.5
                    if np.random.binomial(1, flip_probability) == 1:
                        # Flip the vector.
                        normalisation_factor *= -1
                    # Else don't flip: just normalise.

                normalized_statistics = statistics.apply_elementwise(
                    lambda v: v * normalisation_factor)

        metrics = Metrics([
            (name_formatting_fn('l2 norm bound'),
             Weighted.from_unweighted(self._clipping_bound)),
            (name_formatting_fn('fraction of clipped norms'),
             Weighted.from_unweighted(clipped_count)),
            (name_formatting_fn('norm before clipping'),
             Weighted.from_unweighted(norm)),
        ])
        return (normalized_statistics, metrics)

    def add_noise(
            self,
            statistics: TrainingStatistics,
            cohort_size: int,
            name_formatting_fn=lambda n: StringMetricName(n),
            seed: Optional[int] = None) -> Tuple[TrainingStatistics, Metrics]:

        assert cohort_size == 1

        metadata, weights = statistics.get_weights()
        unit_vector, *reshape_context = get_ops().flatten(weights)
        unit_vector_numpy = get_ops().to_numpy(unit_vector)
        num_dimensions = statistics.num_parameters
        with NumpySeedScope(seed):
            privatized_vector_numpy = priv_unit_functions.privatize(
                self._epsilon, unit_vector_numpy)

        scaling, _ = priv_unit_functions.compute_optimal_cap_offset(
            self._epsilon, num_dimensions)

        assumed_unit_signal_norm = 1.0
        assumed_signal_norm = assumed_unit_signal_norm * self._clipping_bound
        noise_squared_error = ((self._clipping_bound**2) * (scaling**2).value -
                               assumed_signal_norm**2)

        metrics = Metrics([(name_formatting_fn('DP squared error'),
                            Weighted.from_unweighted(noise_squared_error)),
                           (name_formatting_fn('signal-to-DP-noise ratio'),
                            SNRMetric(assumed_signal_norm,
                                      noise_squared_error))])

        # Re-scale by the clipping bound.
        scaled_vector = get_ops().to_tensor(self._clipping_bound *
                                            privatized_vector_numpy)
        scaled_weights = get_ops().reshape(scaled_vector, *reshape_context)
        statistics = statistics.from_weights(metadata, scaled_weights)

        return (statistics, metrics)

    # Implement SquaredErrorLocalPrivacyMechanism.
    def sensitivity_scaling(self, num_dimensions):
        return self._clipping_bound

    def sensitivity_squared_error(self, num_dimensions: int, l2_norm: float):
        magnitude = min(1, l2_norm / self._clipping_bound)

        # Sometimes the input vector is flipped.
        # We always return a vector of magnitude 1, so that the scatter is 1.
        # The mean is magnitude.
        return 1 - magnitude**2

    def add_noise_squared_error(self, num_dimensions: int, cohort_size: int):
        assert cohort_size == 1
        scaling, _ = priv_unit_functions.compute_optimal_cap_offset(
            self._epsilon, num_dimensions)
        # If the sensitivity is 1, then the norm of the resulting vector is
        # always "scaling".
        return ((scaling**2).value - 1) * (self._clipping_bound**2)
