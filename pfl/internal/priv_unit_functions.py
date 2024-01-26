# Copyright © 2023-2024 Apple Inc.
"""
Implement the PrivUnit2 privacy mechanism.

Algorithm1 from
Bhowmick et al. (2018) "Protection Against Reconstruction and
Its Applications in Private Federated Learning".
arxiv:1812.00984

It takes a unit vector and adds (optimal) noise for local differential privacy.
It does so by sampling from the unit sphere, choosing the "cap" around the
original vector with higher probability than the complement.
After drawing from the unit sphere, the vector is scaled up so that the
expected value is correct.
"""

import functools
import math
from typing import Tuple

import numpy as np
import scipy.special

from pfl.internal.distribution import LogFloat
from pfl.internal.distribution.log_float_functions import beta_function, incomplete_beta_function
from pfl.internal.random.draw_from_cap import draw_unit_vector_from_cap


@functools.lru_cache(maxsize=1000)
def _compute_cap_offset_epsilon(cap_offset: float, num_dimensions: int):
    """
    Compute the privacy budget (ε in Theorem 1) for drawing a vector
    from a cap with cap_offset.
    This is not the overall privacy parameter of PrivUnit2 as this doesn't
    add upp the privacy budget for choosing the cap (ε_0 in Theorem 1)

    :param cap_offset:
        Where the cap starts, with 0 being the equator and 1 the pole.
        (γ in the paper).
    :param num_dimensions:
        The number of dimensions of the vector to be privatized.
        d in the paper.
    """
    if cap_offset < 0 or cap_offset >= 1:
        return math.inf

    # Equation (16b) in the paper
    if cap_offset >= math.sqrt(2 / num_dimensions):
        return (math.log(num_dimensions) / 2 + math.log(6) -
                (num_dimensions - 1) / 2 * math.log(1 - cap_offset**2) +
                math.log(cap_offset))

    # Equation (16a) in the paper
    cap_offset_term = cap_offset * math.sqrt(2 *
                                             (num_dimensions - 1) / math.pi)
    if cap_offset_term >= 1:
        return math.inf
    return math.log((1 + cap_offset_term) / (1 - cap_offset_term))


@functools.lru_cache(maxsize=1000)
def _compute_pole_probability(overall_epsilon: float, cap_offset: float,
                              num_dimensions: int):
    """
    Compute the probability of drawing from the cap in the correct direction.

    :param overall_epsilon:
        The overall privacy budget ε for PrivUnit2.
    :param cap_offset:
        Where the cap starts, with 0 being the equator and 1 the pole.
        (γ in the paper).
    :param num_dimensions:
        The number of dimensions of the vector to be privatized.
        (d in the paper).
    """
    cap_size_epsilon = _compute_cap_offset_epsilon(cap_offset, num_dimensions)

    pole_epsilon = overall_epsilon - cap_size_epsilon
    assert pole_epsilon > 0

    one = LogFloat.from_value(1)
    pole_probability = one / (one + LogFloat(+1, -pole_epsilon))
    return (cap_size_epsilon, pole_epsilon, pole_probability)


@functools.lru_cache(maxsize=1000)
def _compute_scaling(cap_offset: float, pole_probability: LogFloat,
                     num_dimensions: int) -> LogFloat:
    """
    Compute the scaling that should be applied in order to guarantee that
    the output of PrivUnit2 is unbiased (Equation (15) in Algorithm 1).

    :param cap_offset:
        Where the cap starts, with 0 being the equator and 1 the pole.
        (γ in the paper).
    :param pole_probability:
        The probability of drawing from the cap in the correct direction.
        (p in the paper).
    :param num_dimensions:
        The number of dimensions of the vector to be privatized.
        (d in the paper).
    """
    alpha = (num_dimensions - 1) / 2
    tau = (1 + cap_offset) / 2

    beta_tau = incomplete_beta_function(alpha, alpha, tau)
    # beta_tau_difference = beta_function(alpha, alpha) - beta_tau
    # More numerically accurate when betainc(...) is close to 0:
    beta_tau_difference = (
        beta_function(alpha, alpha) *
        (LogFloat.from_value(1) -
         LogFloat.from_value(scipy.special.betainc(alpha, alpha, tau))))

    scale_1 = LogFloat.from_value(2)**(num_dimensions - 2)
    divisor_2_new = ((pole_probability / beta_tau_difference) -
                     (LogFloat.from_value(1) - pole_probability) / beta_tau)
    divisor_3 = LogFloat.from_value(1 - cap_offset**2)**alpha
    scale_4 = LogFloat.from_value(num_dimensions - 1)

    # This evaluation order keeps the values closest to 0 for high epsilons,
    # and improves numerical accuracy.
    return ((scale_1 / divisor_2_new) / divisor_3) * scale_4


def _compute_variance(epsilon: float, cap_offset: float, num_dimensions: int):
    """
    Compute the variance of PrivUnit2 for a given cap_offset and dimension.

    :param epsilon:
        The ε parameter of local differential privacy.
    :param cap_offset:
        Where the cap starts, with 0 being the equator and 1 the pole.
        (γ in the paper).
    :param num_dimensions:
        The number of dimensions of the vector to be privatized.
        (d in the paper).
    """
    _, _, pole_probability = _compute_pole_probability(epsilon, cap_offset,
                                                       num_dimensions)
    scale = _compute_scaling(cap_offset, pole_probability, num_dimensions)
    return (scale**2).value - 1


def _privatize_manual(epsilon: float, cap_offset: float,
                      unit_vector: np.ndarray):
    """
    Add noise for differential privacy to a unit vector based
    on PrivUnit2 with a given cap_offset (γ).

    :param epsilon:
        The ε parameter of local differential privacy.
    :param cap_offset:
        Where the cap starts, with 0 being the equator and 1 the pole.
        (γ in the paper).
    :param unit_vector:
        The vector to privatize.
    """
    num_dimensions = unit_vector.size
    assert np.isclose(np.linalg.norm(unit_vector), 1, atol=1e-4)

    _, _, pole_probability = _compute_pole_probability(epsilon, cap_offset,
                                                       num_dimensions)

    # Compute scaling to make the output unbiased
    scaling = _compute_scaling(cap_offset, pole_probability, num_dimensions)

    if np.random.binomial(1, pole_probability.value):
        # Draw unit vector uniformly from the correct cap.
        unit_result = draw_unit_vector_from_cap(unit_vector,
                                                max_dot_product=cap_offset)
    else:
        # Draw a unit vector uniformly over the complement.
        unit_result = draw_unit_vector_from_cap(-unit_vector,
                                                max_dot_product=-cap_offset)

    return scaling.value * unit_result


def _minimize_convex(function, start, end, num_steps):
    for _ in range(num_steps):
        mid_1 = .66 * start + .34 * end
        mid_2 = .34 * start + .66 * end
        if function(mid_1) > function(mid_2):
            start = mid_1
        else:
            end = mid_2
    return .5 * start + .5 * end


@functools.lru_cache(maxsize=1000)
def compute_optimal_cap_offset(epsilon: float,
                               num_dimensions: int) -> Tuple[LogFloat, float]:
    """
    Find an optimal cap offset (γ in the paper) by bisecting
    in two regimes (based on Theorem 1 in the paper).

    :param epsilon:
        The ε parameter of local differential privacy.
    :param num_dimensions:
        The number of dimensions of the vector to be privatized
        (d in the paper).

    :return:
        A tuple of the scaling and the cap_offset that minimised the scaling.
    """

    def cap_offset_scaling(cap_offset: float) -> LogFloat:
        cap_size_epsilon = _compute_cap_offset_epsilon(
            cap_offset, num_dimensions=num_dimensions)

        if cap_size_epsilon > epsilon:
            return LogFloat(+1, math.inf)

        (_, _,
         pole_probability) = _compute_pole_probability(epsilon, cap_offset,
                                                       num_dimensions)

        return _compute_scaling(cap_offset, pole_probability, num_dimensions)

    middle = math.sqrt(2 / num_dimensions)

    # For many settings, the regimes are different between (0, middle) and
    # (middle, 1), but always convex in both ranges (see Theorem 1).
    cap_offset1 = _minimize_convex(cap_offset_scaling, 0., middle, 50)
    cap_offset2 = _minimize_convex(cap_offset_scaling, middle, 1., 50)

    return min([
        (cap_offset_scaling(cap_offset1), cap_offset1),
        (cap_offset_scaling(cap_offset2), cap_offset2),
    ])


def privatize(epsilon, unit_vector: np.ndarray) -> np.ndarray:
    """
    Add noise for differential privacy to a unit vector using PrivUnit2.
    This implements PrivUnit2 from Bhowmick et al. (2018) with
    the optimal parameters (p and gama in the paper).
    This is the optimal algorithm for mean estimation
    under local DP (Asi et al. (2022)).

    :param epsilon:
        The ε parameter of local differential privacy.
    :param unit_vector:
        The vector to privatize (which has to be a unit vector).
    """
    num_dimensions = unit_vector.size

    # Find the optimal cap_offset parameter (γ in the paper) by
    # applying binary-search over the possible values
    # and recording the best error
    _, optimal_cap_offset = compute_optimal_cap_offset(epsilon, num_dimensions)

    return _privatize_manual(epsilon, optimal_cap_offset, unit_vector)
