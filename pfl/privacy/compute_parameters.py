# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Compute parameters for the Gaussian mechanism.
'''

import numpy as np

from pfl.internal.bisect import bisect_automatic_range
from pfl.internal.distribution.log_float_functions import exp, normal_cdf


def AnalyticGM_robust_impl(eps, delta):
    """
    Compute (optimally) the noise parameter (sigma) for the Gaussian mechanism,
    for a given epsilon and delta.

    This assumes the L2 sensitivity is 1.

    Implements Algorithm 1 from
    Balle and Wang (2018), "Improving the Gaussian Mechanism for Differential
    Privacy: Analytical Calibration and Optimal Denoising".
    arXiv:1805.06530

    :param eps: The ε parameter of approximate differential privacy.
    :param delta: The δ parameter of approximate differential privacy.
    """
    delta0 = normal_cdf(0) - exp(eps) * normal_cdf(-np.sqrt(2.0 * eps))
    if delta >= delta0:

        def B_plus(v):
            value = (normal_cdf(np.sqrt(eps * v)) -
                     exp(eps) * normal_cdf(-np.sqrt(eps * (v + 2.0))))
            return value.value - delta

        _, v_star = bisect_automatic_range(B_plus, 0.0, 1.)
        alpha = np.sqrt(1.0 + v_star / 2.0) - np.sqrt(v_star / 2.0)
    else:

        def B_minus(v):
            value = (normal_cdf(-np.sqrt(eps * v)) -
                     exp(eps) * normal_cdf(-np.sqrt(eps * (v + 2.0))))
            return value.value - delta

        u_star, _ = bisect_automatic_range(B_minus, 0.0, 1.)
        alpha = np.sqrt(1.0 + u_star / 2.0) + np.sqrt(u_star / 2.0)
    sigma = alpha / np.sqrt(2.0 * eps)
    return sigma


def AnalyticGM_robust(eps, delta, k=1.0, l2=1):
    """
    Compute (optimally) the noise parameter (sigma) for the Gaussian mechanism,
    for a given epsilon and delta.

    :param eps: The ε parameter of approximate differential privacy.
    :param delta: The δ parameter of approximate differential privacy.
    :param k:
        The number of repetitions.
        Note that it might be advantageous to use moments accountant instead
        of this when k > 1.
    :param l2:
        The L2 sensitivity.
    """
    return AnalyticGM_robust_impl(eps / k, delta / k) * l2
