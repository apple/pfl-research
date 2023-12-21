# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Functions that work on or return LogFloat.
For some computations, this makes the results much simpler.

These are mostly simple wrappers for Numpy functions that exist already.
"""

import math
from typing import Iterable

import scipy.special
import scipy.stats

from .log_float import LogFloat


def exp(x: float) -> LogFloat:
    """
    The ``exp`` function that returns a ``LogFloat``.
    Since ``LogFloat`` holds the logarithm of a value, underlyingly this does
    nothing.
    However, it is clearer in code.
    """
    return LogFloat(+1, x)


def beta_function(alpha: float, beta: float):
    """
    Compute the beta function in log space.
    """
    log_result = scipy.special.betaln(alpha, beta)
    return LogFloat(+1, log_result)


def incomplete_beta_function(alpha: float, beta: float, x: float):
    """
    Compute the incomplete beta function in log space.
    """
    normalization = beta_function(alpha, beta)
    return normalization * LogFloat.from_value(
        scipy.special.betainc(alpha, beta, x))


def normal_cdf(x: float) -> LogFloat:
    """
    The CDF of a standard normal 𝒩(0,1).
    The result is returned as a LogFloat, so that it is particularly accurate
    in the left tail.
    """
    return LogFloat(+1, scipy.stats.norm.logcdf(x))


def erfc(value: float) -> LogFloat:
    """
    Evaluate the complementary error function, 1-erf(value).

    :return:
        1-erfc(value) as a ``LogFloat``, for numerical precision for
        ``value>0``.
    """
    # Scipy does not have a log of erfc directly, but it does have
    # stats.norm.logcdf (and stats.special.log_ndtr).
    return LogFloat.from_value(2) * normal_cdf(-value * math.sqrt(2))


def binomial_coefficients(exponent: float) -> Iterable[LogFloat]:
    """
    Yield the  binomial coefficients (``n`` choose ``k``) for fixed nonnegative
    ``n`` and ``k=0,1,2,3, ...``, as ``LogFloat``.

    If ``exponent`` is an integer, then the generator will stop after
    ``exponent+1`` elements, since the remaining elements would be 0.

    If ``exponent`` is a float, generalized binomial coefficients are produced,
    and the generator continues indefinitely.
    These coefficients are the coefficients one gets when writing out
    (1+x)**exponent.
    """
    assert exponent >= 0
    current = LogFloat.from_value(1)
    yield current
    numerator = exponent
    denominator = 1
    while True:
        if numerator == 0:
            return
        log_numerator = LogFloat.from_value(numerator)

        current = current * log_numerator / LogFloat.from_value(denominator)
        yield current
        numerator -= 1
        denominator += 1
