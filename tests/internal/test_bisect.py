# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Test bisect.py.
'''

import math

import pytest

from pfl.internal.bisect import bisect_automatic_range


# Example functions for bisection.
def log_is_10(x):
    if x == 0:
        return float('-inf')
    return math.log(x) - 10


def log_is_10_2(x):
    if x == 0:
        return float('-inf')
    return -math.log(x) + 10


def quadratic(x):
    return x**2 - 7 * x - 30


class TestGaussianParameterTools:

    @pytest.mark.parametrize('function, correct_result, start, step', [
        (log_is_10, math.exp(10), 0, 1),
        (log_is_10_2, math.exp(10), math.exp(7), .2),
        (quadratic, 10, -2, +1),
    ])
    def test_bisect(self, function, correct_result, start, step):
        exponent_min, exponent_max = (bisect_automatic_range(function,
                                                             start,
                                                             step,
                                                             num_steps=20))

        assert exponent_min <= correct_result <= exponent_max
        assert exponent_min == pytest.approx(correct_result, rel=1e-5)
        assert exponent_max == pytest.approx(correct_result, rel=1e-5)
