# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Test log_float_functions.
"""

import math

import pytest  # type: ignore
import scipy.special
import scipy.stats

from pfl.internal.distribution import log_float_functions


class TestLogFloatSimpleFunctions:

    def test_exp(self):
        for value in [-10, -3, -.5, 0., 1., 4.]:
            assert (log_float_functions.exp(value).value == pytest.approx(
                math.exp(value)))

    def test_incomplete_beta_function(self):
        for alpha in [.01, .1, .5, 1., 5.]:
            for beta in [.01, .1, .5, 1., 5.]:
                incomplete = log_float_functions.incomplete_beta_function(
                    alpha, beta, 1.)
                complete = log_float_functions.beta_function(alpha, beta)
                assert incomplete.value == pytest.approx(complete.value)

    def test_normal_cdf(self):
        for value in [-10, -3, -.5, 0., 1., 4.]:
            assert (
                log_float_functions.normal_cdf(value).value == pytest.approx(
                    scipy.stats.norm.cdf(value)))

    def test_erfc(self):
        for value in [-10, -3, -.5, 0., 1., 4.]:
            assert (log_float_functions.erfc(value).value == pytest.approx(
                scipy.special.erfc(value)))


class TestLogBinomialCoefficients:

    def test_binomial_coefficients(self):
        max_k = 100
        ks = range(max_k + 1)
        ns = [0, 1, 2, 6, .1, .5, .9, 1.1, 1.5, 3.5, 7.5, 15, 20.5, 37]
        for n in ns:
            # print(f'{n}')
            for (k, coefficient) in zip(
                    ks, log_float_functions.binomial_coefficients(n)):
                # print(f'  {k}')
                reference = scipy.special.binom(n, k)
                log_reference = math.log(abs(reference))

                assert coefficient.log_value == pytest.approx(log_reference)
                if reference >= 0:
                    assert coefficient.sign == +1
                else:
                    assert coefficient.sign == -1
                last_k = k

            # Check that the number of elements is correct.
            if n == round(n):
                # If n is an integer, then only n+1 elements should be yielded.
                assert last_k == n
            else:
                # If n is non-integer, then an infinite number of elements
                # should be yielded.
                assert last_k == max_k
