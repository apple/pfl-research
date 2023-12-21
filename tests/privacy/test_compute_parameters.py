# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
'''
Test compute_parameters.py.
'''

import itertools
import math

import pytest

from pfl.privacy.compute_parameters import AnalyticGM_robust


class TestAnalyticGM:

    def test_specific(self):
        # This example comes from an implementation entirely independent of the
        # paper.
        sigma = AnalyticGM_robust(2.25, 0.00040834087)
        assert sigma == pytest.approx(math.sqrt(2))

    def _parameter_settings(self):
        for (epsilon, delta, number_of_iterations) in itertools.product(
            [0.01, 0.2, 1, 2, 20], [1e-6, 2e-4, 4e-2, 0.5], [1, 5, 200]):
            for sensitivity in [1e-5, 4e-3, 0.1, 1.0, 2, 27, 34000]:
                yield (epsilon, delta, number_of_iterations, sensitivity)

    def test_various(self):
        # Note that this inputs floating-point values and integers.
        for (epsilon, delta, number_of_iterations,
             sensitivity) in self._parameter_settings():
            sigma = AnalyticGM_robust(epsilon, delta, number_of_iterations,
                                      sensitivity)
            expected_delta = pytest.gaussian_mechanism_minimum_delta(
                float(epsilon) / number_of_iterations, float(sigma),
                float(sensitivity)) * number_of_iterations
            assert delta == pytest.approx(expected_delta, rel=2e-3)
