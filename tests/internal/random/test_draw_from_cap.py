# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Test draw_from_cap.py.
"""

import math

import numpy as np
import pytest
from scipy.special import betainc

from pfl.internal.random.draw_from_cap import (
    _draw_planar_angle,
    _preprocess_angle_dot_product,
    draw_unit_vector_from_cap,
)


def angle_cdf(angle, num_dimensions):
    """
    Equation (3.17).
    """
    if 0 <= angle <= math.pi / 2:
        return betainc((num_dimensions - 1) / 2, .5, math.sin(angle)**2) / 2
    else:
        return 1 - betainc((num_dimensions - 1) / 2, .5,
                           math.sin(angle)**2) / 2


class TestAngleCDF:
    """
    Sanity-check the helper function above.
    """

    def test_angle_cdf_2_dimensions(self):
        for relative_angle in range(11):
            angle = relative_angle * math.pi / 10
            assert (angle_cdf(angle, 2) == pytest.approx(relative_angle / 10))

    def test_angle_cdf_3_dimensions(self):

        def angle_pdf(angle, num_dimensions):
            # Perform numeric integration
            delta = 1e-6 if angle < 2 * math.pi else -1e-6
            return (angle_cdf(angle + delta, num_dimensions) -
                    angle_cdf(angle, num_dimensions)) / delta

        for relative_angle in range(11):
            angle = relative_angle * math.pi / 10
            # The PDF should be proportional to the circumference of the circle,
            # which is proportional with sin(angle)
            assert (angle_pdf(angle, 3) == pytest.approx(math.sin(angle) / 2,
                                                         abs=1e-4))

    def test_angle_cdf_points(self):
        for num_dimensions in [2, 3, 5, 10, 100, 1000, 10000]:
            assert angle_cdf(math.pi, num_dimensions) == 1
            assert angle_cdf(math.pi / 2, num_dimensions) == 0.5


class TestDrawFromCap:

    def check_histogram(self, bin_starts, expected_fractions, samples,
                        tolerance):
        num_samples = len(samples)

        sample_bins = [0 for _ in range(len(bin_starts))]

        for sample in samples:
            correct_bin = [
                bin_index for (bin_index, bin_start) in enumerate(bin_starts)
                if bin_start < sample
            ][-1]
            sample_bins[correct_bin] += 1

        for expected_fraction, count in zip(expected_fractions, sample_bins):
            assert (count / num_samples) == pytest.approx(expected_fraction,
                                                          abs=tolerance)

    @pytest.mark.parametrize('num_dimensions', [3, 10, 100, 1000000])
    @pytest.mark.parametrize(
        'max_angle', [0., 0.1, .25 * math.pi, .5 * math.pi, 2, 3, math.pi])
    def test_preprocess_angle_dot_product(self, num_dimensions, max_angle):
        """
        Test _preprocess_angle_dot_product, which should give equivalent values
        whether max_angle is passed in or max_dot_product (apart from numerical
        accuracy).
        """
        max_dot_product = math.cos(max_angle)

        max_angle_1, h_1 = _preprocess_angle_dot_product(
            num_dimensions, max_angle, None)
        assert max_angle_1 == max_angle

        max_angle_2, h_2 = _preprocess_angle_dot_product(
            num_dimensions, None, max_dot_product)
        assert max_angle_2 == pytest.approx(max_angle)
        assert h_2 == pytest.approx(h_1)

    def check_angle_distribution(self, max_angle, num_dimensions, angles):

        num_samples = len(angles)
        num_bins = int(math.ceil(num_samples**.25))

        angle_bin_starts = [
            index * max_angle / num_bins for index in range(num_bins + 1)
        ]

        cdf_normalization = angle_cdf(max_angle, num_dimensions)
        tolerance = math.sqrt(num_bins / num_samples)

        expected_fractions = [
            (angle_cdf(bin_end, num_dimensions) -
             angle_cdf(bin_start, num_dimensions)) / cdf_normalization
            for bin_start, bin_end in zip(angle_bin_starts,
                                          angle_bin_starts[1:])
        ]

        self.check_histogram(angle_bin_starts, expected_fractions, angles,
                             tolerance)

    @pytest.mark.parametrize('num_dimensions', [3, 10, 50])
    @pytest.mark.parametrize('max_angle', [.25 * math.pi, 2, 3])
    @pytest.mark.parametrize('num_samples', [10, 100, 1000])
    # These values are for running over lunch:
    # @pytest.mark.parametrize('num_samples',
    #                          [10, 100, 1000, 10000, 100000
    #                          ])
    def test_draw_planar_angle(self, num_dimensions, max_angle, num_samples):
        """
        Test "_draw_planar_angle" to see whether the distribution is correct.
        """
        np.random.seed(123)
        angles = [
            _draw_planar_angle(num_dimensions, max_angle=max_angle)
            for _ in range(num_samples)
        ]

        self.check_angle_distribution(max_angle, num_dimensions, angles)

    @pytest.mark.parametrize('max_angle', [.25 * math.pi, 2, 3])
    @pytest.mark.parametrize('num_dimensions', [3, 10, 50])
    @pytest.mark.parametrize('num_samples', [100, 1000])
    @pytest.mark.parametrize('pole_type',
                             ['uniform', 'random', 'one_hot', 'one_hot_last'])
    def test_draw_unit_vector_from_cap_angle(self, max_angle, num_dimensions,
                                             num_samples, pole_type):
        """
        Test "draw_unit_vector_from_cap_angle" to see whether the distribution
        over angles relative to the pole is correct.
        """
        np.random.seed(456)

        def generate_angles(max_angle, num_samples, pole):
            for _ in range(num_samples):
                vector = draw_unit_vector_from_cap(pole, max_angle=max_angle)
                assert np.linalg.norm(vector) == pytest.approx(1)
                angle = np.arccos(np.dot(vector, pole))
                yield angle

        if pole_type == 'uniform':
            pole = np.ones((num_dimensions, )) / math.sqrt(num_dimensions)
        elif pole_type == 'random':
            unnormalized = np.random.uniform(size=num_dimensions)
            pole = unnormalized / np.linalg.norm(unnormalized)
        elif pole_type == 'one_hot':
            pole = np.zeros(num_dimensions)
            pole[1] = 1
        elif pole_type == 'one_hot_last':
            pole = np.zeros(num_dimensions)
            pole[-1] = 1
        else:
            raise AssertionError()

        # Sanity
        assert np.linalg.norm(pole) == pytest.approx(1)

        angles = list(generate_angles(max_angle, num_samples, pole))

        self.check_angle_distribution(max_angle, num_dimensions, angles)

    @pytest.mark.parametrize('max_angle', [.25 * math.pi, 2, 3])
    @pytest.mark.parametrize('num_samples', [100, 1000])
    @pytest.mark.parametrize('directions', [(0, 1, 2), (1, 2, 0), (2, 1, 0)])
    def test_draw_unit_vector_from_cap_distribution(self, max_angle,
                                                    num_samples, directions):
        """
        Test "draw_unit_vector_from_cap_angle" to check the angle of the
        distribution around the pole.
        In three dimensions, this angle is only one dimension, so that we can
        check the histogram.
        """
        np.random.seed(234)

        def generate_two_dimensional_angles(max_angle, num_samples,
                                            directions):
            # Having a simple pole means that the angle on the plane is in the
            # two other directions.
            pole = np.asarray([0, 0, 0])
            pole[directions[0]] = 1
            for _ in range(num_samples):
                vector = draw_unit_vector_from_cap(pole, max_angle=max_angle)
                angle = np.arctan2(vector[directions[1]],
                                   vector[directions[2]])
                print(vector, pole, directions, vector[directions[1]],
                      vector[directions[2]])
                assert -math.pi <= angle <= math.pi
                yield angle

        angles = list(
            generate_two_dimensional_angles(max_angle, num_samples,
                                            directions))

        num_bins = int(math.ceil(num_samples**.25))

        angle_bin_starts = [(index / num_bins * 2 - 1) * math.pi
                            for index in range(num_bins + 1)]

        tolerance = 10 / math.sqrt(num_bins * num_samples)

        expected_fractions = [1 / num_bins for _ in range(num_bins)]

        self.check_histogram(angle_bin_starts, expected_fractions, angles,
                             tolerance)

    @pytest.mark.parametrize('num_dimensions', [3, 10])
    @pytest.mark.parametrize('max_dot_product',
                             [1, .5, 0.01, 0., -.1, -.5, -.8])
    @pytest.mark.parametrize('num_samples', [100])
    def test_draw_unit_vector_from_cap_angle_max_dot_product(
            self, num_dimensions, max_dot_product, num_samples):
        """
        Test "draw_unit_vector_from_cap" with the max_dot_product
        parameterization.
        """
        np.random.seed(789)

        def generate_dot_products(max_dot_product, num_samples, pole):
            for _ in range(num_samples):
                vector = draw_unit_vector_from_cap(
                    pole, max_dot_product=max_dot_product)
                assert np.linalg.norm(vector) == pytest.approx(1)
                yield np.dot(vector, pole)

        pole = np.ones((num_dimensions, )) / math.sqrt(num_dimensions)

        min_dot_product = min(
            generate_dot_products(max_dot_product, num_samples, pole))

        assert max_dot_product - 1e-5 <= min_dot_product
        tolerance = 1e-5 + 5 * (1 - max_dot_product) / math.sqrt(num_samples)
        assert max_dot_product == pytest.approx(min_dot_product, abs=tolerance)
