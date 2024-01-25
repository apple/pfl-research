# Copyright © 2023-2024 Apple Inc.
'''
Test privacy_snr.py.
'''

import math

import pytest
from numpy import isnan

from pfl.privacy.privacy_snr import SNRMetric


class TestSNRMetric:

    def test_zeros(self):
        assert isnan(SNRMetric(0., 0.).overall_value)
        assert SNRMetric(10., 0.).overall_value == math.inf

    # The following tests check the signal-to-noise ratio numerically.
    # Note that to compute the exactly correct expected standard deviation of
    # the noise, we need to use the Pochhammer symbol.
    # However, this can make our head hurt.
    # Instead, the following compares approximately to the square root of the
    # variance and the dimensionality.
    # This is close enough for higher dimensionalities.

    def test_single(self):
        # Intuitively the noise std val should be very close to sqrt(100) = 10.
        snr = SNRMetric(10., 100.)
        assert snr.overall_value == pytest.approx(1, rel=.01)

    def test_three(self):
        num_dimensions = 400
        snr = (SNRMetric(7., num_dimensions * 3.) +
               SNRMetric(4., num_dimensions * 5.) +
               SNRMetric(2., num_dimensions * 6.))
        signal_norm = 7 + 4 + 2
        rough_noise_norm = math.sqrt(num_dimensions * (3 + 5 + 6))
        rough_snr = signal_norm / rough_noise_norm
        assert snr.overall_value == pytest.approx(rough_snr, rel=0.01)

    def test_serialization(self):
        metric_1 = SNRMetric(3., 4.)
        metric_2 = SNRMetric(7., 8.)

        # Addition in the serialized space should have the same effect as
        # normal addition.
        summed_vector = metric_1.to_vector() + metric_2.to_vector()
        result = metric_1.from_vector(summed_vector)

        assert result.signal_l2_norm == (metric_1 + metric_2).signal_l2_norm
        assert result.squared_error == (metric_1 + metric_2).squared_error
        assert result.overall_value == (metric_1 + metric_2).overall_value
