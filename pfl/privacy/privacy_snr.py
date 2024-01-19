# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Maintain and compute aggregate SNR metrics.
"""

import math

import numpy as np

import pfl.internal.logging_utils as logging_utils
from pfl.metrics import MetricValue


class SNRMetric(MetricValue):
    """
    A signal-to-noise metric for the Gaussian mechanism.

    The "signal" is defined as the L2 norm of all statistics after clipping
    but before adding the DP noise. Thus, the maximum value of the signal
    is equal to the norm clipping bound.
    The "noise" is defined as the L2 norm of the vector of standard deviations
    of noise added to each statistic. Since the noise added to each parameter
    has the same standard deviation `noise_stddev`, the overall noise is
    defined as `sqrt(num_dimensions) * noise_stddev`.

    All objects of type ``SNRMetric`` form a commutative monoid
    (with ``+`` as the operator).
    Intermediate values maintain two values.
    The first is the sum of L2 norms from each data vector.
    Note that the implicit assumption is that each data vector is in the same
    direction, which is usually an overestimate.
    The second is the sum of expected squared errors.
    Note that the expected standard deviation of the noise can be computed from
    this, but it does not sum.
    (This is the reason that local DP with the Gaussian mechanism can be useful
    at all.)

    :param signal_l2_norm:
        The L2 norm of the data vector.
    :param squared_error:
        The expected squared L2 error that the mechanism has added to the
        signal. This is summed over all elements of the vector, i.e.
        it is equal to num_dimensions * noise_stddev**2 where noise_stddev
        is the standard deviation of Gaussian noise added to each
        statistic.
    """

    def __init__(self, signal_l2_norm: float, squared_error: float):
        self._signal_l2_norm = float(signal_l2_norm)
        self._squared_error = float(squared_error)

    def __eq__(self, other):
        return (self.signal_l2_norm == other.signal_l2_norm
                and self.squared_error == other.squared_error)

    @property
    def signal_l2_norm(self) -> float:
        """
        The (summed) L2 norm of the data vectors before adding noise.
        """
        return self._signal_l2_norm

    @property
    def squared_error(self) -> float:
        """
        The summed variance of the Gaussian noise that is added.
        """
        return self._squared_error

    def __repr__(self):
        return (f'SNR({self.signal_l2_norm}/'
                f'~sqrt({self.squared_error}))')

    def __add__(self, other: 'SNRMetric') -> 'SNRMetric':
        return SNRMetric(self.signal_l2_norm + other.signal_l2_norm,
                         self.squared_error + other.squared_error)

    @property
    def overall_value(self) -> float:
        # The expected magnitude of the error is not quite the same as the root
        # of the expected squared L2 error, but we are going to ignore this
        # difference.
        # The difference depends on the distribution of the noise.
        if self._squared_error == 0:
            if self._signal_l2_norm == 0:
                return float('nan')
            return math.inf

        rough_noise_norm = math.sqrt(self._squared_error)

        return self._signal_l2_norm / rough_noise_norm

    def to_vector(self) -> np.ndarray:
        """
        Get a vector representation of this metric value, with
        ``dtype=float32``.
        Summing two vectors in this space must be equivalent as summing the two
        original objects.

        This serializes only the signal norm and the noise variance.
        The dimensionality is assumed to match.
        """
        return np.asarray([self._signal_l2_norm, self._squared_error],
                          dtype=np.float32)

    def from_vector(self, vector: np.ndarray) -> 'SNRMetric':
        """
        Create a new metric value of this class from a vector representation.
        """
        # Only these two are encoded.
        signal_l2_norm, squared_error = vector
        return SNRMetric(signal_l2_norm, squared_error)


@logging_utils.encode.register(SNRMetric)
def _encode_snr_metric(arg: SNRMetric):
    return {
        'signal': arg.signal_l2_norm,
        'squared_error': arg.squared_error,
        'snr': arg.overall_value,
    }
