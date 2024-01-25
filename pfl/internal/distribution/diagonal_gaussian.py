# Copyright © 2023-2024 Apple Inc.
"""
A trainable multivariate Gaussian distribution with a diagonal variance.
"""

import math
from typing import List, Union

import numpy as np

from .distribution import Distribution, any_product
from .log_float import LogFloat


class DiagonalGaussian(Distribution[np.ndarray]):  # pylint: disable=unsubscriptable-object
    """
    A multivariate Gaussian distribution with a diagonal variance.

    :param mean:
        The mean, as a Numpy scalar or vector.
    :param variance:
        The variance, as a Numpy scalar or vector.
        This must have the same shape as the mean.
    """

    def __init__(self, mean: Union[np.ndarray, List[float], float],
                 variance: Union[np.ndarray, List[float], float]):
        self._mean = np.asarray(mean)
        self._variance = np.asarray(variance)
        assert self._mean.shape == self._variance.shape
        point_shape = self._mean.shape
        assert len(self._mean.shape) == len(self._variance.shape)
        assert len(self._mean.shape) in [0, 1]
        assert len(self._variance.shape) in [0, 1]

        if point_shape == ():
            self._num_dimensions = 1
        else:
            (self._num_dimensions, ) = point_shape

        assert (self._variance > 0).all()

        # Compute the log-determinant and store it as a LogFloat.
        if point_shape == ():
            self._normalisation_constant = LogFloat.from_value(
                2 * math.pi * self._variance.item())**(-.5)
        else:
            determinant = any_product(
                LogFloat.from_value(element) for element in self._variance)

            self._normalisation_constant = (
                LogFloat.from_value(2 * math.pi)**self._num_dimensions *
                determinant)**(-.5)

    def __str__(self):
        return f'𝒩({self._mean}, diag({self._variance}))'  # noqa: RUF001

    @property
    def point_shape(self):
        return self._mean.shape

    @property
    def num_dimensions(self):
        return self._num_dimensions

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def density(self, point: Union[np.ndarray, List[float],
                                   float]) -> LogFloat:
        point = np.asarray(point)
        assert point.shape == self.point_shape
        deviation = point - self._mean
        return self._normalisation_constant * LogFloat(
            +1, -.5 * float(np.sum(np.square(deviation) / self._variance)))

    def sample(self, number):
        cholesky_diagonal = np.sqrt(self._variance)
        samples = np.random.normal(size=(number, self._num_dimensions))
        return self._mean + samples * cholesky_diagonal

    def split(self, offset=0.1):
        """
        Split up this Gaussian, changing the mean along the direction of the
        highest variance, and keeping the variance.
        Note that the sum of the densities of the two new Gaussians is not the
        same as density of this one: that is impossible.

        In the full-covariance case, this would require finding the first
        eigenvector, but since this Gaussian has a diagonal (co)variance, the
        highest entry of the variance vector is used.

        :param offset:
            The offset as a fraction of the standard deviation in the direction
            of maximum variance.
        """
        if self._variance.shape == ():
            offset_vector = np.asarray(offset * math.sqrt(self._variance))
        else:
            split_variance, split_index = max(
                (variance, index)
                for (index, variance) in enumerate(self._variance))
            offset_vector = np.zeros_like(self._mean)
            offset_vector[split_index] = offset * math.sqrt(split_variance)
        mean_1 = self._mean + offset_vector
        mean_2 = self._mean - offset_vector
        return (DiagonalGaussian(mean_1, self._variance),
                DiagonalGaussian(mean_2, self._variance))


def diagonal_standard_gaussian(num_dimensions=1) -> DiagonalGaussian:
    """
    Return a unit Gaussian, i.e. with mean 0 and variance 1.

    :param num_dimensions:
        The number of dimensions of the Gaussian.
    """
    return DiagonalGaussian([0] * num_dimensions, [1] * num_dimensions)
