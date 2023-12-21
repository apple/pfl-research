# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Helpers for probabilistic distributions, particularly in the exponential
family.
"""

import functools
import operator
from abc import ABC, abstractmethod
from typing import Generic, Iterable, TypeVar

import numpy as np  # type: ignore

from .log_float import LogFloat

Element = TypeVar('Element')


def any_sum(elements: Iterable[Element]) -> Element:
    """
    Compute the sum of the elements in the iterable.
    This can be of any type, unlike standard Python ``sum``.
    """
    return functools.reduce(operator.add, elements)


def any_product(elements: Iterable[Element]) -> Element:
    """
    Compute the product of the elements in the iterable.
    This can be of any type.
    """
    return functools.reduce(operator.mul, elements)


Point = TypeVar('Point', float, np.ndarray)


class Distribution(ABC, Generic[Point]):
    """
    A base class representing a (probability) distribution.

    This is parameterised by ``Point``, which is the type of a single
    observation.
    This is either a float or a Numpy array.
    """

    @property
    def point_shape(self):
        """
        The shape of points.
        This is predefined for single-dimensional distributions as an empty
        tuple.
        Otherwise, subclasses should override this and specify the type of point
        they expect.
        """
        return ()

    @abstractmethod
    def density(self, point: Point) -> LogFloat:
        """
        :return:
            The density of the distribution at point ``point``.
        """

    def log_densities(self, points: np.ndarray) -> np.ndarray:
        """
        Compute the log-density at multiple points.
        The result is a Numpy array, which cannot contain ``LogFloat``.
        Subclasses may implement this in a faster way than calling ``density``
        many times.

        :return:
            Log-densities at many points.
        """
        return np.asarray([self.density(point).log_value for point in points])

    @abstractmethod
    def sample(self, number: int) -> np.ndarray:
        """
        Draw samples from the distribution.

        :param number:
            The number of samples to be drawn.
        """
        raise NotImplementedError()
