# Copyright © 2023-2024 Apple Inc.
"""
Test distribution.py
"""

from math import log

import numpy as np
import pytest

from pfl.internal.distribution import Distribution, LogFloat


class SingleDimensionalDistribution(Distribution[float]):  # pylint: disable=unsubscriptable-object
    """
    Simple Distribution subclass for testing.
    """

    # Defined correctly for "float" by default.
    # def point_shape(self):

    def density(self, point: float) -> LogFloat:
        if point < 0:
            return LogFloat.from_value(0)
        elif point < 1:
            return LogFloat.from_value(.25)
        elif point < 2:
            return LogFloat.from_value(.5)
        elif point < 3:
            return LogFloat.from_value(.25)
        else:
            return LogFloat.from_value(0)

    # Defined correctly by default.
    # def log_densities(self, points: np.ndarray) -> np.ndarray:

    def sample(self, number: int) -> np.ndarray:
        raise AssertionError()


class TwoDimensionalDistribution(Distribution[np.ndarray]):  # pylint: disable=unsubscriptable-object
    """
    Simple two-dimensional Distribution subclass for testing.
    """

    # Defined correctly for "float" by default.
    @property
    def point_shape(self):
        return (2, )

    def density(self, point: np.ndarray) -> LogFloat:
        x, y = point
        if 0 < x < 1 and 0 < y < 1:
            return LogFloat.from_value(.5)
        elif 1 < x < 2 and 0 < y < 1:
            return LogFloat.from_value(.25)
        elif 0 < x < 1 and 1 < y < 2:
            return LogFloat.from_value(.25)

        return LogFloat.from_value(0)

    # Defined correctly by default.
    # def log_densities(self, points: np.ndarray) -> np.ndarray:

    def sample(self, number: int) -> np.ndarray:
        raise AssertionError()


def test_distribution_single():
    d = SingleDimensionalDistribution()
    assert d.point_shape == ()

    log_densities = d.log_densities(np.asarray([-.5, .2, 1.4, 2.8, 4.5]))

    assert log_densities == pytest.approx(
        np.asarray([
            -np.inf,  # log(0)
            log(.25),
            log(.5),
            log(.25),
            -np.inf,
        ]))


def test_distribution_two():
    d = TwoDimensionalDistribution()
    assert d.point_shape == (2, )

    assert d.density(np.asarray([.5, 1.5])) == LogFloat.from_value(.25)

    log_densities = d.log_densities(
        np.asarray([[-3, .5], [.5, +3], [.5, .5], [1.5, .2], [.6, 1.4]]))

    assert log_densities == pytest.approx(
        np.asarray([
            -np.inf,  # log(0)
            -np.inf,
            log(.5),
            log(.25),
            log(.25),
        ]))
