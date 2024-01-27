# Copyright © 2023-2024 Apple Inc.
"""
Draw a vector from a cap on a hypersphere.

The main implementation uses
Arun and Venkatapathi, "An O(n) algorithm for generating uniform random vectors
in n-dimensional cones".
https://arxiv.org/pdf/2101.00936.pdf

This also implements less optimal algorithms for reference.
"""

import math
from typing import Optional

import numpy as np

from pfl.internal.distribution.log_float import LogFloat


def draw_random_unit_vector(num_dimensions: int):
    """
    Draw a random unit vector, using numpy.random.
    Each coordinate is drawn from a unit Gaussian distribution and
    the vector is then normalized to have the L2 norm of 1.
    """
    random_vector = np.random.standard_normal(size=num_dimensions)
    return random_vector / np.linalg.norm(random_vector)


def _rotate_from_nth_canonical_basis(vector: np.ndarray,
                                     direction: np.ndarray) -> np.ndarray:
    """
    Rotate a vector from around [0, 0, ..., 0, 1] to around a different
    vector.
    This implements Arun and Venkatapathi (2021), Algorithm 5.2.

    :param vector:
        The vector to be rotated.
    :param direction:
        The vector that the vector should be rotated to.
    """
    assert vector.size == direction.size

    if direction[-1] == 1:
        # If direction == [0, 0, ..., 0, 1], then return the input.
        return vector

    factor = math.sqrt(1 - (direction[-1]**2))

    projection_1 = np.zeros_like(direction)
    projection = np.stack([projection_1, direction / factor], axis=1)
    projection[-1][0] = 1
    projection[-1][1] = 0

    g_minus_i = np.asarray([
        [direction[-1] - 1, -factor],
        [+factor, direction[-1] - 1],
    ])
    return vector + np.dot(np.dot(projection, g_minus_i),
                           np.dot(projection.transpose(), vector))


def _preprocess_angle_dot_product(num_dimensions: int,
                                  max_angle: Optional[float],
                                  max_dot_product: Optional[float]):
    """
    Compute h and θ_0 from Arun and Venkatapathi (2021), Algorithm 5.4.
    Whether max_angle is given or max_dot_product, the result must be
    equivalent, modulo numerical accuracy.
    """
    if max_dot_product is None:
        # This is the computation from the paper.
        assert max_angle is not None
        if max_angle == 0:
            return (0, math.inf)
        h = (num_dimensions - 2) * math.log(
            math.sin(min(max_angle, math.pi / 2)))
    else:
        # The following is the equivalent of the case where max_angle is given,
        # above, with good numeric accuracy, especially when max_dot_product
        # is close to 1.
        assert max_angle is None
        if max_dot_product == 1:
            return (0, math.inf)

        max_angle = np.arccos(max_dot_product)
        # Note that: math.sin(max_angle) = 1 - max_dot_product**2
        one = LogFloat.from_value(1)
        sin_term = (one - LogFloat.from_value(max_dot_product)**2
                    )**0.5 if max_dot_product > 0 else one
        h = (num_dimensions - 2) * sin_term.log_value

    return max_angle, h


def _draw_planar_angle(num_dimensions: int,
                       *,
                       max_angle: Optional[float] = None,
                       max_dot_product: Optional[float] = None) -> float:
    """
    Generate an angle with a distribution that can be used for drawing from a
    cap on the hypersphere.
    This uses rejection sampling in one dimension, and is therefore quite quick.
    This implements Arun and Venkatapathi (2021), Algorithm 5.4.
    The size of the cap can be parameterized with the maximum angle with the
    pole, or the minimum dot product with the pole.

    :param num_dimensions:
        The number of dimensions for the hypersphere.
        This affect the distribution of the angle.
    :param max_angle:
        The angle around the pole that the cap covers.
        This should be 0 <= max_angle <= π.
        Pass in either ``max_angle`` or ``max_dot_product``.
    :param max_dot_product:
        Where the cap starts, with 0 being the equator and 1 the pole.
        Pass in either ``max_angle`` or ``max_dot_product``.
    """
    max_angle, h = _preprocess_angle_dot_product(num_dimensions, max_angle,
                                                 max_dot_product)
    assert max_angle is not None
    if max_angle == 0:
        return 0
    while True:
        uniform_value = np.random.uniform()
        angle = np.random.uniform(0, max_angle)
        f = h + math.log(uniform_value)
        if f < (num_dimensions - 2) * math.log(math.sin(angle)):
            return angle


def draw_unit_vector_from_cap(
        pole: np.ndarray,
        *,
        max_angle: Optional[float] = None,
        max_dot_product: Optional[float] = None) -> np.ndarray:
    """
    Draw a random unit vector from a cap on the unit hypersphere.

    This implements Arun and Venkatapathi (2021), Algorithm 5.5.

    :param pole:
        The pole that the cap is around.
        This should be a unit vector.
    :param max_angle:
        The angle around the pole that the cap covers.
        This should be 0 <= max_angle <= π.
        Pass in either ``max_angle`` or ``max_dot_product``.
    :param max_dot_product:
        Where the cap starts, with 0 being the equator and 1 the pole.
        Pass in either ``max_angle`` or ``max_dot_product``.
    """

    num_dimensions = pole.size
    angle = _draw_planar_angle(num_dimensions,
                               max_angle=max_angle,
                               max_dot_product=max_dot_product)
    almost_all_dimensions = (math.sin(angle) *
                             draw_random_unit_vector(num_dimensions - 1))
    last_dimension = math.cos(angle)
    vector = np.concatenate([almost_all_dimensions, [last_dimension]])
    return _rotate_from_nth_canonical_basis(vector, pole)
