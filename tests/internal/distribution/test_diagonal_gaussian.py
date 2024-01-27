# Copyright © 2023-2024 Apple Inc.
"""
Test diagonal_gaussian.py
"""

import math

import numpy as np
import pytest
import scipy.stats

from pfl.internal.distribution import DiagonalGaussian, LogFloat, diagonal_standard_gaussian


def test_properties():
    g = DiagonalGaussian(1., 2.)
    assert g.point_shape == ()
    assert g.num_dimensions == 1
    assert np.all(g.mean == np.array(1.))
    assert np.all(g.variance == np.array(2.))

    g = DiagonalGaussian([1.], [2.])
    assert g.point_shape == (1, )
    assert g.num_dimensions == 1
    assert np.all(g.mean == np.array([1.]))
    assert np.all(g.variance == np.array([2.]))

    g = DiagonalGaussian([1., 4.], [3., 2.])
    assert g.point_shape == (2, )
    assert g.num_dimensions == 2
    assert np.all(g.mean == np.array([1., 4.]))
    assert np.all(g.variance == np.array([3., 2.]))

    g = DiagonalGaussian(np.arange(3), np.ones(3))
    assert g.point_shape == (3, )
    assert g.num_dimensions == 3
    assert np.array_equal(g.mean, np.arange(3))
    assert np.array_equal(g.variance, np.ones([3]))


def test_failed_initialization():
    with pytest.raises(Exception):
        DiagonalGaussian([1.], [2., 3.])  # different dims
    with pytest.raises(Exception):
        DiagonalGaussian([1., 3.], [2.])  # different dims
    with pytest.raises(Exception):
        DiagonalGaussian(np.ones((3, 1)), np.ones((3, 1)))  # 2D not 1D
    with pytest.raises(Exception):
        DiagonalGaussian(np.ones((2, 2)), np.ones((2, 2)))  # 2D not 1D
    with pytest.raises(Exception):
        DiagonalGaussian([1., 4.], [0., 2.])  # variance <= 0
    with pytest.raises(Exception):
        DiagonalGaussian([1., 4.], [-1., 2.])  # variance <= 0


def test_normalization():
    """
    Test that the normalization constant is correct.
    Use numerical integration.
    This is only feasible for the one-dimensional case.
    """
    for gaussian, start, end in [
        (DiagonalGaussian(2., 4.), -10., +20.),
        (DiagonalGaussian([2.], [4.]), [-10.], [+20.]),
    ]:
        point_number = 1000
        step_size = (np.asarray(end) - np.asarray(start)) / point_number
        points = [start + index * step_size for index in range(point_number)]
        total = LogFloat.from_value(0.)
        for point in points:
            total += gaussian.density(point)
        integral = total.value * step_size
        assert integral == pytest.approx(1)


def test_density():
    """
    Test the density against the implementation from scipy.stats.
    """
    for mean, variance, points in [
        (2., 4., [2., -5., -1., 0., .5, 2., 3., 7.]),
        ([2.], [4.], [[2.], [-5.], [-1.], [0.], [.5], [2.], [3.], [7.]]),
        ([-1., +3.], [1., 6.], [
            [-1., +3.],
            [-2., 0.],
            [-2., 7.],
            [-1., 3.],
            [0., 7.],
        ]),
    ]:
        reference = scipy.stats.multivariate_normal(mean=mean, cov=variance)
        reference_pdf = reference.pdf

        gaussian = DiagonalGaussian(mean, variance)
        for point in points:
            assert gaussian.density(point).value == pytest.approx(
                reference_pdf(point))


@pytest.mark.parametrize('gaussian, split_dimension', [
    (DiagonalGaussian(3., 4.), 0),
    (DiagonalGaussian([4.], [5.]), 0),
    (DiagonalGaussian([1., 4.], [2., 5.]), 1),
    (DiagonalGaussian([-1., 3., 7.], [3., 4., 5.]), 2),
])
@pytest.mark.parametrize('offset', [0.001, 0.01, 0.1, .5, 1.])
def test_split(gaussian: DiagonalGaussian, split_dimension: int, offset):
    gaussian1, gaussian2 = gaussian.split(offset=offset)

    # Note for future reference: to generalise this to full-covariance
    # Gaussians, formulate them with an orthonormal transformation of
    # the above examples.
    if gaussian.point_shape == ():
        offset_vector = offset * np.sqrt(gaussian.variance)
    else:
        offset_vector = (offset * np.sqrt(gaussian.variance) *
                         np.asarray([(1 if i == split_dimension else 0)
                                     for i in range(gaussian.num_dimensions)]))

    expected_means = [
        gaussian.mean - offset_vector, gaussian.mean + offset_vector
    ]
    assert ((gaussian1.mean == pytest.approx(expected_means[0]))
            or (gaussian1.mean == pytest.approx(expected_means[1])))
    assert ((gaussian2.mean == pytest.approx(expected_means[0]))
            or (gaussian2.mean == pytest.approx(expected_means[1])))

    # Test that generalises to full-covariance Gaussians:

    # Since the new Gaussians are moved a bit, the density at the
    # old mean should be reduced.
    # By how much?
    # The variance should cancel out, and we're left with this
    # fraction:
    expected_density_fraction = math.exp(-.5 * (offset**2))

    mean_density = gaussian.density(gaussian.mean).value

    mean_density_1 = gaussian1.density(gaussian.mean).value
    mean_density_2 = gaussian2.density(gaussian.mean).value

    assert mean_density_1 == pytest.approx(mean_density_2)
    assert (mean_density_1 == pytest.approx(mean_density *
                                            expected_density_fraction))


@pytest.mark.parametrize("mean, variance", [
    (-2., 8.),
    ([+1.], [4.]),
    ([2., 4.], [2., 8.]),
])
@pytest.mark.parametrize('num_samples', [100, 1000, 10000])
def test_sample(mean, variance, num_samples):
    np.random.seed(27)

    gaussian = DiagonalGaussian(mean=mean, variance=variance)
    samples = gaussian.sample(num_samples)
    assert samples.shape[0] == num_samples

    sample_mean = np.mean(samples, axis=0)
    sample_variance = np.mean(np.square(samples),
                              axis=0) - np.square(sample_mean)

    if gaussian.point_shape == ():
        sample_mean = float(sample_mean)
        sample_variance = float(sample_variance)

    assert sample_mean == pytest.approx(mean, abs=5 / math.sqrt(num_samples))
    assert sample_variance == pytest.approx(variance,
                                            abs=30 / math.sqrt(num_samples))


def test_diagonal_standard_gaussian():
    unit = diagonal_standard_gaussian()
    assert np.array_equal(unit.mean, np.asarray([0]))
    assert np.array_equal(unit.variance, np.asarray([1]))

    unit3 = diagonal_standard_gaussian(3)
    assert np.array_equal(unit3.mean, np.asarray([0, 0, 0]))
    assert np.array_equal(unit3.variance, np.asarray([1, 1, 1]))
