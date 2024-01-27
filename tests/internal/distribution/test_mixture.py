# Copyright © 2023-2024 Apple Inc.
"""
Test mixture.py
"""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from pfl.internal.distribution import Distribution, LogFloat, Mixture


@pytest.fixture
def components():

    def component_with_value(value):
        component = Mock(spec=Distribution)
        component.density.return_value = LogFloat.from_value(value)
        return component

    return [component_with_value(value) for value in [2, 4, 2]]


@pytest.fixture
def mixture(components):
    return Mixture([(2, components[0]), (3, components[1]),
                    (7, components[2])])


def test_construction():
    component_1 = Mock(spec=Distribution, point_shape=(7, 12))
    component_2 = Mock(spec=Distribution, point_shape=(7, 12))

    # Use generator instead of a list.
    def generate_components():
        yield (5, component_1)
        yield (10, component_2)
        yield (35, component_1)

    mixture = Mixture(generate_components())

    assert mixture.point_shape == (7, 12)
    assert len(mixture) == 3
    assert len(mixture.components) == 3
    expected_components = [(.1, component_1), (.2, component_2),
                           (.7, component_1)]

    # Test [].
    for index, expected_component in enumerate(expected_components):
        assert mixture[index] == expected_component

    def check_components(components):
        for ((weight, component),
             (expected_weight,
              expected_component)) in zip(components, expected_components):
            assert weight == pytest.approx(expected_weight)
            assert component is expected_component

    check_components(mixture.components)
    check_components([mixture[0], mixture[1], mixture[2]])
    # Test __iter__.
    check_components(list(mixture))


def test_responsibilities(components, mixture):
    r = mixture.responsibilities(7)

    total = 2 * 2 + 3 * 4 + 2 * 7
    assert len(r) == 3
    assert r[0].value == pytest.approx(2 * 2 / total)
    assert r[1].value == pytest.approx(3 * 4 / total)
    assert r[2].value == pytest.approx(2 * 7 / total)

    components[0].density.assert_called_with(7)
    components[1].density.assert_called_with(7)
    components[2].density.assert_called_with(7)


def test_density(components, mixture):
    density = mixture.density(3)

    expected_density = (2 * 2 + 3 * 4 + 2 * 7) / (2 + 3 + 7)

    assert density.value == pytest.approx(expected_density)

    components[0].density.assert_called_with(3)
    components[1].density.assert_called_with(3)
    components[2].density.assert_called_with(3)


@pytest.mark.parametrize('values, denominator', [
    ([(3, 7), (2, -2)], 5),
    ([(1, [2, 3]), (3, [5, 6])], 4),
])
@pytest.mark.parametrize('num_samples', [100, 1000, 10000])
def test_sample_fixed(values, denominator: float, num_samples: int):
    np.random.seed(27)

    def get_fixed_distribution(value):
        distribution = Mock(spec=Distribution)
        distribution.sample = lambda n: n * [value]
        return distribution

    mixture = Mixture([(weight, get_fixed_distribution(value))
                       for weight, value in values])
    data = mixture.sample(num_samples)

    points, counts = np.unique(data, axis=0, return_counts=True)

    for weight, value in values:
        matching_count = [
            count for (point, count) in zip(points, counts)
            if (point == value).all()
        ]
        assert len(matching_count) == 1
        count, = matching_count
        assert (count / num_samples == pytest.approx(weight / denominator,
                                                     abs=1 /
                                                     math.sqrt(num_samples)))
