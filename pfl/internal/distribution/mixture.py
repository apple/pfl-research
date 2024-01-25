# Copyright © 2023-2024 Apple Inc.
"""
A mixture of components.
"""

from typing import Iterable, Iterator, List, Tuple

import numpy as np

from .distribution import Distribution, any_sum
from .log_float import LogFloat


class Mixture(Distribution):  # pylint: disable=unsubscriptable-object
    """
    A mixture model, which has a density that is a weighted sum over
    "components" from another distribution.

    :param components:
        The components, as pairs of the weight and the component.
        Weights do not have to add up to 1 (they will be normalized
        to sum to 1).
    """

    def __init__(self, components: Iterable[Tuple[float, Distribution]]):
        components = list(components)
        assert components != []
        weights = [weight for weight, component in components]
        total_weight = sum(weights)
        self._components: List[Tuple[float, Distribution]] = [
            (weight / total_weight, component)
            for (weight, component) in components
        ]

    @property
    def point_shape(self):
        _, component = self._components[0]
        return component.point_shape

    @property
    def components(self) -> List[Tuple[float, Distribution]]:
        """
        :return:
            A list of (weight, component).
            The weights add up to 1.
        """
        return self._components

    def __str__(self):
        enumerated = ', '.join((f'({weight:.3}: {component})'
                                for weight, component in self._components))
        return '{' + enumerated + '}'

    def __getitem__(self, component_index: int) -> Tuple[float, Distribution]:
        """
        :return:
            The (weight, component) with index `component_index`.
        """
        return self._components[component_index]

    def __len__(self) -> int:
        """
        :return:
            The number of components in this mixture.
        """
        return len(self._components)

    def __iter__(self) -> Iterator[Tuple[float, Distribution]]:
        """
        :return:
            An iterator over (weight, component) .
        """
        return iter(self._components)

    def responsibilities(self, point) -> List[LogFloat]:
        """
        :return:
            The responsibilities for this point and each of the components.
            The responsibility is the posterior probability of the component
            having generated the point.
        """
        posteriors = [
            LogFloat.from_value(weight) * component.density(point)
            for (weight, component) in self._components
        ]
        likelihood = any_sum(posteriors)
        return [posterior / likelihood for posterior in posteriors]

    def density(self, point) -> LogFloat:
        return any_sum(
            LogFloat.from_value(weight) * component.density(point)
            for (weight, component) in self._components)

    def sample(self, number):
        component_probabilities = [
            weight for (weight, _component) in self._components
        ]
        component_numbers = np.random.multinomial(number,
                                                  component_probabilities)

        def samples_for_each_component():
            for number, (_, component) in zip(component_numbers,
                                              self._components):
                yield component.sample(number)

        samples = np.concatenate(list(samples_for_each_component()), axis=0)
        np.random.shuffle(samples)
        return samples
