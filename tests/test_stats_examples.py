# Copyright © 2023-2024 Apple Inc.
"""
Test whether the base classes for TrainingStatistics are suit for purpose
using specific examples of extending the base classes.
"""

from collections import defaultdict

from numpy.testing import assert_almost_equal

from pfl.stats import TrainingStatistics, WeightedStatistics


class SetStatistics(TrainingStatistics):
    """
    Keep track of whether a specific element is seen in the data.
    This is like a mapping from element to boolean.

    This would be suitable for a simple nonparametric model.
    """

    def __init__(self, elements=None):
        if elements is None:
            self._elements = set()
        else:
            self._elements = set(elements)

    @property
    def elements(self):
        return self._elements

    @property
    def num_parameters(self) -> int:
        # This method should not have to be implemented.
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def from_weights(self, metadata, weights):
        raise NotImplementedError()

    def __add__(self, other):
        assert isinstance(other, SetStatistics)
        return SetStatistics(self.elements | other.elements)

    def __repr__(self):
        return repr(self.elements)


class NGramStatistics(WeightedStatistics):
    """
    Keep N-gram statistics.
    This is a mapping from tuples of elements to counts.
    The counts are pre-weighted by the weight.
    """

    def __init__(self, weight, counts):
        super().__init__(weight)
        self._counts = dict(counts)

    @property
    def counts(self):
        return self._counts

    @counts.setter
    def counts(self, new_counts) -> None:
        self._counts = new_counts

    @property
    def num_parameters(self) -> int:
        # This method should not have to be implemented.
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def from_weights(self, metadata, weights):
        raise NotImplementedError()

    def __add__(self, other):
        assert isinstance(other, NGramStatistics)
        new_counts = defaultdict(int)
        for sequence, num in self.counts.items():
            new_counts[sequence] += num
        for sequence, num in other.counts.items():
            new_counts[sequence] += num
        return NGramStatistics(self.weight + other.weight, new_counts)

    def __repr__(self):
        return repr(self.counts)

    def reweight(self, new_weight):
        reweight_factor = new_weight / self.weight
        new_counts = {
            sequence: count * reweight_factor
            for sequence, count in self.counts.items()
        }
        self.counts = new_counts
        self.weight = new_weight


class TestSetStatistics:

    def test_addition(self):
        stats1 = SetStatistics([1, 'a', 'c'])
        stats2 = SetStatistics([7, 'a', 'b'])
        assert (stats1 + stats2).elements == {1, 7, 'a', 'b', 'c'}


class TestNGramStatistics:

    def test_addition(self):
        counts1 = NGramStatistics(1, {
            ('this', ): 20,
            ('I', ): 10,
            ('this', 'is'): 5,
            ('I', 'am'): 1,
        })
        counts2 = NGramStatistics(
            2, {
                ('Broccoli', ): 17,
                ('this', 'is'): 3,
                ('Broccoli', 'is'): 5,
                ('Broccoli', 'is', 'great'): 4,
            })

        total = counts1 + counts2
        assert total.weight == 3

        total_counts = total.counts
        assert total_counts[('this', )] == 20
        assert total_counts[('I', )] == 10
        assert total_counts[('this', 'is')] == 8
        assert total_counts[('I', 'am')] == 1
        assert total_counts[('Broccoli', )] == 17
        assert total_counts[('Broccoli', 'is')] == 5
        assert total_counts[('Broccoli', 'is', 'great')] == 4

        total.reweight(1)
        total_counts = total.counts
        assert_almost_equal(total_counts[('this', )], 20 / 3)
        assert_almost_equal(total_counts[('I', )], 10 / 3)
        assert_almost_equal(total_counts[('this', 'is')], 8 / 3)
        assert_almost_equal(total_counts[('I', 'am')], 1 / 3)
        assert_almost_equal(total_counts[('Broccoli', )], 17 / 3)
        assert_almost_equal(total_counts[('Broccoli', 'is')], 5 / 3)
        assert_almost_equal(total_counts[('Broccoli', 'is', 'great')], 4 / 3)
