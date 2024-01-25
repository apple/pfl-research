# Copyright © 2023-2024 Apple Inc.
import unittest

import numpy as np
import pytest

from pfl.common_types import Population
from pfl.metrics import Histogram, MetricName, Metrics, StringMetricName, Summed, TrainMetricName, Weighted, Zero


class TestMetrics(unittest.TestCase):
    """
    Note that the test uses plain numerical values, whereas `Weighted` will be
    used.
    """

    def setUp(self):
        self.zero = Zero
        self.empty_metrics = Metrics()

        self.name_1 = StringMetricName('1')
        self.name_2 = StringMetricName('2')
        self.name_3 = '3'
        self.name_4 = StringMetricName('4')

        self.metrics_1 = Metrics([(self.name_1, 1)])
        self.metrics_2 = Metrics.from_dict({self.name_1: 2, self.name_2: 4})
        self.metrics_3 = Metrics([(self.name_1, 8), (self.name_3, 16)])
        self.metrics_4 = Metrics([(self.name_3, 32), (self.name_4, 64)])
        self.metrics_5 = Metrics([(self.name_3, 128), (self.name_4, 256)])

    def test_len(self):
        self.assertEqual(len(self.empty_metrics), 0)
        self.assertEqual(len(self.metrics_1), 1)
        self.assertEqual(len(self.metrics_5), 2)

    def test_iter(self):
        self.assertEqual(dict(iter(self.metrics_2)), {
            self.name_1: 2,
            self.name_2: 4
        })

    def test_contains(self):
        self.assertTrue(self.name_1 in self.metrics_1)
        self.assertTrue(self.name_1 in self.metrics_3)
        self.assertFalse(self.name_1 in self.metrics_4)
        self.assertFalse(self.name_2 in self.empty_metrics)

    def test_getitem(self):
        self.assertEqual(self.metrics_3[self.name_3], 16)
        self.assertEqual(self.metrics_3[self.name_1], 8)
        self.assertEqual(self.metrics_3["1"], 8)

    def test_setitem(self):
        self.empty_metrics[self.name_2] = 7
        self.assertEqual(self.empty_metrics[self.name_2], 7)

    def test_add(self):
        # Normal addition.
        new_metrics = self.metrics_4 + self.metrics_5
        self.assertEqual(len(new_metrics), 2)
        self.assertEqual(new_metrics[self.name_3], 160)
        self.assertEqual(new_metrics[self.name_4], 320)

        # Addition with Zero.
        self.assertIs(self.zero + self.zero, self.zero)
        self.assertIs(self.metrics_4 + self.zero, self.metrics_4)
        self.assertIs(self.zero + self.metrics_5, self.metrics_5)

    def test_or(self):
        new_metrics = self.metrics_2 | self.metrics_4
        self.assertEqual(len(new_metrics), 4)
        self.assertEqual(new_metrics[self.name_1], 2)
        self.assertEqual(new_metrics[self.name_2], 4)
        self.assertEqual(new_metrics[self.name_3], 32)
        self.assertEqual(new_metrics[self.name_4], 64)

        with self.assertRaises(ValueError):
            self.metrics_3 | self.metrics_4  # pylint: disable=pointless-statement

    def test_to_simple_dict(self):
        metrics = Metrics([(self.name_1, 2), (self.name_2, Weighted(2, 2))])

        dic = metrics.to_simple_dict()
        assert len(dic) == 2
        assert dic['1'] == 2
        assert dic['2'] == 1

    def test_serialization(self):
        metrics = Metrics([(self.name_1, Weighted(2, 1)), (self.name_2, 256)])

        # Serialization
        vectors = metrics.to_vectors()

        assert len(vectors) == 2
        # Could be in either (well-defined) order.
        if len(vectors[0]) == 1:
            np.testing.assert_array_equal(vectors[0], [256])
            np.testing.assert_array_equal(vectors[1], [2, 1])
            new_vectors = [np.asarray([4]), np.asarray([7, 8])]
        else:
            np.testing.assert_array_equal(vectors[0], [2, 1])
            np.testing.assert_array_equal(vectors[1], [256])
            new_vectors = [np.asarray([7, 8]), np.asarray([4])]

        # Deserialize.
        new_metrics = metrics.from_vectors(new_vectors)
        assert new_metrics[self.name_1] == Weighted(7, 8)
        assert new_metrics[self.name_2] == 4


class TestWeighted(unittest.TestCase):

    def setUp(self):
        self.zero_0 = Weighted(0, 0)
        self.zero_1 = Weighted.from_unweighted(0)
        self.one = Weighted(1., 1.)
        self.one_2 = Weighted(1., 2.)
        self.one_3 = Weighted.from_unweighted(.5, 2.)

        self.three_half = Weighted(3.5, 2.)

        self.examples = [
            self.zero_0, self.zero_1, self.one, self.one_2, self.one_3,
            self.three_half
        ]

    def test_construction(self):
        self.assertEqual(self.zero_0.overall_value, 0)
        self.assertEqual(self.zero_0.weighted_value, 0)
        self.assertEqual(self.zero_0.weight, 0)

        self.assertEqual(self.zero_1.overall_value, 0)
        self.assertEqual(self.zero_1.weighted_value, 0)
        self.assertEqual(self.zero_1.weight, 1)

        self.assertEqual(self.one.overall_value, 1)
        self.assertEqual(self.one.weighted_value, 1)
        self.assertEqual(self.one.weight, 1)

        self.assertEqual(self.one_2.overall_value, .5)
        self.assertEqual(self.one_2.weighted_value, 1)
        self.assertEqual(self.one_2.weight, 2.)

        self.assertEqual(self.one_3.overall_value, .5)
        self.assertEqual(self.one_3.weighted_value, 1.)
        self.assertEqual(self.one_3.weight, 2.)

        self.assertEqual(self.three_half.overall_value, 1.75)
        self.assertEqual(self.three_half.weighted_value, 3.5)
        self.assertEqual(self.three_half.weight, 2.)

    def test_zero_weight_exception(self):
        with pytest.raises(Exception):
            _ = Metrics([("name_1", 2), ("name_2", Weighted(2, 0))])
        with pytest.raises(Exception):
            _ = Metrics([("name_3", Weighted(3, 0))])

    def test_equality(self):
        # Self-equality
        self.assertEqual(self.zero_0, self.zero_0)
        self.assertEqual(self.zero_1, self.zero_1)
        self.assertEqual(self.one, self.one)
        self.assertEqual(self.one_2, self.one_2)
        self.assertEqual(self.one_3, self.one_3)
        self.assertEqual(self.three_half, self.three_half)

        self.assertNotEqual(self.zero_0, self.zero_1)
        self.assertNotEqual(self.zero_0, self.one)
        self.assertNotEqual(self.one, self.one_2)
        self.assertEqual(self.one_2, self.one_3)

    def test_add(self):
        # Spot
        self.assertEqual(self.one + self.one_2, Weighted(2, 3))

        # Exhaustive.
        for left in self.examples:
            for right in self.examples:
                result = left
                result += right
                result_2 = left + right
                self.assertEqual(result.weighted_value,
                                 left.weighted_value + right.weighted_value)
                self.assertEqual(result.weight, left.weight + right.weight)
                self.assertEqual(result_2.weighted_value,
                                 left.weighted_value + right.weighted_value)
                self.assertEqual(result_2.weight, left.weight + right.weight)

    def test_to_from_vector(self):
        weighted = Weighted(0, 0).from_vector(np.array([3, 2]))
        assert weighted.overall_value == 1.5
        np.testing.assert_array_equal(weighted.to_vector(), np.array([3, 2]))

    def test_add_morphism(self):
        """
        Test that adding up serialized vectors is equivalent to adding the
        original values.
        This should be true for all MetricValue's.
        """
        for left in self.examples:
            for right in self.examples:
                vector = (left.to_vector() + right.to_vector())
                total = left.from_vector(vector)
                assert total == (left + right)


class TestSummed(unittest.TestCase):

    def test_construction(self):
        v = Summed(1)
        assert v.overall_value == 1

    def test_equality(self):
        assert Summed(1) == Summed(1)
        assert Summed(1) != Summed(2)

    def test_add(self):
        assert (Summed(1) + Summed(2)).overall_value == 3

    def test_to_from_vector(self):
        summed = Summed(1).from_vector(np.array([2]))
        assert summed.overall_value == 2
        np.testing.assert_array_equal(summed.to_vector(), np.array([2]))

    def test_add_morphism(self):
        left = Summed(1)
        right = Summed(2)
        vector = left.to_vector() + right.to_vector()
        assert left.from_vector(vector) == (left + right)


class TestHistogram:

    @pytest.fixture
    def bins(self):
        return [-1, 0, 2, 3]

    @pytest.fixture
    def counts(self):
        return [0, 1, 2]

    def test_construction(self, counts, bins):
        h1 = Histogram(counts, bins)
        np.testing.assert_array_equal(h1.bin_counts, counts)
        np.testing.assert_array_equal(h1.bins, bins)

        h2 = Histogram.from_values_bins([1, 2.2, 3, 4], bins)
        np.testing.assert_array_equal(h2.bin_counts, counts)
        np.testing.assert_array_equal(h2.bins, bins)

        h3 = Histogram.from_values_range([1, 2.2, 3, 4],
                                         num_bins=3,
                                         min_bound=-1,
                                         max_bound=3)
        np.testing.assert_array_equal(h3.bin_counts, counts)
        np.testing.assert_array_almost_equal(h3.bins, [-1, 1 / 3, 5 / 3, 3])

    def test_equality(self, counts, bins):
        h1 = Histogram(counts, bins)
        h2 = Histogram.from_values_bins([1, 2.2, 3, 4], bins)
        assert h1 == h2

    def test_add(self, counts, bins):
        h1 = Histogram(counts, bins)
        h2 = Histogram([1, 2, 4], bins)
        np.testing.assert_array_equal((h1 + h2).bin_counts, [1, 3, 6])
        np.testing.assert_array_equal((h1 + h2).bins, bins)

    def test_to_from_vector(self, counts, bins):
        h = Histogram([0, 0, 0], bins).from_vector(counts)
        np.testing.assert_array_equal(h.bin_counts, counts)
        np.testing.assert_array_equal(h.bins, bins)
        np.testing.assert_array_equal(h.to_vector(), counts)

    def test_add_morphism(self, bins):
        left = Histogram([1, 2, 3], bins)
        right = Histogram([3, 2, 1], bins)
        vector = left.to_vector() + right.to_vector()
        assert left.from_vector(vector) == (left + right)
        np.testing.assert_array_equal(vector, [4., 4., 4.])


class TestMetricNames(unittest.TestCase):

    # pytype: disable=wrong-arg-count,wrong-keyword-args
    def test_metric_name(self):
        metric_name = MetricName(description='loss',
                                 population=Population.TRAIN)
        self.assertEqual(str(metric_name), 'train population | loss')
        metric_name = MetricName(description='loss', population=Population.VAL)
        self.assertEqual(str(metric_name), 'val population | loss')
        metric_name = MetricName(description='loss',
                                 population=Population.TEST)
        self.assertEqual(str(metric_name), 'test population | loss')

    def test_train_metric_name(self):
        metric_name = TrainMetricName(description='loss',
                                      population=Population.TRAIN,
                                      after_training=True,
                                      local_partition='val')
        self.assertEqual(
            str(metric_name),
            'train population | val set | loss after local training')

        metric_name = TrainMetricName(description='loss',
                                      population=Population.VAL,
                                      after_training=True,
                                      local_partition='val')
        self.assertEqual(
            str(metric_name),
            'val population | val set | loss after local training')

        metric_name = TrainMetricName(description='loss',
                                      population=Population.TEST,
                                      after_training=False,
                                      local_partition='train')
        self.assertEqual(
            str(metric_name),
            'test population | train set | loss before local training')

        metric_name = TrainMetricName(description='loss',
                                      population=Population.TRAIN,
                                      after_training=True)
        self.assertEqual(str(metric_name),
                         'train population | loss after local training')

    # pytype: enable=wrong-arg-count,wrong-keyword-args


if __name__ == '__main__':
    unittest.main()
