# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import numpy as np
import pytest

from pfl.stats import ElementWeightedMappedVectorStatistics, MappedVectorStatistics


@pytest.fixture(scope='function')
def stats_setup():
    weight = 2.
    weight2 = 3.
    data1 = np.arange(12).reshape(3, 4).astype(np.float32)
    data2 = np.ones((2, 1)).astype(np.float32)
    data1_2 = np.ones((3, 4)).astype(np.float32)
    data2_2 = np.arange(2).reshape(2, 1).astype(np.float32).astype(np.float32)
    stats = MappedVectorStatistics({'data1': data1, 'data2': data2}, weight)
    stats2 = MappedVectorStatistics({
        'data1': data1_2,
        'data2': data2_2
    }, weight2)
    return {
        'data1': data1.copy(),
        'data2': data2.copy(),
        'data1_2': data1_2.copy(),
        'data2_2': data2_2.copy(),
        'stats': stats,
        'stats2': stats2,
        'weight': weight,
        'weight2': weight2
    }


class TestVectorStatistics:

    def test_behaves_like_dict(self, stats_setup):
        stats = stats_setup['stats']
        assert 'data1' in stats
        assert stats['data1'].shape == (3, 4)
        assert len(stats) == 2
        assert len([v for k, v in stats.items()]) == 2
        assert set(stats.keys()) == {'data1', 'data2'}
        assert len(list(stats.values())) == 2

    def test_properties(self, stats_setup):
        stats = stats_setup['stats']
        assert stats.weight == 2
        stats.weight = 3
        assert stats.weight == 3
        assert stats.num_parameters == 14

    def test_default_weight(self, stats_setup):
        stats = MappedVectorStatistics({
            'data1': stats_setup['data1'],
            'data2': stats_setup['data2']
        })
        assert stats.weight == 1

    def test_add(self, stats_setup):
        stats = stats_setup['stats']
        stats_added = stats + stats_setup['stats2']
        for k in ['data1', 'data2']:
            expected = stats_setup[k] + stats_setup[k + "_2"]
            np.testing.assert_array_equal(stats_added[k], expected)
        assert stats_added.weight == stats_setup['weight'] + stats_setup[
            'weight2']

    def test_average(self, stats_setup):
        stats = stats_setup['stats']
        stats.average()
        for k in ['data1', 'data2']:
            np.testing.assert_array_equal(stats[k], stats_setup[k] / 2)
        assert stats.weight == 1

    def test_iter(self, stats_setup):
        for (k, v), expected_key in zip(stats_setup['stats'].items(),
                                        ['data1', 'data2']):
            assert k == expected_key
            np.testing.assert_array_equal(v, stats_setup[k])

    def test_apply(self, stats_setup):
        stats = stats_setup['stats']

        for transformed_stats in [
                stats.apply(lambda ws: [v * 2 for v in ws]),
                stats.apply_elementwise(lambda v: v * 2)
        ]:

            for k in ['data1', 'data2']:
                np.testing.assert_array_equal(transformed_stats[k],
                                              stats_setup[k] * 2)
            assert transformed_stats.weight == stats_setup['weight']

    def test_reweight(self, stats_setup):
        stats = stats_setup['stats']
        stats.reweight(1)
        for k in ['data1', 'data2']:
            np.testing.assert_array_equal(stats[k], stats_setup[k] / 2)
        assert stats.weight == 1

    def test_to_from_weights(self, stats_setup):
        stats = stats_setup['stats']
        metadata, vectors = stats.get_weights()
        np.testing.assert_array_equal(vectors[0], stats_setup['data1'])
        np.testing.assert_array_equal(vectors[1], stats_setup['data2'])

        new_metadata = metadata + 1
        new_vectors = [v + 1 for v in vectors]

        new_stats = stats.from_weights(new_metadata, new_vectors)
        for key, old_weight in stats.items():
            np.testing.assert_array_equal(new_stats[key], old_weight + 1)
        assert new_stats.weight == 3

    def test_from_vector(self, stats_setup):
        names = ["data1", "data2"]
        shapes = [stats_setup["data1"].shape, stats_setup["data2"].shape]
        # flatten data
        vector = np.concatenate(
            [stats_setup["data1"].flatten(), stats_setup["data2"].flatten()])
        stats = MappedVectorStatistics.from_vector(vector,
                                                   stats_setup["weight"],
                                                   names, shapes)

        np.testing.assert_array_equal(stats["data1"], stats_setup['data1'])
        np.testing.assert_array_equal(stats["data2"], stats_setup['data2'])
        assert stats.weight == stats_setup["weight"]


@pytest.fixture(scope='function')
def element_weighted_mapped_stats_setup():
    data1 = np.arange(12).reshape(3, 4).astype(np.float32)
    data2 = np.ones((2, 1)).astype(np.float32)
    data1_2 = np.ones((3, 4)).astype(np.float32)
    data2_2 = np.arange(2).reshape(2, 1).astype(np.float32).astype(np.float32)
    weights = {
        'data1': np.arange(1, 13).reshape(3, 4),
        'data2': np.ones((2, 1)) * 2.0
    }
    weights2 = {'data1': 2 * weights['data1'], 'data2': 2 * weights['data2']}
    stats = ElementWeightedMappedVectorStatistics(
        {
            'data1': data1,
            'data2': data2
        }, weights)
    stats2 = ElementWeightedMappedVectorStatistics(
        {
            'data1': data1_2,
            'data2': data2_2
        }, weights2)
    avg = {
        'data1': np.array([i / (i + 1) for i in range(12)]).reshape(3, 4),
        'data2': data2 / 2,
    }
    return {
        'data1': data1.copy(),
        'data2': data2.copy(),
        'data1_2': data1_2.copy(),
        'data2_2': data2_2.copy(),
        'stats': stats,
        'stats2': stats2,
        'weights': weights,
        'weights2': weights2,
        'avg': avg,
    }


class TestElementWeightedMappedVectorStatistics:

    def test_properties(self, element_weighted_mapped_stats_setup):
        stats = element_weighted_mapped_stats_setup['stats']
        np.testing.assert_array_equal(
            stats.weights["data1"],
            element_weighted_mapped_stats_setup['weights']['data1'])
        np.testing.assert_array_equal(
            stats.weights["data2"],
            element_weighted_mapped_stats_setup['weights']['data2'])
        stats.weights = {
            "data1": np.ones((3, 4)) * 3.0,
            "data2": np.ones((2, 1)) * 3.0
        }
        np.testing.assert_array_equal(stats.weights["data1"],
                                      np.ones((3, 4)) * 3.0)
        np.testing.assert_array_equal(stats.weights["data2"],
                                      np.ones((2, 1)) * 3.0)
        assert stats.num_parameters == 14

    def test_default_weights(self, element_weighted_mapped_stats_setup):
        stats = ElementWeightedMappedVectorStatistics({
            'data1':
            element_weighted_mapped_stats_setup['data1'],
            'data2':
            element_weighted_mapped_stats_setup['data2']
        })
        np.testing.assert_array_equal(stats.weights['data1'], np.ones((3, 4)))
        np.testing.assert_array_equal(stats.weights['data2'], np.ones((2, 1)))

    def test_add(self, element_weighted_mapped_stats_setup):
        stats = element_weighted_mapped_stats_setup['stats']
        stats2 = element_weighted_mapped_stats_setup['stats2']
        stats_added = stats + stats2
        for k in ['data1', 'data2']:
            expected = (element_weighted_mapped_stats_setup[k] +
                        element_weighted_mapped_stats_setup[k + "_2"])
            np.testing.assert_array_equal(stats_added[k], expected)
            np.testing.assert_array_equal(
                stats_added.weights[k],
                element_weighted_mapped_stats_setup["weights"][k] * 3)

    def test_average(self, element_weighted_mapped_stats_setup):
        stats = element_weighted_mapped_stats_setup['stats']
        stats.average()
        for k in ['data1', 'data2']:
            np.testing.assert_array_almost_equal(
                stats[k],
                element_weighted_mapped_stats_setup['avg'][k],
                decimal=6)
            np.testing.assert_array_equal(stats.weights[k],
                                          np.ones_like(stats.weights[k]))

    def test_apply(self, element_weighted_mapped_stats_setup):
        stats = element_weighted_mapped_stats_setup['stats']

        for transformed_stats in [
                stats.apply(lambda ws: [v * 2 for v in ws]),
                stats.apply_elementwise(lambda v: v * 2)
        ]:

            for k in ['data1', 'data2']:
                np.testing.assert_array_equal(
                    transformed_stats[k],
                    element_weighted_mapped_stats_setup[k] * 2)
                np.testing.assert_array_equal(
                    transformed_stats.weights[k],
                    element_weighted_mapped_stats_setup['weights'][k])

    def test_not_implemented(self, element_weighted_mapped_stats_setup):
        stats = element_weighted_mapped_stats_setup['stats']
        with pytest.raises(NotImplementedError):
            stats.reweight(1.0)
        with pytest.raises(NotImplementedError):
            stats.weight == 1.0
        with pytest.raises(NotImplementedError):
            _ = stats.weight

    def test_to_from_weights(self, element_weighted_mapped_stats_setup):
        stats = element_weighted_mapped_stats_setup['stats']
        metadata, vectors = stats.get_weights()
        np.testing.assert_array_equal(
            vectors[0], element_weighted_mapped_stats_setup['data1'])
        np.testing.assert_array_equal(
            vectors[1], element_weighted_mapped_stats_setup['data2'])

        new_metadata = metadata + 1
        new_vectors = [v + 1 for v in vectors]

        new_stats = stats.from_weights(new_metadata, new_vectors)
        for key, old_weight in stats.items():
            np.testing.assert_array_equal(new_stats[key], old_weight + 1)

        np.testing.assert_array_equal(
            new_stats.weights["data1"],
            element_weighted_mapped_stats_setup['weights']['data1'] + 1)
        np.testing.assert_array_equal(
            new_stats.weights["data2"],
            element_weighted_mapped_stats_setup['weights']['data2'] + 1)

    def test_from_vector(self, element_weighted_mapped_stats_setup):
        data1 = element_weighted_mapped_stats_setup["data1"]
        data2 = element_weighted_mapped_stats_setup["data2"]
        weight = element_weighted_mapped_stats_setup["weights"]
        names = ["data1", "data2"]
        shapes = [data1.shape, data2.shape]
        # flatten data and weight
        vector = np.concatenate([data1.flatten(), data2.flatten()])
        weight_vector = np.concatenate(
            [weight["data1"].flatten(), weight["data2"].flatten()])
        stats = ElementWeightedMappedVectorStatistics.from_vector(
            vector, weight_vector, names, shapes)

        np.testing.assert_array_equal(stats["data1"], data1)
        np.testing.assert_array_equal(stats["data2"], data2)
        np.testing.assert_array_equal(stats.weights["data1"], weight["data1"])
        np.testing.assert_array_equal(stats.weights["data2"], weight["data2"])
