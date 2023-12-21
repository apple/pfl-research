# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from unittest.mock import MagicMock

from pfl.aggregate.weighting import WeightByDatapoints, WeightByUser
from pfl.stats import WeightedStatistics


def test_weight_by_user(user_context):
    weighting_strategy = WeightByUser()
    mock_stats = MagicMock(spec=WeightedStatistics)
    out_stats, metrics = weighting_strategy.postprocess_one_user(
        stats=mock_stats, user_context=user_context)
    mock_stats.reweight.assert_called_once_with(1)
    assert out_stats is mock_stats
    assert len(metrics) == 0


def test_weight_by_datapoints(user_context):
    weighting_strategy = WeightByDatapoints()
    mock_stats = MagicMock(spec=WeightedStatistics)
    out_stats, metrics = weighting_strategy.postprocess_one_user(
        stats=mock_stats, user_context=user_context)
    mock_stats.reweight.assert_called_once_with(2)
    assert out_stats is mock_stats
    assert len(metrics) == 0
