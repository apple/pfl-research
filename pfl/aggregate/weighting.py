# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
"""
Weighting strategies are :class:`~pfl.postprocessor.base.Postprocessor`
that provide different ways of applying a weight to each
individual statistics contribution before aggregation. This is a way to
increase the magnitude of contributions that are considered important by
some metric and decrease the magnitude of less important contributions.

How to use this feature is a philosophical question. When using federated SGD
(1 local iteration), then it makes sense to weight by datapoints using
:class:`~pfl.aggregate.weighting.WeightByDatapoints`, since this sums the
gradient over the whole data set. However, when the number of local gradient
steps is much higher than ``1``, and the local models are overtrained, it
makes more sense to assign each locally overtrained model the same weight,
i.e. weighting by users using
:class:`~pfl.aggregate.weighting.WeightByUser`.

.. warning::

    Scaling user statistics with different weights will require you to adjust
    the clipping bound of your privacy mechanism accordingly.
"""
from abc import ABC, abstractmethod
from typing import Tuple

from pfl.context import UserContext
from pfl.metrics import Metrics
from pfl.postprocessor.base import Postprocessor
from pfl.stats import WeightedStatistics


class WeightingStrategy(ABC, Postprocessor[WeightedStatistics]):
    """
    Base class for weighting schemes.
    """

    @abstractmethod
    def postprocess_one_user(
            self, *, stats: WeightedStatistics,
            user_context: UserContext) -> Tuple[WeightedStatistics, Metrics]:
        """
        Re-weights ``stats`` in-place according to the weighting scheme of
        this class.
        """
        raise NotImplementedError


class WeightByUser(WeightingStrategy):
    """
    Re-weights ``statistics`` in-place by user.
    This is generally used with federated averaging if you want a
    simple unweighted average of local model updates (statistics).
    """

    def postprocess_one_user(
            self, *, stats: WeightedStatistics,
            user_context: UserContext) -> Tuple[WeightedStatistics, Metrics]:
        stats.reweight(1)
        return stats, Metrics()


class WeightByDatapoints(WeightingStrategy):
    """
    Re-weights ``statistics`` in-place by the number of data points of user.
    This is generally used with federated averaging if you want to weight the
    averaging by the number of data points used for training the model on each
    device.
    """

    def postprocess_one_user(
            self, *, stats: WeightedStatistics,
            user_context: UserContext) -> Tuple[WeightedStatistics, Metrics]:
        stats.reweight(user_context.num_datapoints)
        return stats, Metrics()
