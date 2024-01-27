# Copyright © 2023-2024 Apple Inc.
import logging
from typing import Generic, Tuple

from pfl.context import CentralContext, UserContext
from pfl.metrics import Metrics
from pfl.stats import StatisticsType

logger = logging.getLogger(name=__name__)


class Postprocessor(Generic[StatisticsType]):
    """
    A postprocessor defines an interface for features to interact with
    statistics after local training and after central aggregation. Many
    different categories of features operate in this way, e.g. weighting,
    adaptive hyperparameter methods, sparsification, privacy mechanisms.
    """

    def postprocess_one_user(
            self, *, stats: StatisticsType,
            user_context: UserContext) -> Tuple[StatisticsType, Metrics]:
        """
        Do any postprocessing of client's statistics before it is communicated
        back to the server.

        :param stats:
            Statistics returned from the local training procedure of this user.
        :param user_context:
            Additional information about the current user.
        :return:
            A tuple `(transformed_stats, metrics)`, where `transformed_stats`
            is `stats` after it is processed by the postprocessor, and
            `metrics` is any new metrics to track. Default implementation
            does nothing.
        """
        return stats, Metrics()

    def postprocess_server(
            self, *, stats: StatisticsType, central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[StatisticsType, Metrics]:
        """
        Do any postprocessing of the aggregated statistics object after
        central aggregation.

        :param stats:
            The aggregated statistics.
        :param central_context:
            Information about aggregation and other useful server-side
            properties.
        :return:
            A tuple `(transformed_stats, metrics)`, where `transformed_stats`
            is `stats` after it is processed by the postprocessor, and
            `metrics` is any new metrics to track. Default implementation
            does nothing.
        """
        return stats, Metrics()

    def postprocess_server_live(
            self, *, stats: StatisticsType, central_context: CentralContext,
            aggregate_metrics: Metrics) -> Tuple[StatisticsType, Metrics]:
        """
        Just like `postprocess_server`, but for live training. Default
        implementation is to call `postprocess_server`.
        Only override this in certain circumstances when you want different
        behaviour for live training, e.g. central DP.
        """
        return self.postprocess_server(stats=stats,
                                       central_context=central_context,
                                       aggregate_metrics=aggregate_metrics)
