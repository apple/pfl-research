# Copyright © 2023-2024 Apple Inc.
import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from pfl.context import CentralContext
from pfl.internal import ops
from pfl.metrics import MetricName, Metrics, StringMetricName
from pfl.model.base import Model
from pfl.stats import StatisticsType, TrainingStatistics

get_num_params_weight_name = lambda: StringMetricName('number of parameters')
# pytype: disable=wrong-arg-count
get_num_datapoints_weight_name = lambda category: MetricName(
    'number of data points', category)
get_num_devices_weight_name = lambda category: MetricName(
    'number of devices', category)
get_total_weight_name = lambda category: MetricName('total weight', category)

# pytype: enable=wrong-arg-count


class Backend(ABC):
    """
    Backend base class.
    """

    def __init__(self):
        super().__init__()
        # The algorithm needs to keep a random state for any central random
        # operations.
        self._random_state = np.random.RandomState(
            np.random.randint(0, 2**32, dtype=np.uint32))

    def gather_results(
        self,
        model: Model,
        training_algorithm,
        *,
        central_context: CentralContext,
    ) -> Tuple[Optional[TrainingStatistics], Metrics]:
        """
        Evaluate or compute training statistics on devices and aggregate them.

        This call blocks.

        :param model:
            A model to use for training users.
            Subclasses can refine the type of ``Model`` they expect
        :param training_algorithm:
            An object that defines the training algorithm.
            Subclasses can have additional requirements on the type of this
            object.
        :param central_context:
            Settings to use for this round.
        :returns:
            A dictionary of the raw statistics (if this is a training round) and
            aggregated metrics.
        """
        return asyncio.run(
            self.async_gather_results(model,
                                      training_algorithm,
                                      central_context=central_context))

    @abstractmethod
    async def async_gather_results(
        self,
        model: Model,
        training_algorithm,
        *,
        central_context: CentralContext,
    ) -> Tuple[Optional[TrainingStatistics], Metrics]:
        """
        Should implement the map-reduce procedure of locally training on devices
        and aggregating the model updates.
        Can be either a simulation or using live infrastructure.
        If an implementation of ``async_gather_results`` requires runs of it to
        be non-overlapping in time, don't use ``await`` in it.

        Implementations of this may require that calls to it are started in a
        fixed order.

        For the parameters, see `gather_results`.
        """
        raise NotImplementedError


class Aggregator(ABC):
    """
    Base class for aggregating :class:`pfl.stats.TrainingStatistics` and
    :class:`pfl.metrics.Metrics`.
    """

    @abstractmethod
    def accumulate(self, *, accumulated: Optional[StatisticsType],
                   user_stats: StatisticsType) -> StatisticsType:
        """
        Accumulate user statistics on current worker process.

        :param accumulated:
            The state of the server-side statistics accumulator. If `None`,
            then ``user_stats`` is the first contribution in the
            aggregation.
        :param user_stats:
            Statistics from the user to accumulate. Each variable in the
            object will be accumulated separately, which means each accumulated
            user needs to have statistics of the same structure, e.g.
            the same number of variables of the same shapes.
        """

    def worker_reduce(
            self, *, aggregated_worker_stats: StatisticsType,
            central_context: CentralContext, aggregated_worker_metrics: Metrics
    ) -> Tuple[StatisticsType, Metrics]:
        """
        User statistics and metrics are first summed on each worker process,
        then they are reduced across workers using this method.
        This method can be ignored in concrete aggregator classes if they are
        not compatible with multi-process training.
        """
        raise NotImplementedError

    def worker_reduce_metrics_only(
            self, *, central_context: CentralContext,
            aggregated_worker_metrics: Metrics) -> Metrics:
        """
        This method is run instead of ``worker_reduce`` when the
        accumulated statistics is ``None``, what happens e.g. in a
        validation iteration.

        User metrics are first summed on each worker process, then they
        are reduced across workers using this method. This method
        can be ignored in concrete aggregator classes if they are not
        compatible with multi-process training.
        """
        raise NotImplementedError


class SumAggregator(Aggregator):
    """
    Aggregation of user statistics as a regular sum.
    Reduction across worker processes are done with an all-reduce sum.
    """

    def accumulate(self, *, accumulated: Optional[StatisticsType],
                   user_stats: StatisticsType) -> StatisticsType:
        if accumulated is None:
            return user_stats
        else:
            return user_stats + accumulated

    def worker_reduce_metrics_only(
            self, *, central_context: CentralContext,
            aggregated_worker_metrics: Metrics) -> Metrics:
        return ops.all_reduce_metrics(aggregated_worker_metrics)

    def worker_reduce(
            self, *, aggregated_worker_stats: StatisticsType,
            central_context: CentralContext, aggregated_worker_metrics: Metrics
    ) -> Tuple[StatisticsType, Metrics]:
        return ops.all_reduce_metrics_and_stats(aggregated_worker_stats,
                                                aggregated_worker_metrics)
