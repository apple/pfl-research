# Copyright © 2023-2024 Apple Inc.
import logging
import typing
from typing import Any, Callable, List, Tuple, Union

from pfl.aggregate.base import get_num_datapoints_weight_name
from pfl.callback.training_process import TrainingProcessCallback
from pfl.common_types import Population
from pfl.metrics import Metrics, StringMetricName, get_overall_value
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class ConvergenceCallback(TrainingProcessCallback):
    """
    Track convergence using a performance measure and stop training when
    converged.

    Convergence is defined as when the performance becomes better than
    a threshold and afterwards stays that way for `patience`
    iterations. If the run is terminated, a new metric is added that
    stores the number of data points processed until the convergence
    was achieved (when the metric reached the threshold for the
    first time).

    :param metric_name:
        The name of the metric to track for convergence.
    :param patience:
        The run will be terminated when the metric `metric_name` is better
        than `performance threshold` for at least `patience` iterations.
    :param performance_threshold:
        The performance required to start considering whether training has
        converged.
    :param performance_is_better:
        A binary function that returns true if the first argument,
        indicating a performance level, is "better" than the second
        argument.
        For accuracy metrics, this is normally `operator.gt`, since higher
        is better.
        For loss or error metrics, lower is better, and this should be set to
        `operator.lt`.
    """

    def __init__(self, metric_name: Union[str, StringMetricName],
                 patience: int, performance_threshold: float,
                 performance_is_better: Callable[[Any, Any], bool]):
        self._metric_name = metric_name
        self._patience = patience
        self._performance_threshold = performance_threshold
        self._performance_is_better = performance_is_better
        self._convergence_history: List = []
        self._total_training_data = 0.
        self._num_datapoints_weight_name = get_num_datapoints_weight_name(
            Population.TRAIN)

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:

        assert aggregate_metrics[self._metric_name] is not None, (
            f"{self._metric_name} does not exist in aggregate_metrics: {aggregate_metrics}"
        )

        should_stop = False
        self._total_training_data += typing.cast(
            float, aggregate_metrics[self._num_datapoints_weight_name])

        if self._performance_is_better(
                get_overall_value(aggregate_metrics[self._metric_name]),
                self._performance_threshold):
            # Above threshold, start recording.
            self._convergence_history.append(
                (aggregate_metrics[self._metric_name],
                 self._total_training_data))
        else:
            # Not above threshold, reset history.
            self._convergence_history = []

        returned_metrics = Metrics()
        if len(self._convergence_history) >= self._patience:
            # Converged.
            should_stop = True
            # In hindsight, convergence started when the performance
            # threshold was crossed.
            _, first_total_training_data = self._convergence_history[0]
            returned_metrics[StringMetricName(
                'data points for convergence')] = first_total_training_data

        return should_stop, returned_metrics
