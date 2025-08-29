# Copyright © 2023-2024 Apple Inc.
import logging
import operator
from typing import Any, Callable, Optional, Tuple, Union

from pfl.callback.base import TrainingProcessCallback
from pfl.metrics import Metrics, StringMetricName, get_overall_value
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)


class EarlyStoppingCallback(TrainingProcessCallback):
    """
    Implements early stopping as a callback to use in the training process.
    The criteria for this callback to stop training is if the metric, given
    by ``metric_name``, has not reached a new best value for ``patience``
    consecutive central iterations.
    An improvement is defined by ``performance_is_better``.

    :param metric_name:
        The name of the metric to track for early stopping, usually in the
        form of a ``pfl.metrics.MetricName``.
    :param patience:
        Number of central iterations to wait for an improvement in the
        tracked metric before interrupting the training process.
    :param performance_is_better:
        A binary function that returns true if the first argument,
        indicating a performance level, is "better" than the second
        argument.
        For accuracy metrics, this is normally `operator.gt`, since higher
        is better.
        For loss or error metrics, lower is better, and this should be set to
        `operator.lt`. It is set to `operator.lt` by default because you
        would normally perform early stopping on a loss or error metric.
    """

    def __init__(self,
                 metric_name: Union[str, StringMetricName],
                 patience: int,
                 performance_is_better: Callable[[Any, Any],
                                                 bool] = operator.lt):
        self._metric_name = metric_name
        self._patience = patience
        self._performance_is_better = performance_is_better
        self._iterations_since_last_best = 0
        self._last_best: Optional[float] = None

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:

        assert aggregate_metrics[self._metric_name] is not None, (
            f"{self._metric_name} does not exist in aggregate_metrics: {aggregate_metrics}"
        )

        should_stop = False

        current_performance = get_overall_value(
            aggregate_metrics[self._metric_name])
        if self._last_best is None or self._performance_is_better(
                current_performance, self._last_best):
            # New best, start over and update the last best.
            self._last_best = get_overall_value(
                aggregate_metrics[self._metric_name])
            self._iterations_since_last_best = 0
        else:
            # Not a new best, increase the counter since last best.
            self._iterations_since_last_best += 1

        if self._iterations_since_last_best >= self._patience:
            # Out of patience, signal to stop training.
            should_stop = True

        return should_stop, Metrics()
