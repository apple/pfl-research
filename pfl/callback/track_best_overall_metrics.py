# Copyright © 2023-2024 Apple Inc.
import logging
from typing import Dict, List, Optional, Tuple, Union

from pfl.callback.training_process import TrainingProcessCallback
from pfl.metrics import MetricNamePostfix, Metrics, StringMetricName, get_overall_value
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class TrackBestOverallMetrics(TrainingProcessCallback):
    """
    Track the best value of given metrics over all iterations.
    If the specified metric names are not found for a particular
    central iteration, nothing will happen. Use parameter
    ``assert_metrics_found_within_frequency`` to assert that they
    must eventually be found, e.g. if you are doing central evaluation
    only every nth iteration.

    :param lower_is_better_metric_names:
        A list of metric names to track. Whenever a metric with a name
        in this list is encountered, the lowest value of that metric
        seen through the history of all central iterations is returned.
    :param higher_is_better_metric_names:
        Same as ``lower_is_better_metric_names``, but for metrics where
        a higher value is better.
    :param assert_metrics_found_within_frequency:
        As a precaution, assert that all metrics referenced in
        ``lower_is_better_metric_names`` and
        ``higher_is_better_metric_names`` are found within this many
        iterations. If you e.g. misspelled a metric name or put this
        callback an order before the metric was generated, you will be
        notified.
    """

    def __init__(self,
                 lower_is_better_metric_names: Optional[List[Union[
                     str, StringMetricName]]] = None,
                 higher_is_better_metric_names: Optional[List[Union[
                     str, StringMetricName]]] = None,
                 assert_metrics_found_within_frequency: int = 25):
        self._lower_is_better_metric_names = lower_is_better_metric_names or []
        self._higher_is_better_metric_names = higher_is_better_metric_names or []
        self._assert_metrics_found_within_frequency = assert_metrics_found_within_frequency
        self._init()

    def _init(self):
        self._best_lower_metrics: Dict = {}
        self._best_higher_metrics: Dict = {}
        self._found_metric_at_iteration = None

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        self._init()
        return Metrics()

    def _get_name_with_postfix(self,
                               original_metric_name: Union[str,
                                                           StringMetricName]):
        if isinstance(original_metric_name, str):
            original_metric_name = StringMetricName(original_metric_name)
        return MetricNamePostfix(original_metric_name, 'best overall')

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:

        if self._found_metric_at_iteration is None:
            self._found_metric_at_iteration = dict.fromkeys(
                self._lower_is_better_metric_names +
                self._higher_is_better_metric_names, central_iteration)

        best_overall_metrics = Metrics()
        for (metric_names,
             cmp_op) in [(self._lower_is_better_metric_names, min),
                         (self._higher_is_better_metric_names, max)]:
            for k in metric_names:
                if k in aggregate_metrics:
                    self._found_metric_at_iteration[k] = central_iteration
                    new_value = get_overall_value(aggregate_metrics[k])
                    if k not in self._best_lower_metrics:
                        self._best_lower_metrics[k] = new_value
                    else:
                        self._best_lower_metrics[k] = cmp_op(
                            self._best_lower_metrics[k], new_value)
                    # This will report best overall metrics at same frequency
                    # as the underlying metric values are appearing.
                    best_overall_metrics[self._get_name_with_postfix(
                        k)] = self._best_lower_metrics[k]
                else:
                    if (central_iteration
                            > self._found_metric_at_iteration[k] +
                            self._assert_metrics_found_within_frequency):
                        iterations_past = (central_iteration -
                                           self._found_metric_at_iteration[k])
                        raise ValueError(
                            f"{k} has not been found in the past {iterations_past} "
                            "iterations, check the name of the metric and the "
                            "order of TrackBestOverallMetrics in callbacks.")
        return False, best_overall_metrics
