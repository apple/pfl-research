# Copyright © 2023-2024 Apple Inc.
import logging
import time
from typing import List, Tuple

from pfl.callback.base import TrainingProcessCallback
from pfl.metrics import Metrics, StringMetricName
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)


class StopwatchCallback(TrainingProcessCallback):
    """
    Records the wall-clock time for total time spent training, time
    per central iteration and overall average time per central iteration.

    :param decimal_points:
        Number of decimal points to round the wall-clock time metrics.
    :param measure_round_in_minutes:
        If ``True``, measure time for central iteration in minutes,
        not seconds. If you want this, it means your training is very slow!
    """

    def __init__(self,
                 decimal_points: int = 2,
                 measure_round_in_minutes: bool = False):
        self._decimal_points = decimal_points
        self._lap_start_time = time.time()
        self._start_time = self._lap_start_time
        self._laps: List = []
        self._round_postfix = 'min' if measure_round_in_minutes else 's'
        self._round_divider = 60 if measure_round_in_minutes else 1

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        """
        Starts the stopwatch.
        """
        self._lap_start_time = time.time()
        self._start_time = self._lap_start_time
        self._laps = []
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:

        returned_metrics = Metrics()

        current_time = time.time()
        self._laps.append(current_time - self._lap_start_time)
        returned_metrics[StringMetricName(
            'overall time elapsed (min)')] = round(
                (current_time - self._start_time) / 60, self._decimal_points)
        returned_metrics[StringMetricName(
            f'duration of iteration ({self._round_postfix})')] = round(
                self._laps[-1] / self._round_divider, self._decimal_points)
        average_duration_sec = sum(self._laps) / len(self._laps)
        returned_metrics[StringMetricName(
            f'overall average duration of iteration ({self._round_postfix})'
        )] = round(average_duration_sec / self._round_divider,
                   self._decimal_points)
        self._lap_start_time = current_time

        return False, returned_metrics
