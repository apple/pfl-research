# Copyright © 2023-2024 Apple Inc.
"""
Callbacks that evaluate the model against a centrally held dataset.
"""
import logging
from typing import Callable, Optional, Tuple

from pfl.callback.base import TrainingProcessCallback
from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import ModelHyperParams
from pfl.internal import ops
from pfl.metrics import MetricName, Metrics, StringMetricName
from pfl.model.base import EvaluatableModelType, ModelType, StatefulModel
from pfl.model.ema import CentralExponentialMovingAverage

logger = logging.getLogger(name=__name__)


class CentralEvaluationCallback(TrainingProcessCallback[EvaluatableModelType]):
    """
    Callback for performing evaluation on a centrally held dataset in between
    central iterations.
    The first evaluation is done before training begins.

    :param dataset:
        A ``Dataset`` that represents a central dataset.
        It has nothing to do with a user.
        The class ``Dataset`` is solely used to properly plug in to pfl.
    :param model_eval_params:
        The model parameters to use when evaluating the model. Can be ``None``
        if the model doesn't require hyperparameters for evaluation.
    :param frequency:
        Perform central evaluation every ``frequency`` central iterations.
    :param distribute_evaluation:
        Evaluate by distributing the computation across each worker used.
        If set to false, each worker runs evaluation independently. This will
        take longer to run than distributed evaluation. However, it may be
        necessary to disable distributed evaluation for some models and
        features, which do not support this mode.
    :param format_fn:
        A callable `(metric_name) -> MetricName` that formats the metric
        string name `metric_name` into a pfl metric name representation.
        The default value is

        .. code-block:: python

            lambda n: StringMetricName(f'Central val | {n}')

        It can be necessary to override the default when you are using multiple
        instances of this class, otherwise the metric names might conflict with
        each other.
    """

    def __init__(self,
                 dataset: AbstractDatasetType,
                 model_eval_params: Optional[ModelHyperParams] = None,
                 frequency: int = 1,
                 distribute_evaluation: bool = True,
                 format_fn: Optional[Callable[[str], MetricName]] = None):
        self._dataset = dataset
        self._model_eval_params = model_eval_params
        self._partial_dataset = dataset.get_worker_partition()
        self._frequency = frequency
        self._distribute_evaluation = distribute_evaluation
        if format_fn is None:
            self._format_fn = lambda n: StringMetricName(f'Central val | {n}')
        else:
            self._format_fn = format_fn

    def _eval(self, model):
        datapoints_metric_name = self._format_fn('number of data points')

        if (self._distribute_evaluation
                and model.allows_distributed_evaluation is not None):
            assert model.allows_distributed_evaluation, (
                'Your model does '
                'not support distributing the evaluation. This may be because '
                'you are using postprocessing functions for the users\' '
                'metrics or because the model itself does not support this '
                'mode. Disable distributed evaluation by setting '
                'distribute_evaluation=False in CentralEvaluationCallback.')

            # Do distributed multi-worker evaluation.
            metrics = model.evaluate(self._partial_dataset,
                                     name_formatting_fn=self._format_fn,
                                     eval_params=self._model_eval_params)
            metrics[datapoints_metric_name] = len(self._partial_dataset)
            metrics = ops.all_reduce_metrics(metrics)
        else:
            # Evaluation on all data on each worker.
            metrics = model.evaluate(self._dataset,
                                     name_formatting_fn=self._format_fn,
                                     eval_params=self._model_eval_params)
            metrics[datapoints_metric_name] = len(self._dataset)

        return (False, metrics)

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        # On first central iteration, evaluation should be done before training
        return self._eval(model)[1]

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: EvaluatableModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if (central_iteration == 0
                or central_iteration % self._frequency != 0):
            # No central evaluation this central iteration.
            return False, Metrics()

        return self._eval(model)


class CentralEvaluationWithEMACallback(CentralEvaluationCallback[StatefulModel]
                                       ):
    """
    Callback for performing evaluation with the exponential moving average of
    trained model on a centrally held dataset in between central iterations.
    The callback will update the EMA parameters after each central iteration,
    and will assign the EMA parameters to the model for evaluation.

    :param dataset:
        A ``Dataset`` that represents a central dataset.
        It has nothing to do with a user.
        The class ``Dataset`` is solely used to properly plug in to pfl.
    :param ema:
        A ``CentralExponentialMovingAverage`` that holds the EMA variables for
        the model to be evaluated.
        See :class:`~pfl.model.ema.CentralExponentialMovingAverage` for
        more details.
    :param model_eval_params:
        The model parameters to use when evaluating the model.
    :param frequency:
        Perform central evaluation every ``frequency`` central iterations.
    :param distribute_evaluation:
        Evaluate by distributing the computation across each worker used.
        If set to false, each worker runs evaluation independently. This will
        take longer to run than distributed evaluation. However, it may be
        necessary to disable distributed evaluation for some models and
        features, which do not support this mode.
    :param format_fn:
        A callable `(metric_name) -> MetricName` that formats the metric
        string name `metric_name` into a pfl metric name representation.
        The default value is

        .. code-block:: python

            lambda n: StringMetricName(f'Central val EMA | {n}')

        It can be necessary to override the default when you are using multiple
        instances of this class, otherwise the metric names might conflict with
        eachother.
    """

    def __init__(self,
                 dataset: AbstractDatasetType,
                 ema: CentralExponentialMovingAverage,
                 model_eval_params: Optional[ModelHyperParams] = None,
                 frequency: int = 1,
                 distribute_evaluation: bool = True,
                 format_fn: Optional[Callable[[str], MetricName]] = None):
        super().__init__(dataset, model_eval_params, frequency,
                         distribute_evaluation, format_fn)
        if format_fn is None:
            self._format_fn = lambda n: StringMetricName(
                f'Central val EMA | {n}')
        self._ema = ema
        self._model_eval_params = model_eval_params

    def _eval_ema(self, aggregate_metrics, model, central_iteration):
        # backup the current model variables
        model_state = model.get_parameters()
        # assign the EMA variables to the model for evaluation
        self._ema.assign()
        _, metrics = super().after_central_iteration(
            aggregate_metrics, model, central_iteration=central_iteration)
        # restore the model variables for next round of training
        model.set_parameters(model_state)
        return False, metrics

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        # On first central iteration, evaluation should be done before training
        return self._eval_ema(Metrics(), model, 0)[1]

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: StatefulModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        # perform EMA update
        self._ema.update()

        if (central_iteration == 0
                or central_iteration % self._frequency != 0):
            return False, Metrics()

        return self._eval_ema(aggregate_metrics, model, central_iteration)
