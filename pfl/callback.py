# Copyright © 2023-2024 Apple Inc.
"""
A callback provides hooks into the training process. Different methods provides
hooks into different stages of the central training loop.
"""
import cProfile
import logging
import operator
import os
import re
import subprocess
import time
import typing
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Union

from pfl.aggregate.base import get_num_datapoints_weight_name
from pfl.common_types import Population, Saveable
from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam.base import ModelHyperParams
from pfl.internal import ops
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import MetricName, MetricNamePostfix, Metrics, StringMetricName, get_overall_value
from pfl.model.base import EvaluatableModelType, ModelType, StatefulModel
from pfl.model.ema import CentralExponentialMovingAverage

logger = logging.getLogger(name=__name__)

# pylint: disable=too-many-lines


class TrainingProcessCallback(Generic[ModelType]):
    """
    Base class for callbacks.
    """

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        """
        Called before the first central iteration.
        """
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Finalize any computations after each central iteration.

        :param aggregate_metrics:
            A :class:`~pfl.metrics.Metrics` object with aggregated metrics
            accumulated from local training on users and central updates
            of the model.
        :param model:
            A reference to the `Model` that is trained.
        :param central_iteration:
            The current central iteration number.
        :returns:
            A tuple.
            The first value returned is a boolean, signaling that training
            should be interrupted if ``True``.
            Can be useful for implementing features with early stopping or
            convergence criteria.
            The second value returned is new metrics.
            Do not include any of the aggregate_metrics!
        """
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        """
        Called at the end of training.
        """
        pass


class RestoreTrainingCallback(TrainingProcessCallback):
    """
    Add fault-tolerance to your training. If the training run fails
    and you restart it, this callback will restore all recent
    checkpoints of the ``saveables`` before starting training again.
    Be careful if you've implemented any stateful component,
    these will only be restored if you've properly implemented the
    :class:`~pfl.common_types.Saveable` interface on the component
    and input it to this callback. For restoring a checkpoint, it
    is assumed that all ``saveables`` were successfully stored
    in the last attempt.

    :param saveables:
        The objects that need to save their states so that they can be
        loaded if training is interrupted and then resumed.
    :param checkpoint_dir:
        Root dir for where to store the saveables' states.
        Let this be a list of directory paths to specify a unique
        checkpoint directory for each saveable.
        Location will be relative to root dir on current platform.
    :param checkpoint_frequency:
        Save checkpoints of ``saveables`` every this many iterations.
    """

    def __init__(self,
                 saveables: List[Saveable],
                 checkpoint_dir: Union[str, List[str]],
                 checkpoint_frequency: int = 1):
        self._saveables = saveables

        from pfl.internal.platform.selector import get_platform
        self._checkpoint_dirs: List[str]
        if isinstance(checkpoint_dir, list):
            assert len(saveables) == len(checkpoint_dir)
            self._checkpoint_dirs = get_platform(
            ).create_checkpoint_directories(checkpoint_dir)
        else:
            self._checkpoint_dirs = get_platform(
            ).create_checkpoint_directories([checkpoint_dir]) * len(saveables)
        self._checkpoint_frequency = checkpoint_frequency

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        """
        Restore from previous run's checkpoints if exists.
        """
        # Restore saveables.
        num_components_restored = 0
        for saveable, checkpoint_dir in zip(self._saveables,
                                            self._checkpoint_dirs):
            try:
                saveable.load(checkpoint_dir)
            except CheckpointNotFoundError as e:
                logger.info('RestoreTrainingRunCallback - %s for %s', e,
                            saveable)
            else:
                logger.info(
                    'RestoreTrainingRunCallback - Restored checkpoint for %s',
                    saveable)
                num_components_restored += 1
        return Metrics([(StringMetricName('restored components'),
                         num_components_restored)])

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if (central_iteration % self._checkpoint_frequency == 0
                and get_ops().distributed.local_rank == 0):
            for saveable, checkpoint_dir in zip(self._saveables,
                                                self._checkpoint_dirs):
                saveable.save(checkpoint_dir)
                logger.info(
                    'RestoreTrainingRunCallback - Saved checkpoint for %s',
                    saveable)
        return False, Metrics()


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


class TensorBoardCallback(TrainingProcessCallback):
    """
    Log events for TensorBoard: metrics, graph visualization, weight
    histograms.
    Launch tensorboard with command:

    .. code-block::

        tensorboard --logdir=<path to log_dir>

    .. note::

        Only supported with TF
        (:class:`pfl.model.tensorflow.TFModel`) right now.

    :param log_dir:
        Dir path where to store the TensorBoard log files.
        This path should be unique for every run if you run multiple
        trainings on the same machine.
    :param write_weights:
        Save weight histograms and distributions for the layers of the model
        There are 3 different modes:

        * ``False`` - disable this feature.
        * ``True`` - save histograms every time the algorithm performs an
          evaluation iteration (``evaluation_frequency`` in
          :class:`~pfl.hyperparam.base.ModelHyperParams`).
        * An integer - Perform every this many central iterations.

    :param write_graph:
        Visualize the model graph in TensorBoard. Disable this to keep the
        size of the TensorBoard data small.
    :param tensorboard_port:
        Port to use when hosting TensorBoard.
    """

    def __init__(self,
                 log_dir: str,
                 write_weights: Union[bool, int] = False,
                 write_graph: bool = True,
                 tensorboard_port: Optional[int] = None):
        import tensorflow as tf
        self.tf = tf
        self._log_dir = log_dir
        self._write_weights = write_weights
        self._should_write_train_graph = write_graph
        self._tensorboard_port = tensorboard_port

        self._writers = {
            p: tf.summary.create_file_writer(os.path.join(log_dir, p.value))
            for p in Population
        }
        self._population_prefix = r'({}) population \| '.format("|".join(
            [p.name.capitalize() for p in Population]))

    def _write_keras_model_train_graph(self, model):
        from pfl.model.tensorflow import TFModel
        assert isinstance(
            model,
            TFModel), ('Writing train graph only works for keras model.')
        # Write forward prop graph to TensorBoard.
        from tensorflow.python.ops import summary_ops_v2
        with self._writers[Population.TRAIN].as_default(
        ), self.tf.summary.record_if(True):
            ops = get_ops()
            try:
                for concrete_fn in ops.graph_cache.values():
                    summary_ops_v2.graph(concrete_fn.graph)
            except KeyError:
                logger.warning('Failed to save graphs to TensorBoard. '
                               f'Ops module is {ops}, needs to be TensorFlow')

    def _write_keras_model_weights(self, model, iteration):
        # Write histograms of the model weights to TensorBoard.
        from pfl.model.tensorflow import TFModel
        assert isinstance(model, TFModel), (
            'Writing weights histograms only works for keras model.')
        with self._writers[Population.TRAIN].as_default(
        ), self.tf.summary.record_if(True):
            for layer in model.keras_model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(":", "_")
                    # Add a suffix to prevent summary tag name collision.
                    self.tf.summary.histogram(weight_name + "/histogram",
                                              weight,
                                              step=iteration)
            self._writers[Population.TRAIN].flush()

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if (self._tensorboard_port is not None
                and get_ops().distributed.global_rank == 0):
            subprocess.Popen([
                'tensorboard', '--logdir', self._log_dir, '--port',
                str(self._tensorboard_port), '--bind_all'
            ],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             env=os.environ.copy())
        return Metrics()

    def _filter_ignored_metric_names(self, metrics):

        def should_ignore(name):
            try:
                if name.ignore_serialization:
                    return True
            except AttributeError:
                pass
            return False

        return [n for n, _ in metrics if not should_ignore(n)]

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        # Only save to TensorBoard with 1 process on worker 0
        if get_ops().distributed.global_rank != 0:
            return False, Metrics()

        if self._should_write_train_graph:
            # Can only write graph after model has been used.
            # Only do this once.
            self._write_keras_model_train_graph(model)
            self._should_write_train_graph = False

        # Writers in outer loop to minimize re-initialization.
        with self.tf.summary.record_if(True):
            for population, writer in self._writers.items():
                with writer.as_default():
                    for (str_name, value), name in zip(
                            aggregate_metrics.to_simple_dict().items(),
                            self._filter_ignored_metric_names(
                                aggregate_metrics)):
                        # Write metrics which has no particular population
                        # with the "train" writer.
                        no_population = (not hasattr(name, 'population')
                                         ) and population == Population.TRAIN
                        correct_population = hasattr(
                            name,
                            'population') and name.population == population
                        if correct_population:
                            # Remove population prefix.
                            str_name = re.sub(self._population_prefix, '',
                                              str_name)
                        if no_population or correct_population:
                            self.tf.summary.scalar(str_name,
                                                   value,
                                                   step=central_iteration)

        if (self._write_weights
                and central_iteration % self._write_weights == 0):
            self._write_keras_model_weights(model, central_iteration)
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        for writer in self._writers.values():
            writer.close()


class CheckpointPolicy(ABC):
    """
    Controls when `PolicyBasedModelCheckpointingCallback` should checkpoint.
    """

    @abstractmethod
    def should_checkpoint_now(self, aggregate_metrics: Metrics,
                              central_iteration: int) -> bool:
        """
        Invoked at the end of each central iteration to decide whether
        a checkpoint should be made.
        """
        raise NotImplementedError

    @abstractmethod
    def should_checkpoint_at_end(self) -> bool:
        """
        Invoked at the end of training to decide whether a checkpoint should
        be made.
        """
        raise NotImplementedError


class IterationFrequencyCheckpointPolicy:
    """
    Checkpoint policy for `PolicyBasedModelCheckpointingCallback` that
    saves a checkpoint after every `checkpoint_frequency` iterations if the
    value is positive or at the end of training if it is zero.
    """

    def __init__(self, checkpoint_frequency: int):
        self.checkpoint_frequency = checkpoint_frequency

    def should_checkpoint_now(self, aggregate_metrics: Metrics,
                              central_iteration: int) -> bool:
        """
        Return true when the number of `central_iteration`s that have
        completed is a non-zero multiple of `self.checkpoint_frequency`.
        """
        return (self.checkpoint_frequency > 0
                and central_iteration % self.checkpoint_frequency
                == (self.checkpoint_frequency - 1))

    def should_checkpoint_at_end(self) -> bool:
        return self.checkpoint_frequency == 0


class MetricImprovementCheckpointPolicy(CheckpointPolicy):
    """
    Stateful checkpoint policy for `PolicyBasedModelCheckpointingCallback`
    to save a checkpoint after any iteration where the value of `metric_name`
    has improved versus the prior best value.

    :param metric_name:
        The metrics whose value to track.

    :param threshold_value:
        If present, only save a checkpoint if the metric value is better than
        this value.

    :param performance_is_better:
        A binary predicate indicating that `lhs` is better `rhs`.

        For metrics where higher values are better, like precision,
        you would want to use `operator.gt`, and for metrics like
        loss, you would want to use `operator.lt` (the default).
    """

    metric_name: MetricName
    best_value: float | None
    performance_is_better: Callable[[Any, Any], bool]

    def __init__(self,
                 metric_name: MetricName,
                 *,
                 threshold_value: float | None = None,
                 performance_is_better: Callable[[Any, Any],
                                                 bool] = operator.lt):
        self.metric_name = metric_name
        self.best_value = threshold_value
        self.performance_is_better = performance_is_better

    def should_checkpoint_now(self, aggregate_metrics: Metrics,
                              central_iteration: int):
        cur_value = get_overall_value(aggregate_metrics[self.metric_name])
        if (self.best_value is None
                or self.performance_is_better(cur_value, self.best_value)):
            self.best_value = cur_value
            return True
        return False

    def should_checkpoint_at_end(self):
        return False


class PolicyBasedModelCheckpointingCallback(TrainingProcessCallback):
    """
    Callback to save model checkpoints after iterations and after
    training, when indicated by `policy`.

    :param model_checkpoint_dir:
        A path to disk for saving the trained model.
        If running on Bolt, this will be a path relative to
        ``ARTIFACT_DIR``.
    :param policy:
        An instance of a `CheckpointPolicy` subclass.

    :param numbered: If true, include the iteration number in each
        checkpoint's path to save all the checkpoints without
        overwriting.
    """

    def __init__(self,
                 model_checkpoint_dir: str,
                 *,
                 checkpoint_policy: CheckpointPolicy,
                 numbered: bool = False):
        if get_ops().distributed.local_rank == 0:
            self.numbered = numbered
            self.checkpoint_policy = checkpoint_policy
            from pfl.internal.platform.selector import get_platform
            self.model_checkpoint_dir_name = model_checkpoint_dir
            if not numbered:
                self.model_checkpoint_dir = get_platform(
                ).create_checkpoint_directories([model_checkpoint_dir])[0]

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: StatefulModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if get_ops().distributed.local_rank == 0:
            if self.checkpoint_policy.should_checkpoint_now(
                    aggregate_metrics, central_iteration):
                if self.numbered:
                    from pfl.internal.platform.selector import get_platform
                    self.model_checkpoint_dir = get_platform(
                    ).create_checkpoint_directories([
                        f'{self.model_checkpoint_dir_name}/'
                        f'{central_iteration:05}'
                    ])[0]
                model.save(self.model_checkpoint_dir)
        return False, Metrics()

    def on_train_end(self, *, model: StatefulModel) -> None:
        if get_ops().distributed.local_rank == 0 and (
                self.checkpoint_policy.should_checkpoint_at_end()):
            if self.numbered:
                from pfl.internal.platform.selector import get_platform
                self.model_checkpoint_dir = get_platform(
                ).create_checkpoint_directories(
                    [f'{self.model_checkpoint_dir_name}/final'])[0]
            model.save(self.model_checkpoint_dir)


class ModelCheckpointingCallback(PolicyBasedModelCheckpointingCallback):
    """
    Callback to save model checkpoints. Note that the model checkpoints
    can also be saved as part of ``RestoreTrainingCallback`` as long as
    the model is ``Saveable`` and provided in the list of saveeables in
    the initialization of the callback.

    :param model_checkpoint_dir:
        A path to disk for saving the trained model. Location
        will be relative to root dir on current platform.
    :param checkpoint_frequency:
        The number of central iterations after which to save a model.
        When zero (the default), the model is saved once after
        training is complete.
    :param numbered: If true, append the iteration number to each
        checkpoint path to save all the checkpoints without
        overwriting.
    """

    def __init__(self,
                 model_checkpoint_dir: str,
                 *,
                 checkpoint_frequency: int = 0,
                 numbered: bool = False):
        super().__init__(model_checkpoint_dir,
                         checkpoint_policy=IterationFrequencyCheckpointPolicy(
                             checkpoint_frequency),
                         numbered=numbered)


class ProfilerCallback(TrainingProcessCallback):
    """
    Profiles the code using Python's profiler, cProfile.

    A profile is a set of statistics that describes how often and for how long
    various parts of a program are executed.

    This callback can be used to independently profile iterations of an
    algorithm, or to profile all iterations of an algorithm together.

    The profile statistics will be saved as an artifact during training. These
    statistics can be read and analysed using `pstats`:

    .. code-block:: python

        import pstats
        stats = pstats.Stats(<profile-stats-filename>)
        stats.sort_stats(*keys)
        stats.print_stats(*restrictions)

    Alternatively, `SnakeViz` can be used to produce a graphical view of the
    profile in the browser.

    :param frequency:
        Controls frequency and duration of profiling. If `frequency` is an
        integer > 0, profiling is performed per-iteration every `frequency`
        central training iterations. If `frequency` is None, a single
        profile is produced covering all central training iterations.
    :param warmup_iterations:
        Commence profiling after this number of central training iterations.
        If `warmup_iterations` > total number of central iterations, no
        profiling will take place.
    :param dir_name:
        Name of directory in which profiles will be saved. Location
        will be relative to root dir on current platform.
    """

    def __init__(self,
                 frequency: Optional[int] = None,
                 warmup_iterations: int = 0,
                 dir_name: str = 'profile'):
        self._profiler = cProfile.Profile()
        self._cprof_dir = self.platform().create_checkpoint_directories(
            [dir_name])[0]
        self._profiler_frequency = frequency
        self._warmup_iterations = warmup_iterations
        self._profiling_commenced = False

    # For mockability
    def platform(self):
        from pfl.internal.platform.selector import get_platform  # pylint: disable=reimported
        return get_platform()

    def _begin_profiling(self) -> None:
        self._profiler.enable()
        self._profiling_commenced = True

    def _end_profiling(self, filename: str) -> None:
        """
        :param filename
            E.g. 'profile-iteration-0.cprof'.
            Name of file to which profile stats will be saved.
        """
        self._profiler.dump_stats(os.path.join(self._cprof_dir, filename))
        self._profiler = cProfile.Profile()

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if self._warmup_iterations == 0:
            self._begin_profiling()
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Commence profiling of one iteration at end of previous central
        iteration.

        `central_iteration` is zero-indexed.

        `warmup_iterations` is the number of central iterations must have completed
        before profiling begins. This means profiling begins at end of
        iteration when `central_iteration == warmup_iterations - 1`.
        """

        if self._profiler_frequency and self._profiling_commenced:
            if (central_iteration -
                    self._warmup_iterations) % self._profiler_frequency == 0:
                self._end_profiling(
                    f'profile-iteration-{central_iteration}.cprof')

            if (central_iteration - 1 -
                    self._warmup_iterations) % self._profiler_frequency == 0:
                self._begin_profiling()

        if central_iteration == (self._warmup_iterations - 1):
            self._begin_profiling()

        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if not self._profiler_frequency and self._profiling_commenced:
            self._end_profiling('profile.cprof')


class AggregateMetricsToDisk(TrainingProcessCallback):
    """
    Callback to write aggregated metrics to disk with a given frequency
    with respect to the number of central iterations.

    :param output_path:
        Path to where the csv file of aggregated metrics should be written
        relative to the root dir on current platform.
    :param frequency:
        Write aggregated metrics to file every ``frequency`` central
        iterations.
        Can be useful to skip iterations where no evaluation is done if that
        is also set at a frequency.
    :param check_existing_file:
        Throw error if ``output_path`` already exists and you don't want to
        overwrite it.
    """

    def __init__(self,
                 output_path: str,
                 frequency: int = 1,
                 decimal_points: int = 6,
                 check_existing_file: bool = False):
        dir_name = os.path.dirname(output_path)
        file_name = os.path.basename(output_path)
        platform_dir = self.platform().create_checkpoint_directories(
            [dir_name])[0]
        output_path = os.path.join(platform_dir, file_name)
        self._frequency = frequency
        self._decimal_points = decimal_points
        self._output_path = output_path
        self._columns_to_index: Dict = OrderedDict()

        if get_ops().distributed.local_rank == 0:
            if check_existing_file:
                assert not os.path.exists(
                    output_path), "File {output_path} already exists"
            self._fp = open(self._output_path, 'w')  # noqa: SIM115

    # For mockability
    def platform(self):
        from pfl.internal.platform.selector import get_platform
        return get_platform()

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if get_ops().distributed.local_rank != 0:
            return Metrics()
        self._columns_to_index = OrderedDict()
        self._columns_to_index['central_iteration'] = 0
        # Empty header, update later.
        self._fp.write('central_iteration,\n')
        return Metrics()

    def _update_header(self):
        """ Replace header with current known columns """
        self._fp.flush()
        self._fp.close()
        with open(self._output_path) as f:
            lines = f.readlines()
        num_columns_before = len(lines[0].split(','))
        lines[0] = ','.join(self._columns_to_index.keys()) + '\n'
        # Append empty values on all rows for new columns.
        for i in range(1, len(lines)):
            lines[i] = (
                lines[i].strip() + ',' *
                (len(self._columns_to_index.keys()) - num_columns_before) +
                '\n')

        with open(self._output_path, 'w') as f:
            f.writelines(lines)
        self._fp = open(self._output_path, 'a')  # noqa: SIM115

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if central_iteration % self._frequency != 0:
            # Don't write to disk this central iteration.
            return False, Metrics()
        if get_ops().distributed.local_rank != 0:
            return False, Metrics()

        raw_metrics = aggregate_metrics.to_simple_dict()
        raw_metrics['central_iteration'] = central_iteration
        columns_changed = False
        for name in raw_metrics:
            if name not in self._columns_to_index:
                next_ix = len(self._columns_to_index)
                self._columns_to_index[name] = next_ix
                columns_changed = True
        if columns_changed:
            self._update_header()

        line = ','.join(
            str(round(raw_metrics[c], self._decimal_points)) if c in
            raw_metrics else '' for c in self._columns_to_index)
        self._fp.write(line + '\n')
        self._fp.flush()
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if get_ops().distributed.local_rank == 0:
            self._fp.flush()
            self._fp.close()


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
            self._found_metric_at_iteration = {
                k: central_iteration
                for k in self._lower_is_better_metric_names +
                self._higher_is_better_metric_names
            }

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


class WandbCallback(TrainingProcessCallback):
    """
    Callback for reporting metrics to Weights&Biases dashboard for comparing
    different PFL runs.
    This callback has basic support for logging metrics. If you seek more
    advanced features from the Wandb API, you should make your own callback.

    See https://wandb.ai/ and https://docs.wandb.ai/ for more information on
    Weights&Biases.

    :param wandb_project_id:
        The name of the project where you're sending the new run. If the
        project is not specified, the run is put in an "Uncategorized" project.
    :param wandb_experiment_name:
        A short display name for this run. Generates a random two-word name
        by default.
    :param wandb_config:
         Optional dictionary (or argparse) of parameters (e.g. hyperparameter
         choices) that are used to tag this run in the Wandb dashboard.
    :param wandb_kwargs:
        Additional keyword args other than ``project``, ``name`` and ``config``
        that you can input to ``wandb.init``, see
        https://docs.wandb.ai/ref/python/init for reference.
    """

    def __init__(self,
                 wandb_project_id: str,
                 wandb_experiment_name: Optional[str] = None,
                 wandb_config=None,
                 **wandb_kwargs):
        self._wandb_kwargs = {
            'project': wandb_project_id,
            'name': wandb_experiment_name,
            'config': wandb_config
        }
        self._wandb_kwargs.update(wandb_kwargs)

    @property
    def wandb(self):
        # Not necessarily installed by default.
        import wandb
        return wandb

    def on_train_begin(self, *, model: ModelType) -> Metrics:
        if get_ops().distributed.global_rank == 0:
            self.wandb.init(**self._wandb_kwargs)
        return Metrics()

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: ModelType, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        """
        Submits metrics of this central iteration to Wandb experiment.
        """
        if get_ops().distributed.global_rank == 0:
            # Wandb package already uses a multithreaded solution
            # to submit log requests to server, such that this
            # call will not be blocking until server responds.
            self.wandb.log(aggregate_metrics.to_simple_dict(),
                           step=central_iteration)
        return False, Metrics()

    def on_train_end(self, *, model: ModelType) -> None:
        if get_ops().distributed.global_rank == 0:
            self.wandb.finish()
