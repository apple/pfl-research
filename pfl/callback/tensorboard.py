# Copyright © 2023-2024 Apple Inc.
import logging
import os
import re
import subprocess
from typing import Optional, Tuple, Union

from pfl.callback.base import TrainingProcessCallback
from pfl.common_types import Population
from pfl.internal.ops.selector import get_default_framework_module as get_ops
from pfl.metrics import Metrics
from pfl.model.base import ModelType

logger = logging.getLogger(name=__name__)


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
