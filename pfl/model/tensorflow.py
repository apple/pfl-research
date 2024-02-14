# Copyright © 2023-2024 Apple Inc.

import logging
import os
import uuid
from typing import Callable, Dict, Optional, Tuple, Union

import tensorflow as tf  # type: ignore

from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops import get_tf_major_version, tensorflow_ops
from pfl.internal.ops.selector import set_framework_module
from pfl.metrics import Metrics, MetricValueType, StringMetricName, Weighted, Zero
from pfl.model.base import StatefulModel
from pfl.stats import MappedVectorStatistics

logger = logging.getLogger(name=__name__)

KerasMetricSpec = Union[tf.keras.metrics.Metric,
                        Tuple[tf.keras.metrics.Metric,
                              Callable[[MetricValueType], MetricValueType]]]

_INPUTS_LABELS_ASSERT_MSG = (
    'Tensorflow Keras requires user Dataset to have 2 '
    'elements. The first one is the input tensor or a list of multiple '
    'input tensors. The second element is the label tensor or a list '
    'of multiple label tensors if there are multiple outputs of the '
    'model.')


class TFModel(StatefulModel):
    """
    :param model:
        A Tensorflow Keras model to train. Can either be defined from the
        functional API or the sequential API.
        If the model is a `Sequential`, it needs to be build with
        `model.build(input_dims)`.
    :param metrics:
        Specify metrics to use for evaluation. The key is the name of the
        metric and the value is either a ``tf.keras.metrics.Metric`` or a tuple
        of a ``tf.keras.metrics.Metric`` and a function that postprocesses the
        metric value for each user.

        :example:

            .. code-block:: python

                from pfl.metrics import user_average
                metrics = {
                    'overall accuracy': tf.keras.metrics.Accuracy(),
                    'per-user accuracy': (tf.keras.metrics.Accuracy(),
                                          user_average)
                }

    :param central_optimizer:
        An optimizer instance from `tf.keras.optimizers`, which is used to apply
        the aggregated model updates to the variables.
        Learning rate decay can be applied using
        ``tf.keras.optimizers.schedules``.
    :param checkpoint_format_hdf5:
        If `True`, save model checkpoints as hdf5 files. Otherwise,
        save model checkpoints in tensorflow format.
    """

    set_framework_module(tensorflow_ops)

    # Checkpoint constants
    _MODEL_H5_NAME = "model.h5"
    _MODEL_CKPT_NAME = "model.ckpt"
    _CENTRAL_OPTIMIZER_CKPT_NAME = "central_optimizer.ckpt"

    def __init__(self,
                 model,
                 metrics: Dict[str, KerasMetricSpec],
                 central_optimizer,
                 checkpoint_format_hdf5=False):
        super().__init__()

        assert get_tf_major_version(
        ) > 1, "TFModel requires TensorFlow v2 or above."

        assert model.variables, (
            "Looks like your model does not have any variables. "
            "If you are using `Sequential`, you need to build it with "
            "`.build()` before the trainer can handle the model.")

        # Allow distributed central evaluation as long as
        # no user metric postprocess function is used.
        self._allows_distributed_evaluation = not any(
            isinstance(spec, tuple) for spec in metrics.values())

        # Calculate the variable mapping once here because the graph will expand
        # later.
        self._variable_map = {
            variable.name: variable
            for variable in model.trainable_variables
        }
        self._model = model
        self._metrics = metrics
        self._central_optimizer = central_optimizer
        self._checkpoint_format_hdf5 = checkpoint_format_hdf5
        self._central_optimizer_saver = None
        if isinstance(model.loss, str):
            model.loss = tf.keras.losses.get(model.loss)
        # To make the TF graph cache unique for each model instance.
        self._postfix = str(uuid.uuid4())[:8]

    @property
    def uuid(self):
        return self._postfix

    @property
    def keras_model(self):
        return self._model

    @property
    def allows_distributed_evaluation(self) -> Optional[bool]:
        return self._allows_distributed_evaluation

    @property
    def variable_map(self) -> Dict[str, tf.Variable]:
        return self._variable_map

    @property
    def central_optimizer_variable_map(
            self) -> Optional[Dict[str, tf.Variable]]:
        if len(self._central_optimizer.variables()) == 0:
            return None
        return {
            variable.name: variable
            for variable in self._central_optimizer.variables()
        }

    def save(self, dir_path: str) -> None:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)

        if self._checkpoint_format_hdf5:
            # Write HDF5 checkpoint to disk.
            self._model.save(os.path.join(dir_path, self._MODEL_H5_NAME))
        else:
            # Write TensorFlow checkpoint to disk.
            self._model.save_weights(
                os.path.join(dir_path, self._MODEL_CKPT_NAME))
        self._save_central_optimizer(dir_path)

    def load(self, dir_path: str) -> None:
        if self._checkpoint_format_hdf5:
            # Load HDF5 checkpoint from disk.
            save_path = os.path.join(dir_path, self._MODEL_H5_NAME)
            if not os.path.exists(save_path):
                raise CheckpointNotFoundError(save_path)
        else:
            # Load TensorFlow checkpoint from disk.
            save_path = os.path.join(dir_path, self._MODEL_CKPT_NAME)
            if len(tf.io.gfile.glob(save_path + '.index')) == 0:
                raise CheckpointNotFoundError(save_path)

        self._model.load_weights(save_path)
        # load central optimizer as well if it exits
        central_optimizer_prefix = os.path.join(
            dir_path, self._CENTRAL_OPTIMIZER_CKPT_NAME)
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpoint_management.py#L320  # pylint: disable=line-too-long
        if len(tf.io.gfile.glob(central_optimizer_prefix + '.index')) > 0:
            self._load_central_optimizer(central_optimizer_prefix)

    def _get_or_create_central_optimizer_saver(
            self) -> Optional[tf.train.Checkpoint]:
        """
        Get or create a `tf.train.Checkpoint` for central optimizer states.
        Return `None` for stateless optimizer (e.g. SGD)
        """
        if self._central_optimizer_saver is None:
            # create a saver if there are variables from central optimizer
            if self.central_optimizer_variable_map is not None and len(
                    self.central_optimizer_variable_map) > 0:
                self._central_optimizer_saver = tf.train.Checkpoint(
                    **self.central_optimizer_variable_map)
        return self._central_optimizer_saver

    def _save_central_optimizer(self, dir_path: str) -> None:
        saver = self._get_or_create_central_optimizer_saver()
        if saver is not None:
            # use `write` instead of `save` to avoid saving multiple copies
            saver.write(
                os.path.join(dir_path, self._CENTRAL_OPTIMIZER_CKPT_NAME))

    def _load_central_optimizer(self, path: str) -> None:
        # dummy pass to initialize central optimizer variables
        self.apply_model_update(
            MappedVectorStatistics({
                name: tf.zeros_like(variable)
                for name, variable in self.variable_map.items()
            }))
        saver = self._get_or_create_central_optimizer_saver()
        assert saver is not None, (
            f"Central optimizer checkpoint is provided at {path} "
            f"but saver cannot be created")
        saver.restore(path)

    @tensorflow_ops.tf_function
    def _set_parameters(self, source_tensors, destination_variables):
        for source, destination in zip(source_tensors, destination_variables):
            destination.assign(source)

    @tensorflow_ops.tf_function
    def _set_model_parameters(self, source_tensors):
        self._set_parameters(source_tensors, list(self.variable_map.values()))

    def get_parameters(
        self,
        placeholders: Optional[MappedVectorStatistics] = None
    ) -> MappedVectorStatistics:
        if placeholders is None:
            parameters = {
                variable_name:
                tensorflow_ops.clone_variable(variable, name="state")
                for variable_name, variable in self.variable_map.items()
            }

            return MappedVectorStatistics(parameters)
        else:
            source_tensors = list(self.variable_map.values())
            destination_variables = [
                placeholders[name] for name in self.variable_map
            ]
            tensorflow_ops.try_cached_call(
                self._set_parameters, f'get_set_parameters-{self._postfix}',
                source_tensors, destination_variables)
            return placeholders

    def set_parameters(self, w: MappedVectorStatistics) -> None:
        tensorflow_ops.try_cached_call(self._set_parameters,
                                       f'set_parameters-{self._postfix}',
                                       [w[name] for name in self.variable_map],
                                       list(self.variable_map.values()))

    @tensorflow_ops.tf_function
    def _get_model_difference(self, other_parameters, model_variable_map):
        model_diff = {}
        for original, (variable_name,
                       variable) in zip(other_parameters,
                                        model_variable_map.items()):
            model_diff[variable_name] = (variable - original)
        return model_diff

    def get_model_difference(self,
                             other_parameters: MappedVectorStatistics,
                             clone: bool = False) -> MappedVectorStatistics:
        # There is no caching, hence `clone` parameter has no effect.
        return MappedVectorStatistics(
            tensorflow_ops.try_cached_call(
                self._get_model_difference,
                f'get_model_difference-{self._postfix}',
                [other_parameters[name]
                 for name in self.variable_map], self.variable_map))

    @tensorflow_ops.tf_function(experimental_relax_shapes=True)
    def _forward_prop(self, inputs, training):
        return self._model(inputs, training=training)

    @tensorflow_ops.tf_function
    def _reset_local_optimizer(self, optimizer, learning_rate):
        # Reset the variables of the optimizer to all zeros.
        for state in optimizer.variables():
            state.assign(tf.zeros_like(state))
        # Override the learning rate.
        optimizer.lr.assign(learning_rate)

    def do_multiple_epochs_of(self, user_dataset: AbstractDatasetType,
                              train_params: NNTrainHyperParams,
                              train_step_fn: Callable, **kwargs) -> None:
        """
        Perform multiple epochs of training. The customizable training
        function that will use a batch of data to update the local
        model state is defined by ``train_step_fn``.
        If you have specified an optimizer using the parameter
        `local_optimizer_create` in the constructor, the optimizer state will
        be reset before training is performed in this method.

        :param user_dataset:
            Dataset of type ``Dataset`` to train on.
        :param train_params:
            An instance of :class:`~pfl.hyperparam.base.NNTrainHyperParams`
            containing configuration for training.
        :param train_step_fn:
            A function with the following arguments:
            * inputs - the tensor(s) used as input to the Keras model's
            __call__ method.
            * labels - the label tensor(s) used as the first argument to the
            Keras model's loss function.
            * train_kwargs - the ``train_kwargs`` property from the user
            dataset. With this, you can pass user-specific metadata to local
            training.
            * kwargs - other keyword arguments that a custom ``train_step_fn``
            might have.
            Notice that the TF model itself is not passed in the arguments,
            it needs to instead be in the closure of the function when it is
            defined. This is much more performant.
        """
        self._reset_local_optimizer(self._model.optimizer,
                                    train_params.local_learning_rate)
        num_epochs = (1 if train_params.local_num_epochs is None else
                      train_params.get('local_num_epochs'))

        assert train_params.grad_accumulation_steps == 1, (
            "Gradient accumulation is not yet supported in TensorFlow")

        for _ in range(num_epochs):
            for batch_ix, batch in enumerate(
                    user_dataset.iter(train_params.get('local_batch_size'))):
                if batch_ix == train_params.local_num_steps:
                    break
                assert len(batch) == 2, _INPUTS_LABELS_ASSERT_MSG
                tensorflow_ops.try_cached_call(train_step_fn,
                                               f'train_step-{self._postfix}',
                                               *batch,
                                               user_dataset.train_kwargs,
                                               **kwargs)

    def evaluate(self,
                 dataset: AbstractDatasetType,
                 name_formatting_fn=lambda n: StringMetricName(n),
                 eval_params: Optional[NNEvalHyperParams] = None) -> Metrics:
        metrics = Zero

        batch_size = (None if eval_params is None else
                      eval_params.get('local_batch_size'))
        for batch_inputs, batch_labels in dataset.iter(batch_size):
            preds = tensorflow_ops.try_cached_call(
                self._forward_prop,
                f'evaluate_forward_prop-{self._postfix}',
                batch_inputs,
                training=False)

            metrics_one_user = Metrics()
            for raw_name, keras_metric in self._metrics.items():
                if isinstance(keras_metric, tuple):
                    # Is tuple with metric postprocess function as 2nd argument.
                    keras_metric, _ = keras_metric

                metrics_one_user[name_formatting_fn(
                    raw_name)] = tensorflow_ops.KerasMetricValue(
                        keras_metric, batch_labels, preds)
            metrics += metrics_one_user

        processed_metrics = Metrics()
        for raw_name, keras_metric in self._metrics.items():
            name = name_formatting_fn(raw_name)
            if isinstance(keras_metric, tuple):
                _, postprocess = keras_metric
                # Do any per-user postprocessing.
                processed_metrics[name] = postprocess(metrics[name])
            else:
                # Has no postprocess function.
                processed_metrics[name] = metrics[name]

        return processed_metrics

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['TFModel', Metrics]:
        assert isinstance(statistics, MappedVectorStatistics)
        metrics = Metrics()

        # Construct the list of `(statistics, tensor)` pairs that the global
        # optimizer needs to apply a model update.
        model_updates_variables_pairs = []
        for variable_name, difference in statistics.items():
            variable = self.variable_map[variable_name]
            model_updates_variables_pairs.append((-1 * difference, variable))

        self._central_optimizer.apply_gradients(model_updates_variables_pairs)

        lr_or_dict = self._central_optimizer.get_config()['learning_rate']
        if not isinstance(lr_or_dict, dict):
            metrics[StringMetricName('learning rate')] = (
                Weighted.from_unweighted(lr_or_dict))
        else:
            # Uses a learning rate schedule.
            # TODO: find the learning rate for this case.
            pass

        return self, metrics
