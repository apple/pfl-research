# Copyright © 2023-2024 Apple Inc.
import logging
import os
import uuid
import zipfile
from typing import Callable, Dict, Optional, Tuple

import mlx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers

from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops import mlx_ops
from pfl.internal.ops.selector import get_framework_module, set_framework_module
from pfl.metrics import Metrics, MetricsZero, StringMetricName, Weighted, Zero
from pfl.model.base import StatefulModel
from pfl.stats import MappedVectorStatistics

logger = logging.getLogger(name=__name__)


def _savez(path: str, arrays: Dict[str, mx.array]) -> None:
    """
    Write ``arrays`` to an ``.npz`` file like ``mx.savez(path, **arrays)``,
    but for any number of arrays, where ``mx.savez`` takes at most 1024.
    Each array is written as zip64, so it can be larger than 2 GiB.
    """
    with zipfile.ZipFile(path, 'w') as archive:
        for name, array in arrays.items():
            with archive.open(f'{name}.npy', 'w', force_zip64=True) as member:
                mx.save(member, array)


class MLXModel(StatefulModel):
    """
    :param model:
        An ``mlx.nn.Module`` representing the MLX model to train.
        The module must have two methods defined:
        * loss - `(*user_data) --> loss_value`, where `user_data` is a user's
        dataset unpacked into the call of `loss`, and `loss_value` is the
        numeric value to minimize.
        * metrics - A function `(*user_data) --> <name:metric_value>` where
        `user_data` is the same as in `loss`, and the return value is a
        dictionary where `name` is the name of the metric and `metric_value`
        is an instance of :class:``~pfl.metric.MetricValue`` or a tuple of a
        :class:``~pfl.metric.MetricValue`` and a function that postprocesses
        the metric value for each user.
        The `metrics` method has the signature:

        .. code-block:: python

            Callable[[*mx.array],
                     Dict[str, Union[
                        MetricValue,
                        Tuple[MetricValue,
                              Callable[[MetricValue], MetricValue]
                        ]
                     ]]
                    ]

        :example:

            .. code-block:: python

                # user data looks like this:
                # UserDataset(raw_data=[x,y], eval_kwargs={'eval':True})
                from pfl.metrics import user_average

                def loss(self, x, y, is_eval=False):
                    self.eval() if eval else self.train()
                    return nn.losses.cross_entropy(self(x),
                        y.squeeze(), reduction="mean")

                def metrics(self, x, y, eval=False):
                    self.eval()
                    loss = mx.sum(
                        nn.losses.cross_entropy(self(x), y)).item()
                    num_samples = len(x)

                    return {
                        'per sample loss': Weighted(loss, num_samples),
                        'per user loss': (Weighted(loss, num_samples),
                                          user_average),
                    }

    :param local_optimizer:
        An ``mlx.optimizers.Optimizer`` instance.
        The learning rate of this optimizer will be replaced by other training
        algorithms that uses this model.
    :param central_optimizer:
        An ``mlx.optimizers.Optimizer`` instance, which is used to apply the
        central model updates to the variables.
    :param save_frozen_parameters:
        If ``False``, ``save`` and ``load`` only include the trainable
        parameters, and ``load`` keeps the frozen parameters as they are.
        This avoids writing a large frozen model, e.g. the base model when
        training adapters, to every checkpoint. Only use this if the frozen
        parameters do not change during training.
    """

    set_framework_module(mlx_ops)

    # Checkpoint constants
    _MODEL_CKPT_NAME = "weights.npz"
    _CENTRAL_OPTIMIZER_CKPT_NAME = "central_optimizer.npz"

    def __init__(self,
                 model,
                 local_optimizer,
                 central_optimizer,
                 save_frozen_parameters: bool = True):
        super().__init__()

        self._model = model
        self._local_optimizer = local_optimizer
        self._central_optimizer = central_optimizer
        self._save_frozen_parameters = save_frozen_parameters

        # Calculate this later dynamically in `evaluate`.
        self._allows_distributed_evaluation: Optional[bool] = None

        self._original_values: Dict = {}
        self._model_diff: MappedVectorStatistics = MappedVectorStatistics()

        # To make the MLX grad function cache unique for each model instance.
        self._postfix = str(uuid.uuid4())[:8]

    @property
    def uuid(self):
        return self._postfix

    @property
    def allows_distributed_evaluation(self) -> Optional[bool]:
        return self._allows_distributed_evaluation

    @property
    def mlx_model(self) -> nn.Module:
        return self._model

    @property
    def variable_map(self) -> Dict[str, mx.array]:
        # Need to calculate this every time because can't store
        # references. Not updated in-place in MLX.
        return dict(mlx.utils.tree_flatten(self._model.trainable_parameters()))

    @property
    def central_optimizer_variable_map(
            self) -> Optional[Dict[Tuple[str, str], mx.array]]:
        flat_state = dict(mlx.utils.tree_flatten(
            self._central_optimizer.state))
        del flat_state['step']
        del flat_state['learning_rate']
        return flat_state

    def _reset_local_optimizer(self, learning_rate):
        self._local_optimizer.init(self._model.trainable_parameters())
        self._local_optimizer.state['learning_rate'] = mx.array(
            learning_rate, dtype=mx.float32)

    def save(self, dir_path: str) -> None:
        """
        Save model weights and the central optimizer state to disk.
        The learning rate is not saved, so after ``load`` the central
        optimizer keeps the learning rate it was constructed with, or a
        schedule continues from the restored step.

        :param dir_path:
            Directory on disk to store state.
        """
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        parameters = (self._model.parameters() if self._save_frozen_parameters
                      else self._model.trainable_parameters())
        _savez(os.path.join(dir_path, self._MODEL_CKPT_NAME),
               dict(mlx.utils.tree_flatten(parameters)))
        optimizer_state = dict(
            mlx.utils.tree_flatten(self._central_optimizer.state))
        optimizer_state.pop('learning_rate', None)
        _savez(os.path.join(dir_path, self._CENTRAL_OPTIMIZER_CKPT_NAME),
               optimizer_state)

    def load(self, dir_path: str) -> None:
        """
        Load model weights and central optimizer state from disk, which is
        the state previously saved with ``save``.

        :param dir_path:
            Path to root directory where weights can be loaded from.
            Should be same path as used with ``save``.
        """
        save_path = os.path.join(dir_path, self._MODEL_CKPT_NAME)
        if not os.path.exists(save_path):
            raise CheckpointNotFoundError(save_path)
        weights_path = os.path.join(dir_path, self._MODEL_CKPT_NAME)
        if self._save_frozen_parameters:
            self._model.load_weights(weights_path)
        else:
            weights = mx.load(weights_path)
            weight_shapes = {name: w.shape for name, w in weights.items()}
            trainable_shapes = {
                name: p.shape
                for name, p in mlx.utils.tree_flatten(
                    self._model.trainable_parameters())
            }
            # A non-strict load checks neither names nor shapes.
            if weight_shapes != trainable_shapes:
                raise ValueError(
                    f'{weights_path} does not hold exactly the trainable '
                    'parameters of the model.')
            self._model.load_weights(list(weights.items()), strict=False)
        logger.info(f'Restored model weights from {weights_path}')

        optimizer_path = os.path.join(dir_path,
                                      self._CENTRAL_OPTIMIZER_CKPT_NAME)
        if os.path.exists(optimizer_path):
            optimizer_state = mx.load(optimizer_path)
            # Empty if saved by an earlier version before the first update.
            if optimizer_state:
                # Older MLX versions would otherwise reset the restored
                # state when the optimizer initializes at its first update.
                self._central_optimizer.init(
                    self._model.trainable_parameters())
                self._central_optimizer.state.update(
                    mlx.utils.tree_unflatten(list(optimizer_state.items())))
            logger.info(
                f'Restored central optimizer checkpoint from {optimizer_path}.'
            )
        else:
            logger.info(
                f'No central optimizer checkpoint found at {optimizer_path}.')

    def get_parameters(
        self,
        placeholders: Optional[MappedVectorStatistics] = None
    ) -> MappedVectorStatistics:
        return MappedVectorStatistics({
            k: mx.array(v)
            for k, v in self.variable_map.items()
        })

    def set_parameters(self, w: MappedVectorStatistics) -> None:
        self._model.update(mlx.utils.tree_unflatten(list(w.items())))

    def get_model_difference(self,
                             other_parameters: MappedVectorStatistics,
                             clone: bool = False) -> MappedVectorStatistics:
        model_diff: MappedVectorStatistics = MappedVectorStatistics()
        for variable_name, variable in self.variable_map.items():
            model_diff[
                variable_name] = variable - other_parameters[variable_name]
        return model_diff

    def do_multiple_epochs_of(self, user_dataset: AbstractDatasetType,
                              train_params: NNTrainHyperParams,
                              train_step_fn: Callable, **kwargs) -> None:
        """
        Perform multiple epochs of training. The customizable training
        function that will use a batch of data to update the local
        model state is defined by ``train_step_fn``.
        If you have specified an optimizer using the parameter
        `local_optimizer` in the constructor, the optimizer will
        be reset before training is performed in this method.

        :param user_dataset:
            Dataset of type ``Dataset`` to train on.
        :param train_params:
            An instance of :class:`~pfl.hyperparam.base.NNTrainHyperParams`
            containing configuration for training.
        :param train_step_fn:
            A function with the following arguments:
            * mlx_model - the MLX model object to train on.
            * local_optimizer - the optimizer to use for training.
            * raw_data - an iterable of tensors unpacked into the loss function
            ``mlx_model.loss(*raw_data)``
            * train_kwargs - the ``train_kwargs`` property from the user
            dataset. With this, you can pass user-specific metadata to local
            training.
            * kwargs - other keyword arguments that a custom ``train_step_fn``
            might have.
        """
        num_epochs = (1 if train_params.local_num_epochs is None else
                      train_params.get('local_num_epochs'))
        self._reset_local_optimizer(
            learning_rate=train_params.local_learning_rate)

        for _ in range(num_epochs):
            for batch_ix, batch in enumerate(
                    user_dataset.iter(train_params.get('local_batch_size'))):
                if batch_ix == train_params.get('local_num_steps'):
                    break
                batch = [
                    get_framework_module().to_tensor(data, dtype=None)
                    for data in batch
                ]
                train_step_fn(self._model, self._local_optimizer, batch,
                              user_dataset.train_kwargs, **kwargs)

    def evaluate(self,
                 dataset: AbstractDatasetType,
                 name_formatting_fn: Callable[
                     [str], StringMetricName] = lambda n: StringMetricName(n),
                 eval_params: Optional[NNEvalHyperParams] = None) -> Metrics:
        # Use mini-batches if local_batch_size is set.
        batch_size = (len(dataset) if eval_params is None
                      or eval_params.local_batch_size is None else
                      eval_params.get('local_batch_size'))
        assert isinstance(batch_size, int)
        metrics = Zero

        postprocess_fns = []
        allows_distributed_evaluation = True

        for batch_idx, batch in enumerate(dataset.iter(batch_size)):
            metrics_one_batch = Metrics()
            batch = [
                get_framework_module().to_tensor(data, dtype=None)
                for data in batch
            ]
            for name, metric_value in self._model.metrics(
                    *batch, **dataset.eval_kwargs).items():
                if isinstance(metric_value, tuple):
                    # Is tuple with metric postprocess function as 2nd
                    # argument.
                    metric_value, postprocess_fn = metric_value
                    allows_distributed_evaluation = False
                else:
                    postprocess_fn = lambda x: x
                if batch_idx == 0:
                    # Save for later when postprocessing.
                    postprocess_fns.append(postprocess_fn)

                metrics_one_batch[name_formatting_fn(name)] = metric_value
            metrics += metrics_one_batch

        # Distributed evaluation is only allowed if no postprocess functions
        # are used.
        self._allows_distributed_evaluation = allows_distributed_evaluation
        if isinstance(metrics, MetricsZero):
            raise RuntimeError(  # noqa: TRY004
                f"Accumulated metrics were Zero for user with dataset {dataset}"
            )
        processed_metrics = Metrics([
            (name, postprocess_fn(metric_value))
            for (name,
                 metric_value), postprocess_fn in zip(metrics, postprocess_fns)
        ])

        return processed_metrics

    def apply_model_update(
            self,
            statistics: MappedVectorStatistics) -> Tuple['MLXModel', Metrics]:
        assert isinstance(statistics, MappedVectorStatistics)
        metrics = Metrics()

        parameters = dict(mlx.utils.tree_flatten(self._model.parameters()))
        # Statistics can also hold arrays of other frameworks, e.g. NumPy.
        difference = [
            (name,
             -1 * mlx_ops.to_tensor(update, dtype=parameters[name].dtype))
            for name, update in statistics.items()
        ]
        self._central_optimizer.update(self._model,
                                       mlx.utils.tree_unflatten(difference))

        metrics[StringMetricName('learning rate')] = Weighted.from_unweighted(
            self._central_optimizer.learning_rate.item())

        return self, metrics
