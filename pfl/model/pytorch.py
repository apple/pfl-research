# Copyright © 2023-2024 Apple Inc.
import contextlib
import inspect
import logging
import os
from typing import Callable, Dict, Optional, Tuple

import torch

from pfl.data.dataset import AbstractDatasetType
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam import NNEvalHyperParams, NNTrainHyperParams
from pfl.internal.ops import pytorch_ops
from pfl.internal.ops.selector import get_framework_module, set_framework_module
from pfl.metrics import Metrics, MetricsZero, StringMetricName, Weighted, Zero
from pfl.model.base import StatefulModel
from pfl.stats import MappedVectorStatistics

logger = logging.getLogger(name=__name__)


class PyTorchModel(StatefulModel):
    """
    :param model:
        A torch.nn.Module representing the pytorch model to train.
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

            Callable[[*torch.Tensor],
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
                l1loss = torch.nn.L1Loss(reduction='sum')

                def metrics(self, x, y, eval=False):
                    self.eval() if eval else self.train()
                    loss = l1loss(self(x), y).item()
                    num_samples = len(x)

                    return {
                        'per sample loss': Weighted(loss, num_samples),
                        'per user loss': (Weighted(loss, num_samples),
                                          user_average),
                    }

    :param local_optimizer_create:
        A function to create a torch.optim.optimizer.Optimizer instance.
        The learning rate of this optimizer will be replaced by other training
        algorithms that uses this trainer.
    :param central_optimizer:
        A torch.optim.optimizer.Optimizer instance, which is used to apply the
        central model updates to the variables.
    """

    set_framework_module(pytorch_ops)

    # Checkpoint constants
    _MODEL_CKPT_NAME = "checkpoint.pt"
    _CENTRAL_OPTIMIZER_CKPT_NAME = "central_optimizer.pt"

    def __init__(self,
                 model,
                 local_optimizer_create,
                 central_optimizer,
                 autocast_float_format: Optional[torch.dtype] = None,
                 grad_scaling: bool = False,
                 model_precision_same_as_autocast: bool = False):
        super().__init__()
        assert hasattr(model, "loss") and hasattr(model, "metrics"), (
            "PyTorch module needs to implement `loss` and `metrics` functions."
        )

        self._model = model.to(pytorch_ops.get_default_device())
        self._local_optimizer_create = local_optimizer_create
        self._central_optimizer = central_optimizer

        # Calculate this later dynamically in `evaluate`.
        self._allows_distributed_evaluation: Optional[bool] = None

        # Calculate the variable mapping once here because the graph will expand
        # later.
        self._variable_map = {
            name: variable
            for name, variable in model.named_parameters()
            if variable.requires_grad
        }

        self._original_values: Dict = {}
        self._model_diff = MappedVectorStatistics()
        self._setup_mixed_precision_training(autocast_float_format,
                                             grad_scaling)
        if model_precision_same_as_autocast and self._autocast_context:
            logger.warning("Casting model weight to dtype: "
                           f"{self._autocast_context.fast_dtype}, "
                           "which may cause training to diverge")
            self._model.type(self._autocast_context.fast_dtype)

    def _setup_mixed_precision_training(
            self, autocast_float_format: Optional[torch.dtype],
            grad_scaling: bool):
        self._autocast_context = None
        self._grad_scaler = None
        if autocast_float_format is not None:
            if pytorch_ops.get_default_device() == torch.device("mps"):
                logger.info("Autocast is disabled on MPS")
                autocast_float_format = torch.float32
            if (pytorch_ops.get_default_device() == torch.device("cpu")
                    and autocast_float_format == torch.float16):
                logger.info("Autocast to float16 is disabled on CPU")
                autocast_float_format = torch.float32
            if torch.cuda.is_available():
                if (autocast_float_format == torch.bfloat16
                        and not torch.cuda.is_bf16_supported()):
                    logger.info(
                        "Autocast to bfloat16 is not supported on this GPU.")
                    autocast_float_format = torch.float32

        if autocast_float_format != torch.float32:
            self._autocast_context = torch.amp.autocast(
                "cuda" if torch.cuda.is_available() else "cpu",
                autocast_float_format)
            logger.info(
                "Mixed precision training with PyTorch AMP float type: "
                f"{self._autocast_context.fast_dtype}")
            if grad_scaling and torch.cuda.is_available():
                self._grad_scaler = torch.cuda.amp.GradScaler()
                logger.info("Gradient scaling enabled.")

    @property
    def allows_distributed_evaluation(self) -> Optional[bool]:
        return self._allows_distributed_evaluation

    @property
    def pytorch_model(self) -> torch.nn.Module:
        return self._model

    @property
    def variable_map(self) -> Dict[str, torch.Tensor]:
        return self._variable_map

    @property
    def central_optimizer_variable_map(
            self) -> Optional[Dict[Tuple[str, str], torch.Tensor]]:
        if len(self._central_optimizer.state) == 0:
            return None

        central_optimizer_variable_map = {}
        for variable_name, variable in self.variable_map.items():
            variable_state = self._central_optimizer.state[variable]
            for state_name, state_variable in variable_state.items():
                if state_name == "step":
                    state_variable = torch.tensor(state_variable)
                central_optimizer_variable_map[variable_name,
                                               state_name] = state_variable
        return central_optimizer_variable_map

    def new_local_optimizer(self, learning_rate,
                            **kwargs) -> torch.optim.Optimizer:
        return self._local_optimizer_create(self._model.parameters(),
                                            lr=learning_rate,
                                            **kwargs)

    def save(self, dir_path: str) -> None:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        torch.save(self._model.state_dict(),
                   os.path.join(dir_path, self._MODEL_CKPT_NAME))
        self._save_central_optimizer(dir_path)

    def load(self, dir_path: str) -> None:
        save_path = os.path.join(dir_path, self._MODEL_CKPT_NAME)
        if not os.path.exists(save_path):
            raise CheckpointNotFoundError(save_path)

        # Use the custom implementation, if there is one.
        self._model.load_state_dict(torch.load(save_path))
        # load central optimizer as well if it exits
        central_optimizer_path = os.path.join(
            dir_path, self._CENTRAL_OPTIMIZER_CKPT_NAME)
        if os.path.exists(central_optimizer_path):
            self._load_central_optimizer(central_optimizer_path)

    def _save_central_optimizer(self, dir_path: str) -> None:
        if self.central_optimizer_variable_map is not None:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            torch.save(
                self.central_optimizer_variable_map,
                os.path.join(dir_path, self._CENTRAL_OPTIMIZER_CKPT_NAME))

    def _load_central_optimizer(self, path: str) -> None:
        # dummy pass to initialize central optimizer variables
        self.apply_model_update(
            MappedVectorStatistics({
                name: torch.zeros(*variable.shape)
                for name, variable in self.variable_map.items()
            }))
        assert self.central_optimizer_variable_map is not None, (
            f"Central optimizer checkpoint is provided at {path} "
            f"but central optimizer has no state")
        central_optimizer_state = torch.load(path,
                                             map_location=torch.device('cpu'))
        device = next(self._model.parameters()).device
        for (variable_name,
             state_name), state_variable in central_optimizer_state.items():
            variable = self.variable_map[variable_name]

            # After PyTorch 1.12.0, step needs to be a PyTorch tensor
            if state_name == "step" and torch.__version__ < '1.12.0':
                state_variable = state_variable.item()
            else:
                state_variable = state_variable.to(device)
            self._central_optimizer.state[variable][
                state_name] = state_variable

    def _set_parameters(self, source_tensors, destination_variables):
        for variable_name, variable in destination_variables.items():
            new_value = source_tensors[variable_name]
            variable.data.copy_(new_value)

    def get_parameters(
        self,
        placeholders: Optional[MappedVectorStatistics] = None
    ) -> MappedVectorStatistics:
        if placeholders is None:
            return MappedVectorStatistics({
                k: v.detach().clone()
                for k, v in self.variable_map.items()
            })
        else:
            self._set_parameters(self.variable_map, placeholders)
            return placeholders

    def set_parameters(self, w: MappedVectorStatistics) -> None:
        self._set_parameters(w, self.variable_map)

    @torch.no_grad()
    def get_model_difference(self,
                             other_parameters: MappedVectorStatistics,
                             clone: bool = False) -> MappedVectorStatistics:
        model_diff: MappedVectorStatistics = MappedVectorStatistics()
        cache = model_diff if clone else self._model_diff
        for variable_name, variable in self.variable_map.items():
            if variable_name not in cache:
                # Up to here, we've used float16 or float32.
                # From here on, we always use float32 for numerical stability in
                # norm clipping and normalization.
                cache[variable_name] = torch.empty(variable.shape,
                                                   dtype=torch.float32,
                                                   device=variable.device)
            model_diff[variable_name] = cache[variable_name].data.copy_(
                variable).sub_(other_parameters[variable_name])
        return model_diff

    @staticmethod
    def _prepare_batch(batch):
        if isinstance(batch, Dict):
            return {
                k: get_framework_module().to_tensor(v)
                for k, v in batch.items()
            }
        else:
            return [get_framework_module().to_tensor(data) for data in batch]

    def do_multiple_epochs_of(self, user_dataset: AbstractDatasetType,
                              train_params: NNTrainHyperParams,
                              train_step_fn: Callable, **kwargs) -> None:
        """
        Perform multiple epochs of training. The customizable training
        function that will use a batch of data to update the local
        model state is defined by ``train_step_fn``.
        If you have specified an optimizer using the parameter
        `local_optimizer_create` in the constructor, a new optimizer will
        be initialized before training is performed in this method.

        :param user_dataset:
            Dataset of type ``Dataset`` to train on.
        :param train_params:
            An instance of :class:`~pfl.hyperparam.base.NNTrainHyperParams`
            containing configuration for training.
        :param train_step_fn:
            A function with the following arguments:
            * pytorch_model - the pytorch model object to train on.
            * local_optimizer - the optimizer to use for training.
            * raw_data - an iterable of tensors unpacked into the loss function
            ``pytorch_model.loss(*raw_data)``
            * train_kwargs - the ``train_kwargs`` property from the user
            dataset. With this, you can pass user-specific metadata to local
            training.
            * kwargs - other keyword arguments that a custom ``train_step_fn``
            might have.
        """
        train_step_signature = inspect.signature(train_step_fn)
        if "autocast_context" in train_step_signature.parameters:
            kwargs["autocast_context"] = self._autocast_context
        if "grad_scaler" in train_step_signature.parameters:
            kwargs["grad_scaler"] = self._grad_scaler

        num_epochs = (1 if train_params.local_num_epochs is None else
                      train_params.get('local_num_epochs'))
        local_optimizer = self.new_local_optimizer(
            learning_rate=train_params.local_learning_rate)

        steps = 0
        local_optimizer.zero_grad()
        for _ in range(num_epochs):
            for _batch_ix, batch in enumerate(
                    user_dataset.iter(train_params.get('local_batch_size'))):
                if steps == train_params.get('local_num_steps'):
                    break
                steps += 1
                batch = self._prepare_batch(batch)
                kwargs['optimizer_should_update'] = (
                    steps % train_params.grad_accumulation_steps == 0)
                train_step_fn(self._model, local_optimizer, batch,
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
        autocast_context = self._autocast_context or contextlib.nullcontext()
        for batch_ix, batch in enumerate(dataset.iter(batch_size)):
            metrics_one_batch = Metrics()
            batch = self._prepare_batch(batch)
            with autocast_context:
                if isinstance(batch, Dict):
                    metrics_outputs = self._model.metrics(**{
                        **batch,
                        **dataset.eval_kwargs
                    })
                else:
                    metrics_outputs = self._model.metrics(
                        *batch, **dataset.eval_kwargs)

            for name, metric_value in metrics_outputs.items():
                if isinstance(metric_value, tuple):
                    # Is tuple with metric postprocess function as 2nd
                    # argument.
                    metric_value, postprocess_fn = metric_value
                    allows_distributed_evaluation = False
                else:
                    postprocess_fn = lambda x: x
                if batch_ix == 0:
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
            self, statistics: MappedVectorStatistics
    ) -> Tuple['PyTorchModel', Metrics]:
        assert isinstance(statistics, MappedVectorStatistics)
        metrics = Metrics()

        self._central_optimizer.zero_grad()
        for variable_name, difference in statistics.items():
            if self.variable_map[variable_name].grad is None:
                self.variable_map[variable_name].grad = torch.zeros_like(
                    self.variable_map[variable_name])
            # Interpret the model updates as gradients.

            self.variable_map[
                variable_name].grad.data.copy_(  # type: ignore[union-attr]
                    -1 * pytorch_ops.to_tensor(difference))

        self._central_optimizer.step()

        metrics[StringMetricName('learning rate')] = Weighted.from_unweighted(
            self._central_optimizer.param_groups[0]['lr'])

        return self, metrics
