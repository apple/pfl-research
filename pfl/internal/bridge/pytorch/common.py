# Copyright © 2023-2024 Apple Inc.
import contextlib
from dataclasses import dataclass
from typing import Optional, Union

import torch

from pfl.hyperparam.base import NNTrainHyperParams
from pfl.internal.ops.pytorch_ops import get_default_device
from pfl.model.pytorch import PyTorchModel
from pfl.stats import TrainingStatistics

from ..base import CommonFrameworkBridge


def _to_cpu(tensor):
    detached_tensor = tensor.detach()
    if tensor.device.type != 'cpu':
        detached_tensor = detached_tensor.cpu()
    return detached_tensor


def _stats_tensors_to_cpu(item):
    if isinstance(item, TrainingStatistics):
        item = item.apply_elementwise(_to_cpu)
    return item


def _stats_tensors_to_device(item):
    if isinstance(item, TrainingStatistics):
        item = item.apply_elementwise(
            lambda t: t.to(device=get_default_device()))
    return item


class GradAccumulationState:
    """ Track gradient accumulation during local training. """

    def __init__(self, train_params: Optional[NNTrainHyperParams],
                 user_data_length: Optional[int]):
        if train_params is not None and user_data_length is not None:
            # Get the total number of steps in local training
            num_epochs = (1 if train_params.local_num_epochs is None else
                          train_params.get('local_num_epochs'))
            local_batch_size = train_params.get('local_batch_size')
            if train_params.get('local_num_steps') is not None:
                num_steps = train_params.get('local_num_steps')
            else:
                num_steps = num_epochs
                if local_batch_size is not None:
                    # Multiply by number of batches per epoch
                    num_steps *= (
                        user_data_length // local_batch_size +
                        int(user_data_length % local_batch_size != 0))
            self._num_steps = num_steps
            self._accumulation_steps = train_params.grad_accumulation_steps
        else:
            self._num_steps = None
            self._accumulation_steps = 1
        self._steps = 0

    @property
    def optimizer_should_update(self) -> bool:
        """ Update every `grad_accumulation_steps` or is the last step """
        return (self._steps % self._accumulation_steps == 0
                or self._steps == self._num_steps)

    @property
    def accumulation_steps(self):
        return self._accumulation_steps

    def increment(self):
        self._steps += 1


@dataclass(frozen=True)
class TrainStepArgs:
    # Common args used by different local training algorithms in PyTorch
    amp_context: Union[torch.amp.autocast, contextlib.AbstractContextManager]
    grad_scaler: Optional[torch.cuda.amp.GradScaler]
    max_grad_norm: Optional[float]
    grad_accumulation_state: GradAccumulationState


def get_train_step_args(**kwargs) -> TrainStepArgs:
    return TrainStepArgs(amp_context=kwargs.get("amp_context")
                         or contextlib.nullcontext(),
                         grad_scaler=kwargs.get("grad_scaler"),
                         max_grad_norm=kwargs.get("max_grad_norm"),
                         grad_accumulation_state=kwargs.get(
                             'grad_accumulation_state',
                             GradAccumulationState(None, None)))


def clip_norm_and_update(pytorch_model, local_optimizer,
                         train_step_args: TrainStepArgs):
    grad_accumulation_state = train_step_args.grad_accumulation_state
    # Clipping the gradients followed by a local optimizer step
    if train_step_args.grad_scaler is None:
        if grad_accumulation_state.optimizer_should_update:
            if train_step_args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               train_step_args.max_grad_norm)
            local_optimizer.step()
    else:
        if grad_accumulation_state.optimizer_should_update:
            if train_step_args.max_grad_norm is not None:
                train_step_args.grad_scaler.unscale_(local_optimizer)
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               train_step_args.max_grad_norm)
            train_step_args.grad_scaler.step(local_optimizer)
            train_step_args.grad_scaler.update()

    if grad_accumulation_state.optimizer_should_update:
        local_optimizer.zero_grad()


class PyTorchCommonBridge(CommonFrameworkBridge[PyTorchModel,
                                                NNTrainHyperParams]):

    @staticmethod
    def save_state(state: object, path: str):
        torch.save(_stats_tensors_to_cpu(state), path)

    @staticmethod
    def load_state(path: str):
        return _stats_tensors_to_device(torch.load(path))
