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


@dataclass
class TrainStepArgs:
    # Common args used by different local training algorithms in PyTorch
    amp_context: Union[torch.amp.autocast, contextlib.AbstractContextManager]
    grad_accumulation_steps: int
    grad_scaler: Optional[torch.cuda.amp.GradScaler]
    max_grad_norm: float
    optimizer_should_update: bool


def get_train_step_args(**kwargs) -> TrainStepArgs:
    return TrainStepArgs(
        amp_context=kwargs.get("amp_context") or contextlib.nullcontext(),
        grad_accumulation_steps=kwargs.get("grad_accumulation_steps", 1),
        grad_scaler=kwargs.get("grad_scaler"),
        max_grad_norm=kwargs.get("max_grad_norm"),
        optimizer_should_update=kwargs.get("optimizer_should_update", True))


class PyTorchCommonBridge(CommonFrameworkBridge[PyTorchModel,
                                                NNTrainHyperParams]):

    @staticmethod
    def save_state(state: object, path: str):
        torch.save(_stats_tensors_to_cpu(state), path)

    @staticmethod
    def load_state(path: str):
        return _stats_tensors_to_device(torch.load(path))
