# Copyright © 2023-2024 Apple Inc.
from typing import Dict

import torch

from pfl.data.dataset import AbstractDataset
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel
from pfl.stats import MappedVectorStatistics

from ..base import SCAFFOLDFrameworkBridge
from .common import get_train_step_args


def _control_variate_train_step(pytorch_model, local_optimizer, raw_data,
                                train_kwargs, **kwargs):
    local_c, server_c = kwargs["local_c"], kwargs["server_c"]

    train_step_args = get_train_step_args(**kwargs)
    if train_step_args.optimizer_should_update:
        local_optimizer.zero_grad()

    with train_step_args.amp_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        else:
            loss = pytorch_model.loss(*raw_data, **train_kwargs)
        loss /= train_step_args.grad_accumulation_steps

    def grad_postprocessing():
        for name, var in pytorch_model.named_parameters():
            if not var.requires_grad:
                # Frozen variable
                continue
            var.grad.data += server_c[name] - local_c[name]

    if train_step_args.grad_scaler is None:
        loss.backward()
        if kwargs.get("optimizer_should_update", True):
            grad_postprocessing()
            if train_step_args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               train_step_args.max_grad_norm)
            local_optimizer.step()
    else:
        train_step_args.grad_scaler.scale(loss).backward()
        if kwargs.get("optimizer_should_update", True):
            grad_postprocessing()
            if train_step_args.max_grad_norm is not None:
                train_step_args.grad_scaler.unscale_(local_optimizer)
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               train_step_args.max_grad_norm)
            train_step_args.grad_scaler.step(local_optimizer)
            train_step_args.grad_scaler.update()


class PyTorchSCAFFOLDBridge(SCAFFOLDFrameworkBridge[PyTorchModel,
                                                    NNTrainHyperParams]):
    """
    Concrete implementation of SCAFFOLD utilities in PyTorch, used by
    SCAFFOLD algorithm.
    """

    @staticmethod
    def do_control_variate_sgd(
        model: PyTorchModel,
        user_dataset: AbstractDataset,
        train_params: NNTrainHyperParams,
        local_c: MappedVectorStatistics,
        server_c: MappedVectorStatistics,
    ) -> None:
        model.do_multiple_epochs_of(user_dataset,
                                    train_params,
                                    _control_variate_train_step,
                                    local_c=local_c,
                                    server_c=server_c)
