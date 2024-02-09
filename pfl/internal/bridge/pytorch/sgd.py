# Copyright © 2023-2024 Apple Inc.
import contextlib
from typing import Dict

import torch

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from ..base import SGDFrameworkBridge


def _sgd_train_step(
    pytorch_model,
    local_optimizer,
    raw_data,
    train_kwargs,
    optimizer_should_update,
    max_grad_norm,
    autocast_context=None,
    grad_scaler=None,
):
    if optimizer_should_update:
        local_optimizer.zero_grad()

    if autocast_context is None:
        autocast_context = contextlib.nullcontext()

    with autocast_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        else:
            loss = pytorch_model.loss(*raw_data, **train_kwargs)

    if grad_scaler is None:
        loss.backward()
        if optimizer_should_update:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               max_grad_norm)
            local_optimizer.step()
    else:
        grad_scaler.scale(loss).backward()
        if optimizer_should_update:
            if max_grad_norm is not None:
                grad_scaler.unscale_(local_optimizer)
                torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                               max_grad_norm)
            grad_scaler.step(local_optimizer)
            grad_scaler.update()


class PyTorchSGDBridge(SGDFrameworkBridge[PyTorchModel, NNTrainHyperParams]):
    """
    Concrete PyTorch implementations of utils for stochastic gradient
    descent.
    """

    @staticmethod
    def do_sgd(model: PyTorchModel, user_dataset: AbstractDatasetType,
               train_params: NNTrainHyperParams) -> None:
        model.do_multiple_epochs_of(
            user_dataset,
            train_params,
            _sgd_train_step,
            max_grad_norm=train_params.local_max_grad_norm)
