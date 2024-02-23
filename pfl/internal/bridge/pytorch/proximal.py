# Copyright © 2023-2024 Apple Inc.
from typing import Dict

import torch

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from ..base import FedProxFrameworkBridge
from .common import clip_norm_and_update, get_train_step_args


def _proximal_train_step(pytorch_model, local_optimizer, raw_data,
                         train_kwargs, **kwargs):
    global_weights, mu = kwargs["global_weights"], kwargs["mu"]
    train_step_args = get_train_step_args(**kwargs)

    with train_step_args.amp_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        else:
            loss = pytorch_model.loss(*raw_data, **train_kwargs)

        # Add proximal term (Definition 2)
        for name, param in pytorch_model.named_parameters():
            if param.requires_grad:
                loss += mu / 2 * torch.norm(param - global_weights[name])**2

        # Scale the loss to get the correct scale for the gradients.
        loss /= train_step_args.grad_accumulation_steps

    if train_step_args.grad_scaler is None:
        loss.backward()
    else:
        train_step_args.grad_scaler.scale(loss).backward()

    clip_norm_and_update(pytorch_model, local_optimizer, train_step_args)


class PyTorchFedProxBridge(FedProxFrameworkBridge[PyTorchModel,
                                                  NNTrainHyperParams]):
    """
    Concrete implementation of FedProx utilities in PyTorch, used by
    FedProx algorithm.
    """

    @staticmethod
    def do_proximal_sgd(model: PyTorchModel, user_dataset: AbstractDatasetType,
                        train_params: NNTrainHyperParams, mu: float) -> None:
        global_weights = dict(model.get_parameters().items())
        model.do_multiple_epochs_of(user_dataset,
                                    train_params,
                                    _proximal_train_step,
                                    global_weights=global_weights,
                                    mu=mu)
