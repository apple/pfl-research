# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

import torch

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from ..base import FedProxFrameworkBridge


def _proximal_train_step(pytorch_model, local_optimizer, raw_data,
                         train_kwargs, global_weights, mu):
    local_optimizer.zero_grad()
    loss = pytorch_model.loss(*raw_data, **train_kwargs)

    # Add proximal term (Definition 2)
    for name, param in pytorch_model.named_parameters():
        loss += mu / 2 * torch.norm(param - global_weights[name])**2

    loss.backward()
    local_optimizer.step()


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
