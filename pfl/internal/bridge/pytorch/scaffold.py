# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.

from pfl.data.dataset import AbstractDataset
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel
from pfl.stats import MappedVectorStatistics

from ..base import SCAFFOLDFrameworkBridge


def _control_variate_train_step(pytorch_model, local_optimizer, raw_data,
                                train_kwargs, local_c, server_c):
    local_optimizer.zero_grad()
    pytorch_model.loss(*raw_data, **train_kwargs).backward()

    for name, var in pytorch_model.named_parameters():
        if not var.requires_grad:
            # Frozen variable
            continue

        var.grad.data += server_c[name] - local_c[name]
    local_optimizer.step()


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
