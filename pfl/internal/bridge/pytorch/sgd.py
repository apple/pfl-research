# Copyright © 2023-2024 Apple Inc.
import torch

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from ..base import SGDFrameworkBridge


def _sgd_train_step(pytorch_model, local_optimizer, raw_data, train_kwargs,
                    max_grad_norm):
    local_optimizer.zero_grad()
    pytorch_model.loss(*raw_data, **train_kwargs).backward()
    # local gradient clipping if local_max_grad_norm is set
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(pytorch_model.parameters(),
                                       max_grad_norm)
    local_optimizer.step()


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
