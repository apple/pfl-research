# Copyright © 2023-2024 Apple Inc.
from typing import Dict

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel

from ..base import SGDFrameworkBridge
from .common import clip_norm_and_update, get_train_step_args


def _sgd_train_step(pytorch_model, local_optimizer, raw_data, train_kwargs,
                    **kwargs):
    train_step_args = get_train_step_args(**kwargs)

    with train_step_args.amp_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        else:
            loss = pytorch_model.loss(*raw_data, **train_kwargs)

        # Scale the loss to get the correct scale for the gradients.
        loss /= train_step_args.grad_accumulation_steps

    if train_step_args.grad_scaler is None:
        loss.backward()
    else:
        train_step_args.grad_scaler.scale(loss).backward()

    clip_norm_and_update(pytorch_model, local_optimizer, train_step_args)


class PyTorchSGDBridge(SGDFrameworkBridge[PyTorchModel, NNTrainHyperParams]):
    """
    Concrete PyTorch implementations of utils for stochastic gradient
    descent.
    """

    @staticmethod
    def do_sgd(model: PyTorchModel, user_dataset: AbstractDatasetType,
               train_params: NNTrainHyperParams) -> None:
        model.do_multiple_epochs_of(user_dataset, train_params,
                                    _sgd_train_step)
