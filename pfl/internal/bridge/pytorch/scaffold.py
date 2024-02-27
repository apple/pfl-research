# Copyright © 2023-2024 Apple Inc.
from typing import Dict

from pfl.data.dataset import AbstractDataset
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.model.pytorch import PyTorchModel
from pfl.stats import MappedVectorStatistics

from ..base import SCAFFOLDFrameworkBridge
from .utils import clip_norm_and_update


def _control_variate_train_step(pytorch_model, local_optimizer, raw_data,
                                train_kwargs, train_step_args, **kwargs):
    local_c, server_c = kwargs["local_c"], kwargs["server_c"]

    with train_step_args.amp_context:
        if isinstance(raw_data, Dict):
            loss = pytorch_model.loss(**{**raw_data, **train_kwargs})
        else:
            loss = pytorch_model.loss(*raw_data, **train_kwargs)

        # Scale the loss to get the correct scale for the gradients.
        loss /= train_step_args.grad_accumulation_state.accumulation_steps

    if train_step_args.grad_scaler is None:
        loss.backward()
    else:
        train_step_args.grad_scaler.scale(loss).backward()
    train_step_args.grad_accumulation_state.increment()

    if train_step_args.grad_accumulation_state.optimizer_should_update:
        for name, var in pytorch_model.named_parameters():
            if not var.requires_grad:
                # Frozen variable
                continue
            var.grad.data += server_c[name] - local_c[name]

    clip_norm_and_update(pytorch_model, local_optimizer, train_step_args)


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
