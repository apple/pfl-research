# Copyright © 2023-2024 Apple Inc.
from functools import partial
import logging

import mlx
import mlx.core as mx
import mlx.nn as nn

from pfl.data.dataset import AbstractDatasetType
from pfl.hyperparam.base import NNTrainHyperParams
from pfl.internal.ops.mlx_ops import global_norm
from pfl.model.mlx import MLXModel

from ..base import SGDFrameworkBridge

logger = logging.getLogger(__name__)
_mlx_cache = {}

def get_or_make_loss_grad_fn(model_uuid: str, mlx_model: nn.Module, optimizer,
                             fn):
    """
    Lookup gradient function in cache or create it.
    One function per model with unique uuid is created.
    """
    id_ = f'{fn.__name__}-{model_uuid}'
    if id_ not in _mlx_cache:

        # The state that will be captured as input and output
        state = [mlx_model.state, optimizer.state, mx.random.state]
        loss_and_grad_fn = nn.value_and_grad(mlx_model, fn)

        @partial(mx.compile, inputs=state, outputs=state)
        def step(x, y, max_grad_norm):
            loss, grads = loss_and_grad_fn(x, y)
            if max_grad_norm != -1:
                grads = mlx.optimizers.clip_grad_norm(grads, max_grad_norm)[0]
            optimizer.update(mlx_model, grads)
            return loss

        _mlx_cache[id_] = step
        logger.debug(f'Cached new MLX graph {id_}')
    return _mlx_cache[id_]


def _sgd_train_step(mlx_model, local_optimizer, raw_data, train_kwargs,
                    max_grad_norm, model_uuid):
    # local gradient clipping if local_max_grad_norm is set
    state = [mlx_model.state, local_optimizer.state, mx.random.state]
    train_step_fn = get_or_make_loss_grad_fn(model_uuid, mlx_model,
                                             local_optimizer, mlx_model.loss)
    train_step_fn(*raw_data, max_grad_norm or -1)
    mx.eval(state)


class MLXSGDBridge(SGDFrameworkBridge[MLXModel, NNTrainHyperParams]):
    """
    Concrete MLX implementations of utils for stochastic gradient
    descent.
    """

    @staticmethod
    def do_sgd(model: MLXModel, user_dataset: AbstractDatasetType,
               train_params: NNTrainHyperParams) -> None:
        model.do_multiple_epochs_of(
            user_dataset,
            train_params,
            partial(_sgd_train_step, model_uuid=model.uuid),
            max_grad_norm=train_params.local_max_grad_norm)
