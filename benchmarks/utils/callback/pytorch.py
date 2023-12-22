# -*- coding: utf-8 -*-

# Copyright © 2023-2024 Apple Inc.
from typing import Tuple

from pfl.metrics import Metrics
from pfl.model.pytorch import PyTorchModel
from pfl.callback import TrainingProcessCallback


def polynomial_lr_lambda(current_step: int, lr_init: float, lr_end: float,
                         num_training_steps: int, num_warmup_steps: int,
                         power: float):
    """ polynomial LR decay schedule, implementation followed:
    https://huggingface.co/transformers/v4.6.0/_modules/transformers/optimization.html#get_polynomial_decay_schedule_with_warmup """
    lr_range = lr_init - lr_end
    decay_steps = num_training_steps - num_warmup_steps
    pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
    decay = lr_range * pct_remaining**power + lr_end
    return decay


class CentralLRDecay(TrainingProcessCallback):
    def __init__(self,
                 init_learning_rate: float,
                 end_learning_rate: float,
                 central_num_iterations: int,
                 num_warmup_iterations: int,
                 linear_warmup: bool = False,
                 decay_power: float = 1.0):
        self._num_warmup_iterations = num_warmup_iterations

        if linear_warmup:
            # Linear warmup over central iterations.
            self._warmup = lambda t: float(t) / float(
                max(1, self._num_warmup_iterations)) * init_learning_rate
        else:
            self._warmup = None

        # Linear decay after warmup iterations.
        self._decay = lambda t: polynomial_lr_lambda(
            t, init_learning_rate, end_learning_rate, central_num_iterations,
            self._num_warmup_iterations, decay_power)

    @staticmethod
    def set_central_lr(model: PyTorchModel, curr_lr):
        for param_group in model._central_optimizer.param_groups:
            param_group['lr'] = curr_lr

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: PyTorchModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:
        if central_iteration < self._num_warmup_iterations:
            if self._warmup is not None:
                curr_lr = self._warmup(central_iteration)
                self.set_central_lr(model, curr_lr)
        else:
            curr_lr = self._decay(central_iteration)
            self.set_central_lr(model, curr_lr)
        return False, Metrics()
