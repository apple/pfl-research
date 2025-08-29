# Copyright © 2023-2024 Apple Inc.
from typing import Tuple

import tensorflow as tf

from pfl.callback.base import TrainingProcessCallback
from pfl.metrics import Metrics, StringMetricName
from pfl.model.tensorflow import TFModel


class LocalLRDecay(TrainingProcessCallback):

    def __init__(self, initial_learning_rate, central_num_iterations):
        # Linear warmup over 20 central iterations.
        self._num_warmup_iterations = 20
        self._warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            1e-6,
            self._num_warmup_iterations,
            initial_learning_rate,
            power=1.0)
        # Linear decay to 0 over `central_num_iterations` iterations.
        self._decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate,
            central_num_iterations,
            end_learning_rate=0,
            power=1.0)

    def after_central_iteration(self, central_iteration, config,
                                aggregate_metrics, model):
        if central_iteration < self._num_warmup_iterations:
            config.local_learning_rate = self._warmup_schedule(
                central_iteration).numpy()
        else:
            config.local_learning_rate = self._decay_schedule(
                central_iteration).numpy()
        metrics = Metrics([(StringMetricName('Local learning rate'),
                            config.local_learning_rate)])
        return False, metrics


class CentralLRDecay(TrainingProcessCallback):

    def __init__(self,
                 init_learning_rate: float,
                 end_learning_rate: float,
                 central_num_iterations: int,
                 num_warmup_iterations: int,
                 linear_warmup: bool = False,
                 decay_power: float = 1.0):
        # Linear warmup over 20 central iterations.
        self._num_warmup_iterations = num_warmup_iterations
        if linear_warmup:
            self._warmup_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
                1e-6,
                self._num_warmup_iterations,
                init_learning_rate,
                power=1.0)
        else:
            self._warmup_schedule = None

        # Linear decay to 0 over `central_num_iterations` iterations.
        self._decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            init_learning_rate,
            central_num_iterations,
            end_learning_rate=end_learning_rate,
            power=decay_power)

    @staticmethod
    def set_central_lr(model, curr_lr: float):
        try:
            model._central_optimizer._set_hyper('learning_rate', curr_lr)
        except AttributeError:
            assert hasattr(model._central_optimizer, '_learning_rate')
            model._central_optimizer._learning_rate = curr_lr

    def after_central_iteration(
            self, aggregate_metrics: Metrics, model: TFModel, *,
            central_iteration: int) -> Tuple[bool, Metrics]:

        if central_iteration < self._num_warmup_iterations:
            if self._warmup_schedule is not None:
                curr_lr = self._warmup_schedule(central_iteration).numpy()
                self.set_central_lr(model, curr_lr)
        else:
            curr_lr = self._decay_schedule(central_iteration).numpy()
            self.set_central_lr(model, curr_lr)
        return False, Metrics()
