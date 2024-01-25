# Copyright © 2023-2024 Apple Inc.

import pytest

from pfl.callback import TrainingProcessCallback
from pfl.hyperparam.base import HyperParam, NNTrainHyperParams


class LearningRateStepDecay(HyperParam, TrainingProcessCallback):

    def __init__(self, initial, decay_start, decay_period, decay_factor):
        self._value = initial
        self._decay_start = decay_start
        self._decay_period = decay_period
        self._decay_factor = decay_factor

    def after_central_iteration(self, aggregate_metrics, model,
                                central_iteration):
        if central_iteration >= self._decay_start - 1 and (
                central_iteration - self._decay_start +
                1) % self._decay_period == 0:
            self._value *= self._decay_factor

    def value(self):
        return self._value


@pytest.fixture
def adaptive_lr():
    return LearningRateStepDecay(
        initial=1.0,
        decay_start=5,
        decay_period=2,
        decay_factor=0.5,
    )


def test_init_train_hyperparams():
    params = NNTrainHyperParams(local_batch_size=10,
                                local_num_epochs=20,
                                local_learning_rate=0.1,
                                local_max_grad_norm=None,
                                local_num_steps=None)
    assert params.get('local_batch_size') == 10
    assert params.get('local_num_epochs') == 20
    assert params.get('local_learning_rate') == 0.1
    assert params.get('local_max_grad_norm') is None
    assert params.get('local_num_steps') is None


def test_clone_basic():
    params = NNTrainHyperParams(local_batch_size=10,
                                local_num_epochs=20,
                                local_learning_rate=0.1,
                                local_max_grad_norm=None,
                                local_num_steps=None)
    params_clone = params.static_clone(local_max_grad_norm=1.23)
    assert params_clone.get('local_batch_size') == 10
    assert params_clone.get('local_num_epochs') == 20
    assert params_clone.get('local_learning_rate') == 0.1
    assert params_clone.get('local_max_grad_norm') == 1.23
    assert params_clone.get('local_num_steps') is None


def test_adaptive_hyperparams(adaptive_lr):

    def one_iteration(iteration, expected_lr):
        params_clone = params.static_clone(local_max_grad_norm=1.23)
        assert params_clone.get('local_batch_size') == 10
        assert params_clone.get('local_num_epochs') == 20
        assert params_clone.get('local_learning_rate') == expected_lr
        assert params_clone.get('local_max_grad_norm') == 1.23
        assert params_clone.get('local_num_steps') is None
        adaptive_lr.after_central_iteration(
            central_iteration=iteration,
            aggregate_metrics=None,
            model=None,
        )

    params = NNTrainHyperParams(local_batch_size=10,
                                local_num_epochs=20,
                                local_learning_rate=adaptive_lr,
                                local_max_grad_norm=None,
                                local_num_steps=None)
    for iteration in range(5):
        one_iteration(iteration, expected_lr=1.0)
    for iteration in range(5, 7):
        one_iteration(iteration, expected_lr=0.5)
    for iteration in range(7, 9):
        one_iteration(iteration, expected_lr=0.25)


def test_clone_adaptive(adaptive_lr):
    params = NNTrainHyperParams(local_batch_size=10,
                                local_num_epochs=20,
                                local_learning_rate=adaptive_lr,
                                local_max_grad_norm=None,
                                local_num_steps=None)
    params_clone = params.static_clone()
    adaptive_lr.after_central_iteration(
        central_iteration=4,
        aggregate_metrics=None,
        model=None,
    )
    assert adaptive_lr.value() == 0.5
    assert params.get('local_learning_rate') == 0.5
    assert params_clone.get('local_learning_rate') == 1.0
