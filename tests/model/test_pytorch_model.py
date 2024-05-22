# Copyright © 2023-2024 Apple Inc.
from unittest.mock import Mock

import numpy as np
import pytest

from pfl.hyperparam import NNTrainHyperParams
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.internal.bridge.pytorch.sgd import _sgd_train_step
from pfl.internal.ops import get_pytorch_major_version
from pfl.internal.ops.selector import _internal_reset_framework_module

if get_pytorch_major_version():
    # pylint: disable=ungrouped-imports
    import torch  # type: ignore

    from pfl.internal.ops.pytorch_ops import to_tensor
    _internal_reset_framework_module()


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
class TestPyTorchModel:
    """
    Contains all tests that are unique to PyTorchModel.
    """

    def test_save_and_load_central_optimizer_impl(
            self, pytorch_model_setup,
            check_save_and_load_central_optimizer_impl):
        """
        Test if central optimizer could be save and restored
        """
        pytorch_model_setup.model._central_optimizer = torch.optim.Adam(  # pylint: disable=protected-access
            pytorch_model_setup.model._model.parameters(),  # pylint: disable=protected-access
            lr=1.0)
        check_save_and_load_central_optimizer_impl(pytorch_model_setup)

    @pytest.mark.parametrize('grad_accumulation_steps', [1, 2, 3, 4])
    def test_grad_accumulation(self, grad_accumulation_steps,
                               pytorch_model_setup, user_dataset):
        local_learning_rate = 0.1
        local_num_epochs = 5
        # Get the gradient for 1 backward pass
        per_data_grads = [
            np.asarray([-1., -1., -0., -0.], dtype=np.float32),
            np.asarray([-0., -0., -2., -2.], dtype=np.float32)
        ]
        expected_step_grads = []
        # Expected behavior of gradient accumulation
        accumulated_grads = np.zeros(4, dtype=np.float32)
        for i in range(local_num_epochs):
            for j in range(2):
                accumulated_grads += per_data_grads[j] / grad_accumulation_steps
                if (i * 2 + j + 1) % grad_accumulation_steps == 0:
                    expected_step_grads.append(accumulated_grads)
                    accumulated_grads = np.zeros(4, dtype=np.float32)
        if (local_num_epochs * 2) % grad_accumulation_steps != 0:
            expected_step_grads.append(accumulated_grads)

        parameters = list(pytorch_model_setup.model._model.parameters())
        get_grads = lambda: np.concatenate(
            [p.grad.numpy().flatten() for p in parameters])
        mock_local_optimizer = torch.optim.SGD(parameters, local_learning_rate)
        step_grads = []

        def step_side_effect():
            step_grads.append(get_grads())

        mock_local_optimizer.step = Mock(side_effect=step_side_effect)

        def new_local_optimizer(*args, **kwargs):
            return mock_local_optimizer

        pytorch_model_setup.model.new_local_optimizer = new_local_optimizer
        # This is same as bridges.sgd_bridge().do_sgd, but we want
        # to check the returned metadata as well.
        train_metadata = pytorch_model_setup.model.do_multiple_epochs_of(
            user_dataset,
            NNTrainHyperParams(
                local_learning_rate=local_learning_rate,
                local_num_epochs=local_num_epochs,
                local_batch_size=1,
                grad_accumulation_steps=grad_accumulation_steps),
            _sgd_train_step)

        # Check if optimizer step is called correct number of times
        total_steps = 2 * local_num_epochs
        expected_optimizer_calls = (
            total_steps // grad_accumulation_steps +
            int(total_steps % grad_accumulation_steps != 0))
        assert mock_local_optimizer.step.call_count == expected_optimizer_calls
        assert train_metadata.num_steps == total_steps

        # Check if each step the gradient is accumulated correctly
        assert len(step_grads) == len(expected_step_grads)
        for grads, expected_grads in zip(step_grads, expected_step_grads):
            np.testing.assert_array_equal(grads, expected_grads)


@pytest.fixture(scope='function')
def pytorch_model():
    C = 3
    H = 5
    K = 2
    torch.manual_seed(1)

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(C, H, dtype=torch.float32), torch.nn.ReLU(),
                torch.nn.Linear(H, K))

        def forward(self, inputs, targets):
            outputs = self.layers(inputs)
            diff = outputs - targets
            mse = torch.mean(diff * diff)
            return mse

        @torch.no_grad()
        def metrics(self, inputs, targets):
            return {'loss': self.loss(inputs, targets)}

        def loss(self, inputs, targets):
            return self.forward(to_tensor(inputs), to_tensor(targets))

    yield Model()


@pytest.fixture(scope='function')
def pytorch_dataset():
    np.random.seed(1)
    K = 2
    C = 3
    N = 6
    B = 1

    def gen_dataset(K, C, N, B):
        np.random.seed(1991)
        x = (2 * np.random.randn(1, K, C) +
             np.random.randn(N // K, K, C)).reshape(N, C)
        w = np.random.randn(C, K)
        y = np.matmul(x, w) + 1e-2 * np.random.randn(N, K)
        for i in range(0, N, B):
            yield x[i:i + B], y[i:i + B]

    yield gen_dataset(K, C, N, B)


@pytest.mark.skipif(not get_pytorch_major_version(),
                    reason='PyTorch not installed')
def test_pytorch_model_difference_no_gradient(pytorch_model_setup,
                                              pytorch_dataset, user_dataset):
    try:
        model = pytorch_model_setup.model
        state = model.get_parameters()
        bridges.sgd_bridge().do_sgd(
            model, user_dataset,
            NNTrainHyperParams(local_learning_rate=0.1,
                               local_num_epochs=1,
                               local_batch_size=None))
        statistics = model.get_model_difference(state)
        _, weights = statistics.get_weights()
        for weight in weights:
            assert not weight.requires_grad
    finally:
        _internal_reset_framework_module()
