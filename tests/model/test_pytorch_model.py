# Copyright © 2023-2024 Apple Inc.
import numpy as np
import pytest

from pfl.hyperparam import NNTrainHyperParams
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
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
