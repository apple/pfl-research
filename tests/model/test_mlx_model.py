# Copyright © 2023-2024 Apple Inc.
import zipfile
from unittest.mock import patch

import numpy as np
import pytest

from pfl.data.dataset import Dataset
from pfl.exception import CheckpointNotFoundError
from pfl.hyperparam import NNTrainHyperParams
from pfl.internal.bridge import FrameworkBridgeFactory as bridges
from pfl.internal.ops.common_ops import check_mlx_installed
from pfl.internal.ops.selector import _internal_reset_framework_module
from pfl.stats import MappedVectorStatistics

if check_mlx_installed():
    # pylint: disable=ungrouped-imports
    import mlx
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    _internal_reset_framework_module()


def _assert_trees_equal(expected, actual):
    expected, actual = dict(tree_flatten(expected)), dict(tree_flatten(actual))
    assert expected.keys() == actual.keys()
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        assert mx.array_equal(actual[name], value).item(), name


@pytest.fixture
def linear_layers_model(mlx_ops):
    from pfl.model.mlx import MLXModel

    def _linear_layers_model(num_layers, dtype=None):
        module = nn.Module()
        module.layers = [nn.Linear(2, 2) for _ in range(num_layers)]
        if dtype is not None:
            module.set_dtype(dtype)
        return MLXModel(module,
                        local_optimizer=optim.SGD(learning_rate=1.0),
                        central_optimizer=optim.Adam(learning_rate=1.0))

    return _linear_layers_model


@pytest.fixture
def one_array_model(mlx_ops):
    from pfl.model.mlx import MLXModel

    def _one_array_model(learning_rate=0.1):
        module = nn.Module()
        module.w = mx.zeros((3, 4))
        return MLXModel(
            module,
            local_optimizer=optim.SGD(learning_rate=1.0),
            central_optimizer=optim.Adam(learning_rate=learning_rate))

    return _one_array_model


@pytest.fixture
def adapter_model(mlx_ops):
    from pfl.model.mlx import MLXModel

    def _adapter_model(base_value, save_frozen_parameters=False):
        module = nn.Module()
        module.base = mx.full((3, 4), base_value)
        module.adapter = mx.zeros((3, 4))
        module.freeze(keys=['base'])
        return MLXModel(module,
                        local_optimizer=optim.SGD(learning_rate=1.0),
                        central_optimizer=optim.Adam(learning_rate=0.1),
                        save_frozen_parameters=save_frozen_parameters)

    return _adapter_model


@pytest.mark.skipif(not check_mlx_installed(), reason='MLX not installed')
class TestMLXModel:
    """
    Contains all tests that are unique to MLXModel.
    """

    def test_save_and_load_central_optimizer_impl(
            self, mlx_model_setup, check_save_and_load_central_optimizer_impl):
        """
        Test if central optimizer could be save and restored
        """
        mlx_model_setup.model._central_optimizer = mlx.optimizers.Adam(  # pylint: disable=protected-access
            learning_rate=1.0)
        check_save_and_load_central_optimizer_impl(mlx_model_setup)

    @pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
    def test_save_and_load_more_than_1024_arrays(self, dtype, tmp_path,
                                                 linear_layers_model):
        # `mx.savez` takes the arrays as keyword arguments, at most 1024.
        dtype = getattr(mx, dtype)
        saved = linear_layers_model(600, dtype)
        saved._central_optimizer.init(saved.mlx_model.trainable_parameters())
        saved.save(str(tmp_path))

        loaded = linear_layers_model(600, dtype)
        loaded.load(str(tmp_path))

        _assert_trees_equal(saved.mlx_model.parameters(),
                            loaded.mlx_model.parameters())
        _assert_trees_equal(saved._central_optimizer.state,
                            loaded._central_optimizer.state)

    def test_save_and_load_array_larger_than_zip64_limit(
            self, tmp_path, linear_layers_model):
        # A lowered limit stands in for an array larger than 2 GiB.
        saved = linear_layers_model(1)
        with patch.object(zipfile, 'ZIP64_LIMIT', 8):
            saved.save(str(tmp_path))

        loaded = linear_layers_model(1)
        loaded.load(str(tmp_path))

        _assert_trees_equal(saved.mlx_model.parameters(),
                            loaded.mlx_model.parameters())

    def test_load_resumes_central_optimizer(self, tmp_path, one_array_model):
        learning_rate = optim.exponential_decay(0.1, 0.5)
        updates = [
            MappedVectorStatistics({'w': mx.full((3, 4), float(i + 1))})
            for i in range(4)
        ]
        uninterrupted = one_array_model(learning_rate)
        for update in updates[:3]:
            uninterrupted.apply_model_update(update)
        uninterrupted.save(str(tmp_path))
        resumed = one_array_model(learning_rate)
        resumed.load(str(tmp_path))

        assert resumed._central_optimizer.step.item() == 3
        for model in (uninterrupted, resumed):
            model.apply_model_update(updates[3])
        # The learning rate schedule and the moments continue.
        assert resumed._central_optimizer.learning_rate.item() == (
            pytest.approx(
                uninterrupted._central_optimizer.learning_rate.item()))
        np.testing.assert_array_equal(np.array(resumed.mlx_model.w),
                                      np.array(uninterrupted.mlx_model.w))

    def test_load_keeps_constructor_learning_rate(self, tmp_path,
                                                  one_array_model):
        one_array_model(learning_rate=0.1).save(str(tmp_path))
        resumed = one_array_model(learning_rate=0.5)
        resumed.load(str(tmp_path))

        assert resumed._central_optimizer.learning_rate.item() == (
            pytest.approx(0.5))

    def test_load_checkpoint_from_earlier_version(self, tmp_path,
                                                  one_array_model):
        # Earlier versions saved the optimizer state flattened, without the
        # step.
        saved = one_array_model()
        saved.apply_model_update(MappedVectorStatistics({'w': mx.ones(
            (3, 4))}))
        saved.mlx_model.save_weights(str(tmp_path / 'weights.npz'))
        mx.savez(str(tmp_path / 'central_optimizer.npz'),
                 **saved.central_optimizer_variable_map)

        resumed = one_array_model()
        resumed.load(str(tmp_path))

        _assert_trees_equal(saved.mlx_model.parameters(),
                            resumed.mlx_model.parameters())
        _assert_trees_equal(saved._central_optimizer.state['w'],
                            resumed._central_optimizer.state['w'])
        assert resumed._central_optimizer.step.item() == 0
        resumed.apply_model_update(
            MappedVectorStatistics({'w': mx.ones((3, 4))}))

    def test_load_empty_optimizer_checkpoint_from_earlier_version(
            self, tmp_path, one_array_model):
        saved = one_array_model()
        saved.mlx_model.save_weights(str(tmp_path / 'weights.npz'))
        mx.savez(str(tmp_path / 'central_optimizer.npz'))

        resumed = one_array_model()
        resumed.load(str(tmp_path))

        assert resumed._central_optimizer.step.item() == 0
        resumed.apply_model_update(
            MappedVectorStatistics({'w': mx.ones((3, 4))}))

    def test_save_and_load_without_frozen_parameters(self, tmp_path,
                                                     adapter_model):
        saved = adapter_model(base_value=1.0)
        saved.apply_model_update(
            MappedVectorStatistics({'adapter': mx.ones((3, 4))}))
        saved.save(str(tmp_path))
        assert mx.load(str(tmp_path / 'weights.npz')).keys() == {'adapter'}

        loaded = adapter_model(base_value=2.0)
        loaded.load(str(tmp_path))

        assert mx.array_equal(loaded.mlx_model.adapter,
                              saved.mlx_model.adapter).item()
        assert mx.array_equal(loaded.mlx_model.base, mx.full((3, 4),
                                                             2.0)).item()
        _assert_trees_equal(saved._central_optimizer.state,
                            loaded._central_optimizer.state)

    def test_load_without_frozen_parameters_rejects_other_parameters(
            self, tmp_path, adapter_model):
        adapter_model(base_value=1.0,
                      save_frozen_parameters=True).save(str(tmp_path))

        with pytest.raises(ValueError, match='trainable parameters'):
            adapter_model(base_value=1.0).load(str(tmp_path))

    @pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
    def test_apply_model_update_with_numpy_statistics(self, dtype,
                                                      one_array_model):
        dtype = getattr(mx, dtype)
        numpy_updated, mlx_updated = one_array_model(), one_array_model()
        for model in (numpy_updated, mlx_updated):
            model.mlx_model.set_dtype(dtype)
        update = np.arange(12, dtype=np.float32).reshape(3, 4)

        numpy_updated.apply_model_update(MappedVectorStatistics({'w': update}))
        mlx_updated.apply_model_update(
            MappedVectorStatistics({'w': mx.array(update, dtype=dtype)}))

        _assert_trees_equal(mlx_updated.mlx_model.parameters(),
                            numpy_updated.mlx_model.parameters())

    def test_load_without_checkpoint_raises(self, tmp_path, one_array_model):
        with pytest.raises(CheckpointNotFoundError):
            one_array_model().load(str(tmp_path))


@pytest.mark.skipif(not check_mlx_installed(), reason='MLX not installed')
class TestMLXSGDBridge:

    @pytest.mark.parametrize('local_max_grad_norm', [None, 0.1])
    def test_do_sgd_with_loss_of_three_inputs(self, mlx_ops,
                                              local_max_grad_norm):
        from pfl.model.mlx import MLXModel

        class WeightedLossModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.w = mx.array([[2., 4.], [3., 5.]])

            def loss(self, x, y, sample_weight):
                errors = mx.sum(mx.abs(mx.matmul(x, self.w) - y), axis=1)
                return mx.mean(sample_weight * errors)

        x = np.array([[1., 0.], [0., 2.]], dtype=np.float32)
        y = np.array([[4., 6.], [8., 12.]], dtype=np.float32)
        sample_weight = np.array([1., 3.], dtype=np.float32)
        model = MLXModel(WeightedLossModel(),
                         local_optimizer=optim.SGD(learning_rate=0.0),
                         central_optimizer=optim.SGD(learning_rate=1.0))
        initial_w = np.array(model.mlx_model.w)

        bridges.sgd_bridge().do_sgd(
            model, Dataset((x, y, sample_weight)),
            NNTrainHyperParams(local_learning_rate=1.0,
                               local_num_epochs=1,
                               local_batch_size=None,
                               local_max_grad_norm=local_max_grad_norm))

        residual_signs = np.sign(x @ initial_w - y)
        expected_grad = (
            x.T @ (sample_weight[:, None] * residual_signs)) / len(x)
        if local_max_grad_norm is not None:
            expected_grad *= min(
                1.0, local_max_grad_norm / np.linalg.norm(expected_grad))
        np.testing.assert_allclose(np.array(model.mlx_model.w),
                                   initial_w - expected_grad,
                                   rtol=1e-6)
